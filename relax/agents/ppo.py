from functools import partial
from typing import Tuple, Dict

import distrax
import einops
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import rlax
from chex import PRNGKey, ArrayTree
from dm_env import StepType
from ml_collections import ConfigDict

from relax.agents.agent import Agent, TrainState
from relax.agents.utils import register
from relax.environments.environment import Environment
from relax.math import explained_variance
from relax.networks import get_actor_critic_model_fn, get_network_build_fn
from relax.rollouts import rollout_truncated, SampleBatch
from relax.typing import Metrics, Observation, Action

_DEFAULT_PPO_CONFIG = ConfigDict(
    {
        "rollout_fragment_length": 128,
        "lambda_": 0.95,
        "gamma": 0.99,
        "initial_learning_rate": 2.5e-4,
        "final_learning_rate": 1e-8,
        "learning_rate_decay_steps": 10000,
        "max_grad_norm": 0.5,
        "minibatch_size": 32,
        "num_epochs": 4,
        "clip": 0.2,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "network": "mlp",
        "network_config": {"hidden_units": [64, 64]},
    }
)


@register("PPO")
class PPO(Agent):
    def __init__(self, env: Environment, config: ConfigDict):
        super().__init__(env, config)

        network_build_fn = get_network_build_fn(self.config.network, **self.config.network_config)
        model_fn = get_actor_critic_model_fn(
            env.action_space, network_build_fn, value_network="copy", free_log_std=True
        )
        self._model = hk.transform(model_fn)

        def policy_fn(params: hk.Params, key: PRNGKey, observation: Observation) -> Tuple[Action, Dict[str, ArrayTree]]:
            model_key, sample_key = jax.random.split(key)

            pi, values = self._model.apply(params, model_key, observation)
            action, action_logp = pi.sample_and_log_prob(seed=sample_key)

            return action, {
                SampleBatch.ACTION_LOGP: action_logp,
                SampleBatch.VALUE: values,
            }

        self._rollout_fn = partial(rollout_truncated, env, self.config.rollout_fragment_length, policy_fn)

        self._lr_schedule = optax.linear_schedule(
            self.config.initial_learning_rate, self.config.final_learning_rate, self.config.learning_rate_decay_steps
        )
        self._optimizer = optax.chain(
            optax.scale_by_adam(eps=1e-5),
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.scale_by_schedule(self._lr_schedule),
            optax.scale(-1.0),
        )

    @staticmethod
    def default_config() -> ConfigDict:
        config = super(PPO, PPO).default_config()
        config.update(_DEFAULT_PPO_CONFIG)
        config.lock()
        return config

    def compute_action(
        self, params: hk.Params, key: PRNGKey, observation: Observation, deterministic: bool = True
    ) -> Action:
        pi, _ = self._model.apply(params, key, observation)

        if deterministic:
            if isinstance(pi, distrax.Categorical):
                return pi.mode()

            return pi.mean()
        else:
            return pi.sample(key)

    def initial_train_state(self, consistent_key: PRNGKey, divergent_key: PRNGKey) -> TrainState:
        train_state_key, env_key = jax.random.split(divergent_key)
        sample_obs = jnp.expand_dims(self.env.observation_space.sample(consistent_key), 0)
        params = self._model.init(consistent_key, sample_obs)
        time_step, env_state = self.env.reset(env_key)
        opt_state = self._optimizer.init(params)

        return TrainState.initial(train_state_key, params, opt_state, time_step, env_state)

    def _loss(self, params, key, obs, actions, advantages, returns, old_log_probs, clip):
        pi, value_preds = self._model.apply(params, key, obs)
        log_probs = pi.log_prob(actions)
        entropy = jnp.mean(pi.entropy())
        ratio = jnp.exp(log_probs - old_log_probs)
        pg_loss_unclipped = -advantages * ratio
        pg_loss_clipped = -advantages * jnp.clip(ratio, 1 - clip, 1 + clip)
        policy_loss = jnp.mean(jnp.maximum(pg_loss_unclipped, pg_loss_clipped))
        value_loss = 0.5 * jnp.mean((returns - value_preds) ** 2)
        loss = policy_loss - entropy * self.config.entropy_coef + value_loss * self.config.value_coef

        clip_frac = jnp.mean(jnp.greater(jnp.abs(ratio - 1.0), clip).astype(obs.dtype))

        info = {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "policy_entropy": entropy,
            "value_explained_variance": explained_variance(returns, value_preds),
            "clip_frac": clip_frac,
        }

        return loss, info

    def _update(self, params, opt_state, key, obs, actions, advantages, returns, old_log_probs, clip):
        grads, info = jax.grad(self._loss, has_aux=True)(
            params,
            key,
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(actions),
            jax.lax.stop_gradient(advantages),
            jax.lax.stop_gradient(returns),
            jax.lax.stop_gradient(old_log_probs),
            clip,
        )

        if self.config.num_envs_per_device > 1:
            grads, info = jax.lax.pmean((grads, info), axis_name=self.batch_axis)

        if self.config.num_devices > 1:
            grads, info = jax.lax.pmean((grads, info), axis_name=self.device_axis)

        updates, new_opt_state = self._optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, info

    def train_step(self, train_state: TrainState) -> Tuple[TrainState, Metrics]:
        next_train_state_key, rollout_key, initial_update_key = jax.random.split(train_state.key, 3)

        rollout_data, _, new_time_step, new_env_state = self._rollout_fn(
            train_state.params,
            train_state.key,
            train_state.time_step,
            train_state.env_state,
        )

        _, final_value = self._model.apply(
            train_state.params,
            rollout_key,
            rollout_data[SampleBatch.NEXT_OBSERVATION][-1],
        )
        values = jnp.concatenate([rollout_data[SampleBatch.VALUE], jnp.expand_dims(final_value, axis=0)], axis=0)

        rollout_data[SampleBatch.RETURN] = rlax.discounted_returns(
            rollout_data[SampleBatch.REWARD],
            rollout_data[SampleBatch.DISCOUNT] * self.config.gamma,
            v_t=0.0,
        )

        rollout_data[SampleBatch.ADVANTAGE] = rlax.truncated_generalized_advantage_estimation(
            rollout_data[SampleBatch.REWARD],
            rollout_data[SampleBatch.DISCOUNT] * self.config.gamma,
            self.config.lambda_,
            values,
        )

        def update_scan_fn(carry, minibatch):
            params, opt_state, key = carry

            next_key, update_key = jax.random.split(key)

            new_params, new_opt_state, info = self._update(
                params,
                opt_state,
                update_key,
                minibatch[SampleBatch.OBSERVATION],
                minibatch[SampleBatch.ACTION],
                minibatch[SampleBatch.ADVANTAGE],
                minibatch[SampleBatch.RETURN],
                minibatch[SampleBatch.ACTION_LOGP],
                self.config.clip,
            )

            return (new_params, new_opt_state, next_key), info

        def epoch_scan_fn(carry, _):
            params, opt_state, key = carry

            next_key, shuffle_key = jax.random.split(key)

            minibatches = jax.tree_map(
                lambda x: einops.rearrange(
                    jax.random.permutation(shuffle_key, x, independent=True),
                    "(n m) ... -> n m ...",
                    n=self.config.rollout_fragment_length // self.config.minibatch_size,
                    m=self.config.minibatch_size,
                ),
                rollout_data,
            )

            return jax.lax.scan(update_scan_fn, (params, opt_state, next_key), minibatches)

        (new_params, new_opt_state, _), metrics = jax.lax.scan(
            epoch_scan_fn,
            (train_state.params, train_state.opt_state, initial_update_key),
            None,
            length=self.config.num_epochs,
        )

        metrics = jax.tree_map(jnp.mean, metrics)

        incremental_episodes = jnp.sum(rollout_data[SampleBatch.STEP_TYPE] == StepType.LAST)

        metrics["min_episode_reward"] = jnp.min(
            jnp.where(
                rollout_data[SampleBatch.STEP_TYPE] == StepType.LAST, rollout_data[SampleBatch.EPISODE_REWARD], jnp.inf
            )
        )
        metrics["mean_episode_reward"] = (
            jnp.sum(rollout_data[SampleBatch.EPISODE_REWARD] * rollout_data[SampleBatch.STEP_TYPE] == StepType.LAST)
            / incremental_episodes
        )
        metrics["max_episode_reward"] = jnp.max(
            jnp.where(
                rollout_data[SampleBatch.STEP_TYPE] == StepType.LAST, rollout_data[SampleBatch.EPISODE_REWARD], -jnp.inf
            )
        )

        incremental_timesteps = (
            self.config.rollout_fragment_length * self.config.num_envs_per_device * self.config.num_devices
        )

        if self.config.num_envs_per_device > 1:
            incremental_episodes = jax.lax.psum(incremental_episodes, axis_name=self.batch_axis)

        if self.config.num_devices > 1:
            incremental_episodes = jax.lax.psum(incremental_episodes, axis_name=self.device_axis)

        next_train_state = train_state.update(
            new_key=next_train_state_key,
            incremental_timesteps=incremental_timesteps,
            incremental_episodes=incremental_episodes,
            new_params=new_params,
            new_opt_state=new_opt_state,
            new_time_step=new_time_step,
            new_env_state=new_env_state,
        )

        return next_train_state, metrics
