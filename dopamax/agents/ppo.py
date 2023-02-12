from functools import partial
from typing import Tuple, Dict

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import rlax
from chex import PRNGKey, ArrayTree
from dm_env import StepType
from ml_collections import ConfigDict

from dopamax.agents.agent import Agent, TrainState
from dopamax.agents.utils import register
from dopamax.environments.environment import Environment
from dopamax.math import explained_variance
from dopamax.networks import get_actor_critic_model_fn, get_network_build_fn
from dopamax.rollouts import rollout_truncated, SampleBatch, create_minibatches
from dopamax.typing import Metrics, Observation, Action

_DEFAULT_PPO_CONFIG = ConfigDict(
    {
        # The number of steps to collect in each rollout fragment.
        "rollout_fragment_length": 128,
        # The GAE lambda parameter.
        "lambda_": 0.95,
        # The agent's discount factor.
        "gamma": 0.99,
        # The initial learning rate.
        "initial_learning_rate": 2.5e-4,
        # The final learning rate.
        "final_learning_rate": 2.5e-4,
        # The number of steps over which to linearly decay the learning rate. Note that this is not the number of train
        # iterations, but the number of gradient updates.
        "learning_rate_decay_steps": 1,
        # The maximum gradient norm.
        "max_grad_norm": 0.5,
        # The size of the minibatches to perform gradient updates on. This should be a factor of the rollout fragment
        # length.
        "minibatch_size": 32,
        # The number of epochs to perform gradient updates for at each train step.
        "num_epochs": 4,
        # The PPO clip parameter.
        "clip": 0.2,
        # The coefficient for the entropy bonus.
        "entropy_coef": 0.01,
        # The coefficient for the value loss.
        "value_coef": 0.5,
        # The type of network to use.
        "network": "mlp",
        # The configuration dictionary for the network.
        "network_config": {"hidden_units": [64, 64]},
        # The type of value network to use. Must be one of "copy" or "shared".
        "value_network": "copy",
        # Whether to use a floating scale for Gaussian policies.
        "free_log_std": True,
    }
)


@register("PPO")
class PPO(Agent):
    """Proximal Policy Optimization (PPO) agent.

    Args:
        env: The environment to interact with.
        config: The configuration dictionary for the agent.

    References:
        https://arxiv.org/abs/1707.06347
    """

    def __init__(self, env: Environment, config: ConfigDict):
        super().__init__(env, config)

        network_build_fn = get_network_build_fn(self.config.network, **self.config.network_config)
        model_fn = get_actor_critic_model_fn(
            env.action_space,
            network_build_fn,
            value_network=self.config.value_network,
            free_log_std=self.config.free_log_std,
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
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.scale_by_adam(eps=1e-5),
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

        metrics = {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "policy_entropy": entropy,
            "value_explained_variance": explained_variance(
                returns,
                value_preds,
                self.config.num_envs_per_device,
                self.config.num_devices,
            ),
            "clip_frac": clip_frac,
        }

        return loss, metrics

    def _update(self, params, opt_state, key, obs, actions, advantages, returns, old_log_probs, clip):
        grads, info = jax.grad(self._loss, has_aux=True)(
            params, key, obs, actions, advantages, returns, old_log_probs, clip
        )

        grads, info = self._maybe_all_reduce("pmean", (grads, info))

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

        self._send_episode_updates(rollout_data)

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

            minibatches = create_minibatches(shuffle_key, rollout_data, self.config.minibatch_size)

            return jax.lax.scan(update_scan_fn, (params, opt_state, next_key), minibatches)

        (new_params, new_opt_state, _), metrics = jax.lax.scan(
            epoch_scan_fn,
            (train_state.params, train_state.opt_state, initial_update_key),
            None,
            length=self.config.num_epochs,
        )

        metrics = jax.tree_map(jnp.mean, metrics)
        metrics = self._maybe_all_reduce("pmean", metrics)

        incremental_timesteps = (
            self.config.rollout_fragment_length * self.config.num_envs_per_device * self.config.num_devices
        )

        incremental_episodes = jnp.sum(rollout_data[SampleBatch.STEP_TYPE] == StepType.LAST)
        incremental_episodes = self._maybe_all_reduce("psum", incremental_episodes)

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
