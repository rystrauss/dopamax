from functools import partial
from typing import Tuple, Dict, Optional

import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import optax
from brax.training.replay_buffers import UniformSamplingQueue
from chex import PRNGKey, Array, ArrayTree
from mctx._src.base import RecurrentState, RecurrentFnOutput
from ml_collections import ConfigDict

from dopamax.agents.anakin.base import AnakinAgent, AnakinTrainState, AnakinTrainStateWithReplayBuffer
from dopamax.agents.utils import register
from dopamax.environments.environment import Environment, EnvState
from dopamax.environments.pgx.base import PGXEnvironment
from dopamax.networks import get_network_build_fn, get_actor_critic_model_fn
from dopamax.rollouts import SampleBatch, rollout_truncated
from dopamax.typing import Metrics, Observation, Action

_DEFAULT_ALPHAZERO_CONFIG = ConfigDict(
    {
        "rollout_fragment_length": 256,
        "num_simulations": 800,
        "max_depth": 100,
        "root_dirichlet_alpha": 0.3,
        "root_exploration_fraction": 0.25,
        "pb_c_base": 19652,
        "pb_c_init": 1.25,
        "weight_decay": 1e-4,
        "momentum": 0.9,
        "lr_init": 2e-3,
        "lr_decay_rate": 1.0,
        "lr_decay_steps": 10000,
        "value_loss_coefficient": 1.0,
        # The type of network to use.
        "network": "mlp",
        # The configuration dictionary for the network.
        "network_config": {"hidden_units": [64, 64]},
        # The type of value network to use. Must be one of "copy" or "shared".
        "value_network": "copy",
        # Whether to use a floating scale for Gaussian policies.
        "free_log_std": True,
        "buffer_size": 100000,
        "batch_size": 32,
        "num_updates": 8,
    }
)


@register("AlphaZero")
class AlphaZero(AnakinAgent):
    """AlphaZero agent.

    Note that this implementation is slightly modified from original version of the algorithm as to adhere to the
    Anakin architecture. It also uses a more modern version of Monte Carlo Tree Search, as implemented by
    `mctx.muzero_policy`.

    Args:
        env: The environment to interact with. This should be a subclass of `PGXEnvironment`.
        config: The configuration dictionary for the agent.

    References:
        https://arxiv.org/abs/1712.01815
    """
    def __init__(self, env: Environment, config: ConfigDict):
        super().__init__(env, config)

        assert isinstance(env, PGXEnvironment), "AlphaZero only supports `PGXEnvironment`s."

        network_build_fn = get_network_build_fn(self.config.network, **self.config.network_config)
        model_fn = get_actor_critic_model_fn(
            env.action_space,
            network_build_fn,
            value_network=self.config.value_network,
            free_log_std=self.config.free_log_std,
            tanh_value=True,
        )
        self._model = hk.transform(model_fn)

        def recurrent_fn(
            params: hk.Params, key: PRNGKey, action: Array, embedding: RecurrentState
        ) -> Tuple[RecurrentFnOutput, RecurrentState]:
            env_state = jax.tree_map(lambda x: jnp.squeeze(x, axis=0), embedding)
            action = jnp.squeeze(action, axis=0)

            env_key, model_key = jax.random.split(key)

            next_time_step, next_env_state = env.step(env_key, env_state, action)
            next_time_step, next_env_state = jax.tree_map(
                lambda x: jnp.expand_dims(x, axis=0), (next_time_step, next_env_state)
            )

            pi, value = self._model.apply(params, model_key, next_time_step.observation["observation"])

            prior_logits = pi.logits - jnp.max(pi.logits, axis=-1, keepdims=True)
            prior_logits = jnp.where(
                next_time_step.observation["invalid_actions"], jnp.finfo(prior_logits.dtype).min, prior_logits
            )

            recurrent_fn_output = mctx.RecurrentFnOutput(
                reward=next_time_step.reward,
                discount=-next_time_step.discount,
                prior_logits=prior_logits,
                value=value,
            )

            return recurrent_fn_output, next_env_state

        self._recurrent_fn = recurrent_fn

        def policy_fn(
            params: hk.Params, key: PRNGKey, observation: Observation, env_state: EnvState
        ) -> Tuple[Action, Dict[str, ArrayTree]]:
            model_key, search_key = jax.random.split(key)
            observation = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), observation)
            env_state = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), env_state)

            pi, value = self._model.apply(params, model_key, observation["observation"])

            root = mctx.RootFnOutput(prior_logits=pi.logits, value=value, embedding=env_state)

            invalid_actions = (
                observation["invalid_actions"]
                if isinstance(observation, dict) and "invalid_actions" in observation
                else None
            )

            policy_output = mctx.muzero_policy(
                params=params,
                rng_key=search_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=self.config.num_simulations,
                invalid_actions=invalid_actions,
                max_depth=self.config.max_depth,
                dirichlet_fraction=self.config.root_exploration_fraction,
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                pb_c_base=self.config.pb_c_base,
                pb_c_init=self.config.pb_c_init,
            )

            target_value = policy_output.search_tree.summary().value

            policy_output, value, target_value = jax.tree_map(
                lambda x: jnp.squeeze(x, axis=0), (policy_output, value, target_value)
            )

            return policy_output.action, {
                SampleBatch.ACTION: policy_output.action,
                SampleBatch.VALUE: value,
                "search_target_value": target_value,
                "search_action_weights": policy_output.action_weights,
            }

        self._rollout_fn = partial(
            rollout_truncated, env, self.config.rollout_fragment_length, policy_fn, pass_env_state_to_policy=True
        )

        self._lr_schedule = optax.exponential_decay(
            init_value=self.config.lr_init,
            transition_steps=self.config.lr_decay_steps,
            decay_rate=self.config.lr_decay_rate,
        )
        self._optimizer = optax.chain(
            optax.trace(decay=self.config.momentum),
            optax.add_decayed_weights(self.config.weight_decay),
            optax.scale_by_schedule(self._lr_schedule),
            optax.scale(-1.0),
        )

        temp_key = jax.random.PRNGKey(0)
        temp_train_state = self._initial_train_state_without_replay_buffer(temp_key, temp_key)
        rollout_data_spec, _, _, _ = jax.eval_shape(
            self._rollout_fn,
            temp_train_state.params,
            temp_train_state.key,
            temp_train_state.time_step,
            temp_train_state.env_state,
        )

        sample = jax.tree_map(lambda x: jnp.empty(x.shape, x.dtype)[0], rollout_data_spec)

        self._buffer = UniformSamplingQueue(self.config.buffer_size, sample, self.config.batch_size)

    @staticmethod
    def default_config() -> ConfigDict:
        config = super(AlphaZero, AlphaZero).default_config()
        config.update(_DEFAULT_ALPHAZERO_CONFIG)
        return config

    def compute_action(
        self,
        params: hk.Params,
        key: PRNGKey,
        observation: Observation,
        env_state: EnvState,
        deterministic: bool = True,
        num_simulations: Optional[int] = None,
    ) -> Action:
        model_key, search_key = jax.random.split(key)

        pi, value = self._model.apply(params, model_key, observation["observation"])

        if num_simulations == 0:
            if deterministic:
                return pi.mode()
            else:
                return pi.sample(key=key)

        root = mctx.RootFnOutput(prior_logits=pi.logits, value=value, embedding=env_state)

        invalid_actions = (
            observation["invalid_actions"]
            if isinstance(observation, dict) and "invalid_actions" in observation
            else None
        )

        policy_output = mctx.muzero_policy(
            params=params,
            rng_key=search_key,
            root=root,
            recurrent_fn=self._recurrent_fn,
            num_simulations=num_simulations or self.config.num_simulations,
            invalid_actions=invalid_actions,
            max_depth=self.config.max_depth,
            dirichlet_fraction=self.config.root_exploration_fraction,
            dirichlet_alpha=self.config.root_dirichlet_alpha,
            pb_c_base=self.config.pb_c_base,
            pb_c_init=self.config.pb_c_init,
            temperature=0.0 if deterministic else 1.0,
        )

        return policy_output.action

    def _initial_train_state_without_replay_buffer(
        self, consistent_key: PRNGKey, divergent_key: PRNGKey
    ) -> AnakinTrainState:
        train_state_key, env_key = jax.random.split(divergent_key)
        sample_obs = jnp.expand_dims(self.env.observation_space.sample(consistent_key)["observation"], 0)
        params = self._model.init(consistent_key, sample_obs)
        time_step, env_state = self.env.reset(env_key)
        opt_state = self._optimizer.init(params)

        return AnakinTrainState.initial(train_state_key, params, opt_state, time_step, env_state)

    def initial_train_state(self, consistent_key: PRNGKey, divergent_key: PRNGKey) -> AnakinTrainStateWithReplayBuffer:
        train_state_without_replay_buffer = self._initial_train_state_without_replay_buffer(
            consistent_key, divergent_key
        )
        return AnakinTrainStateWithReplayBuffer(
            **dict(train_state_without_replay_buffer), buffer_state=self._buffer.init(consistent_key)
        )

    def _loss(self, params, key, obs, search_action_weights, search_target_value):
        pi, value = self._model.apply(params, key, obs)

        value_loss = jnp.mean((value - search_target_value) ** 2)
        policy_loss = jnp.mean(optax.softmax_cross_entropy(pi.logits, search_action_weights))

        loss = self.config.value_loss_coefficient * value_loss + policy_loss

        metrics = {
            "loss": loss,
            "value_loss": value_loss,
            "policy_loss": policy_loss,
        }

        return loss, metrics

    def _update(self, params, opt_state, key, obs, search_action_weights, search_target_value):
        grads, info = jax.grad(self._loss, has_aux=True)(params, key, obs, search_action_weights, search_target_value)

        grads, info = self._maybe_all_reduce("pmean", (grads, info))

        updates, new_opt_state = self._optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, info

    def train_step(
        self, train_state: AnakinTrainStateWithReplayBuffer
    ) -> Tuple[AnakinTrainStateWithReplayBuffer, Metrics]:
        next_train_state_key, rollout_key, initial_update_key = jax.random.split(train_state.key, 3)

        rollout_data, _, new_time_step, new_env_state = self._rollout_fn(
            train_state.params,
            rollout_key,
            train_state.time_step,
            train_state.env_state,
        )

        new_buffer_state = self._buffer.insert(train_state.buffer_state, rollout_data)

        def update_scan_fn(carry, _):
            params, opt_state, buffer_state, key = carry

            next_key, update_key = jax.random.split(key)

            new_buffer_state, batch = self._buffer.sample(buffer_state)

            new_params, new_opt_state, info = self._update(
                params,
                opt_state,
                update_key,
                batch[SampleBatch.OBSERVATION]["observation"],
                batch["search_action_weights"],
                batch["search_target_value"],
            )

            schedule_count = opt_state[-2].count
            info["learning_rate"] = self._lr_schedule(schedule_count)

            return (new_params, new_opt_state, new_buffer_state, next_key), info

        (new_params, new_opt_state, new_buffer_state, _), metrics = jax.lax.scan(
            update_scan_fn,
            (train_state.params, train_state.opt_state, new_buffer_state, initial_update_key),
            None,
            length=self.config.num_updates,
        )

        metrics = jax.tree_map(jnp.mean, metrics)

        next_train_state = train_state.update(
            new_key=next_train_state_key,
            rollout_data=rollout_data,
            new_params=new_params,
            new_opt_state=new_opt_state,
            new_time_step=new_time_step,
            new_env_state=new_env_state,
            new_buffer_state=new_buffer_state,
            maybe_all_reduce_fn=self._maybe_all_reduce,
        )

        return next_train_state, metrics
