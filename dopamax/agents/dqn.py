from functools import partial
from typing import Tuple, Dict

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import rlax
from brax.training.replay_buffers import UniformSamplingQueue
from chex import PRNGKey, ArrayTree
from dm_env import StepType
from ml_collections import ConfigDict

from dopamax.agents.agent import Agent, TrainState, TrainStateWithReplayBuffer
from dopamax.agents.utils import register
from dopamax.environments.environment import Environment
from dopamax.networks import get_network_build_fn, get_discrete_q_network_model_fn
from dopamax.rollouts import rollout_truncated, SampleBatch
from dopamax.spaces import Discrete
from dopamax.typing import Metrics, Observation, Action

_DEFAULT_DQN_CONFIG = ConfigDict(
    {
        # The initial epsilon value for epsilon-greedy exploration.
        "initial_epsilon": 1.0,
        # The final epsilon value for epsilon-greedy exploration.
        "final_epsilon": 0.02,
        # The number of training steps over which to anneal epsilon.
        "epsilon_decay_steps": 10000,
        # The discount factor.
        "gamma": 0.99,
        # The learning rate.
        "learning_rate": 5e-4,
        # The number of training steps to take before beginning gradient updates.
        "learning_starts": 1000,
        # The size of experience batches to sample from the replay buffer and learn on.
        "batch_size": 32,
        # The maximum size of the replay buffer.
        "buffer_size": 10000,
        # The frequency with which the target network is updated.
        "target_update_freq": 500,
        # The type of network to use.
        "network": "mlp",
        # The configuration dictionary for the network.
        "network_config": {"hidden_units": [64, 64]},
        # Whether to use double q-learning.
        "double": True,
        # Whether to use a dueling q-network architecture..
        "dueling": False,
        # Additional fully-connected layers to add after the base network before the output layer. At least one layer
        # must be added when using the dueling architecture.
        "final_hidden_units": (),
    }
)


@register("DQN")
class DQN(Agent):
    """Deep Q-Network (DQN) agent.

    This is a simple DQN implementation that is very close to the original paper. It doesn't have a lot of the more
    recent bells and whistles that have been developed.

    Args:
        env: The environment to interact with.
        config: The configuration dictionary for the agent.

    References:
        https://www.nature.com/articles/nature14236
    """

    def __init__(self, env: Environment, config: ConfigDict):
        super().__init__(env, config)

        assert isinstance(self.env.action_space, Discrete), "DQN only supports discrete action spaces."

        network_build_fn = get_network_build_fn(self.config.network, **self.config.network_config)
        model_fn = get_discrete_q_network_model_fn(
            env.action_space, network_build_fn, self.config.final_hidden_units, self.config.dueling
        )
        self._model = hk.transform(model_fn)

        def policy_fn(
            params: hk.Params, key: PRNGKey, observation: Observation, epsilon: float
        ) -> Tuple[Action, Dict[str, ArrayTree]]:
            model_key, sample_key = jax.random.split(key)

            preferences = self._model.apply(params, model_key, observation)
            action = distrax.EpsilonGreedy(preferences, epsilon).sample(seed=sample_key)

            return action, {}

        self._epsilon_schedule = optax.linear_schedule(
            self.config.initial_epsilon, self.config.final_epsilon, self.config.epsilon_decay_steps
        )
        self._optimizer = optax.adam(self.config.learning_rate)

        self._rollout_fn = partial(rollout_truncated, env, 1, policy_fn)

        temp_key = jax.random.PRNGKey(0)
        temp_train_state = self._initial_train_state_without_replay_buffer(temp_key, temp_key)
        rollout_data, _, _, _ = self._rollout_fn(
            temp_train_state.params["online"],
            temp_train_state.key,
            temp_train_state.time_step,
            temp_train_state.env_state,
            epsilon=1.0,
        )

        # TODO: Maybe wrap buffer for pmap, and check key if so.
        self._buffer = UniformSamplingQueue(self.config.buffer_size, rollout_data, self.config.batch_size)

    @staticmethod
    def default_config() -> ConfigDict:
        config = super(DQN, DQN).default_config()
        config.update(_DEFAULT_DQN_CONFIG)
        config.lock()
        return config

    def compute_action(
        self,
        params: hk.Params,
        key: PRNGKey,
        observation: Observation,
        deterministic: bool = True,
    ) -> Action:
        preferences = self._model.apply(params["online"], key, observation)
        pi = distrax.Categorical(logits=preferences)
        return pi.mode() if deterministic else pi.sample(seed=key)

    def _initial_train_state_without_replay_buffer(self, consistent_key: PRNGKey, divergent_key: PRNGKey) -> TrainState:
        train_state_key, env_key = jax.random.split(divergent_key)
        sample_obs = jnp.expand_dims(self.env.observation_space.sample(consistent_key), 0)
        params = self._model.init(consistent_key, sample_obs)
        params = {"online": params, "target": params}
        time_step, env_state = self.env.reset(env_key)
        opt_state = self._optimizer.init(params["online"])

        return TrainState.initial(train_state_key, params, opt_state, time_step, env_state)

    def initial_train_state(self, consistent_key: PRNGKey, divergent_key: PRNGKey) -> TrainStateWithReplayBuffer:
        train_state_without_replay_buffer = self._initial_train_state_without_replay_buffer(
            consistent_key, divergent_key
        )
        return TrainStateWithReplayBuffer(
            **dict(train_state_without_replay_buffer), buffer_state=self._buffer.init(consistent_key)
        )

    def _loss(self, online_params, target_params, key, obs, actions, rewards, next_obs, discounts):
        next_q_values = self._model.apply(target_params, key, next_obs)
        q_values = self._model.apply(online_params, key, obs)

        if self.config.double:
            next_q_selectors = self._model.apply(online_params, key, next_obs)
            td_error = jax.vmap(rlax.double_q_learning)(
                q_values, actions, rewards, discounts, next_q_values, next_q_selectors
            )
        else:
            td_error = jax.vmap(rlax.q_learning)(q_values, actions, rewards, discounts, next_q_values)

        loss = jnp.mean(rlax.l2_loss(td_error))

        metrics = {
            "loss": loss,
        }

        return loss, metrics

    def _update(self, train_step, params, opt_state, key, obs, actions, rewards, next_obs, discounts):
        grads, info = jax.grad(self._loss, has_aux=True)(
            params["online"], params["target"], key, obs, actions, rewards, next_obs, discounts
        )

        grads, info = self._maybe_all_reduce("pmean", (grads, info))

        updates, new_opt_state = self._optimizer.update(grads, opt_state, params["online"])
        new_online_params = optax.apply_updates(params["online"], updates)

        new_target_params = jax.tree_map(
            lambda a, b: jax.lax.select(
                jnp.logical_and(
                    jnp.any(train_step > self.config.learning_starts),
                    jnp.any(train_step % self.config.target_update_freq == 0),
                ),
                a,
                b,
            ),
            new_online_params,
            params["target"],
        )

        new_params = {"online": new_online_params, "target": new_target_params}

        return new_params, new_opt_state, info

    def train_step(self, train_state: TrainStateWithReplayBuffer) -> Tuple[TrainStateWithReplayBuffer, Metrics]:
        next_train_state_key, rollout_key, update_key = jax.random.split(train_state.key, 3)

        current_epsilon = self._epsilon_schedule(train_state.train_step)

        rollout_data, _, new_time_step, new_env_state = self._rollout_fn(
            train_state.params["online"],
            rollout_key,
            train_state.time_step,
            train_state.env_state,
            epsilon=current_epsilon,
        )

        self._send_episode_updates(rollout_data)

        new_buffer_state = self._buffer.insert(train_state.buffer_state, rollout_data)
        new_buffer_state, sample = self._buffer.sample(new_buffer_state)

        sample = jax.tree_map(lambda x: jnp.squeeze(x, 1), sample)

        new_params, new_opt_state, metrics = self._update(
            train_state.train_step,
            train_state.params,
            train_state.opt_state,
            update_key,
            sample[SampleBatch.OBSERVATION],
            sample[SampleBatch.ACTION],
            sample[SampleBatch.REWARD],
            sample[SampleBatch.NEXT_OBSERVATION],
            sample[SampleBatch.DISCOUNT],
        )

        metrics = jax.tree_map(jnp.mean, metrics)
        metrics = self._maybe_all_reduce("pmean", metrics)

        incremental_timesteps = self.config.num_envs_per_device * self.config.num_devices
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
            new_buffer_state=new_buffer_state,
        )

        warmup_train_state = train_state.update(
            new_key=next_train_state_key,
            incremental_timesteps=incremental_timesteps,
            incremental_episodes=incremental_episodes,
            new_params=train_state.params,
            new_opt_state=train_state.opt_state,
            new_time_step=new_time_step,
            new_env_state=new_env_state,
            new_buffer_state=new_buffer_state,
        )

        next_train_state = jax.tree_map(
            lambda a, b: jax.lax.select(jnp.any(train_state.train_step > self.config.learning_starts), a, b),
            next_train_state,
            warmup_train_state,
        )

        return next_train_state, metrics
