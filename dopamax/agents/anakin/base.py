from abc import ABC
from typing import Optional, Sequence, Callable

import einops
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax.training.replay_buffers import ReplayBufferState, Queue
from chex import PRNGKey, ArrayTree, dataclass
from dm_env import StepType
from ml_collections import ConfigDict
from tqdm import tqdm

from dopamax.agents.base import Agent, TrainState, _EPISODE_BUFFER_SIZE
from dopamax.environments.environment import TimeStep, EnvState
from dopamax.rollouts import SampleBatch


@dataclass(frozen=True)
class AnakinTrainState(TrainState):
    """Dataclass for storing the training state of an Anakin agent."""

    key: PRNGKey
    train_step: int
    total_timesteps: int
    total_episodes: int
    episode_buffer_state: ReplayBufferState
    params: hk.Params
    opt_state: optax.OptState
    time_step: TimeStep
    env_state: EnvState

    @staticmethod
    def episode_buffer() -> Queue:
        return Queue(
            max_replay_size=_EPISODE_BUFFER_SIZE,
            dummy_data_sample={
                SampleBatch.EPISODE_LENGTH: 0,
                SampleBatch.EPISODE_REWARD: 0.0,
            },
            sample_batch_size=_EPISODE_BUFFER_SIZE,
            cyclic=True,
        )

    @classmethod
    def initial(
        cls, key: PRNGKey, params: hk.Params, opt_state: optax.OptState, time_step: TimeStep, env_state: EnvState
    ) -> "AnakinTrainState":
        """Creates an initial training state.

        Args:
            key: A PRNGKey.
            params: The initial parameters of the agent.
            opt_state: The initial optimizer state.
            time_step: The environment's initial time step.
            env_state: The environment's initial state.

        Returns:
            The initial training state.
        """
        q = AnakinTrainState.episode_buffer()
        episode_buffer_state = q.init(key)
        episode_buffer_state = q.insert(
            episode_buffer_state,
            {
                SampleBatch.EPISODE_LENGTH: -jnp.ones((_EPISODE_BUFFER_SIZE,), jnp.int32),
                SampleBatch.EPISODE_REWARD: -jnp.ones((_EPISODE_BUFFER_SIZE,), jnp.float32),
            },
        )

        return cls(
            key=key,
            train_step=0,
            total_timesteps=0,
            total_episodes=0,
            episode_buffer_state=episode_buffer_state,
            params=params,
            opt_state=opt_state,
            time_step=time_step,
            env_state=env_state,
        )

    def update(
        self,
        new_key: PRNGKey,
        rollout_data: SampleBatch,
        new_params: hk.Params,
        new_opt_state: optax.OptState,
        new_time_step: TimeStep,
        new_env_state: EnvState,
        maybe_all_reduce_fn: Callable[[str, ArrayTree], ArrayTree],
    ) -> "AnakinTrainState":
        """Updates a training state after the completion of a new training iteration.

        Args:
            new_key: A new PRNGKey.
            rollout_data: The rollout data generated in the last training iteration.
            new_params: The new parameters of the agent.
            new_opt_state: The new optimizer state.
            new_time_step: The new environment time step.
            new_env_state: The new environment state.
            maybe_all_reduce_fn: A function to use for all-reducing relevant metrics.

        Returns:
            The updated training state.
        """
        incremental_episodes = jnp.sum(rollout_data[SampleBatch.STEP_TYPE] == StepType.LAST)
        incremental_timesteps = rollout_data[SampleBatch.STEP_TYPE].shape[0]

        incremental_episodes = maybe_all_reduce_fn("psum", incremental_episodes)
        incremental_timesteps = maybe_all_reduce_fn("psum", incremental_timesteps)

        q = AnakinTrainState.episode_buffer()

        def scan_fn(state, sample):
            state = jax.lax.cond(
                sample[SampleBatch.STEP_TYPE] == StepType.LAST,
                lambda: q.insert(
                    state,
                    jax.tree_map(
                        lambda t: jnp.expand_dims(t, -1),
                        {
                            SampleBatch.EPISODE_LENGTH: sample[SampleBatch.EPISODE_LENGTH],
                            SampleBatch.EPISODE_REWARD: sample[SampleBatch.EPISODE_REWARD],
                        },
                    ),
                ),
                lambda: state,
            )
            return state, None

        new_episode_buffer_state, _ = jax.lax.scan(scan_fn, self.episode_buffer_state, rollout_data)

        return AnakinTrainState(
            key=new_key,
            train_step=self.train_step + 1,
            total_timesteps=self.total_timesteps + incremental_timesteps,
            total_episodes=self.total_episodes + incremental_episodes,
            episode_buffer_state=new_episode_buffer_state,
            params=new_params,
            opt_state=new_opt_state,
            time_step=new_time_step,
            env_state=new_env_state,
        )


@dataclass(frozen=True)
class AnakinTrainStateWithReplayBuffer(AnakinTrainState):
    buffer_state: ReplayBufferState

    @classmethod
    def initial(
        cls,
        key: PRNGKey,
        params: hk.Params,
        opt_state: optax.OptState,
        time_step: TimeStep,
        env_state: EnvState,
        buffer_state: ReplayBufferState,
    ) -> "AnakinTrainStateWithReplayBuffer":
        """Creates an initial training state.

        Args:
            key: A PRNGKey.
            params: The initial parameters of the agent.
            opt_state: The initial optimizer state.
            time_step: The environment's initial time step.
            env_state: The environment's initial state.
            buffer_state: The replay buffer's initial state.

        Returns:
            The initial training state.
        """
        q = AnakinTrainState.episode_buffer()
        episode_buffer_state = q.init(key)
        episode_buffer_state = q.insert(
            episode_buffer_state,
            {
                SampleBatch.EPISODE_LENGTH: -jnp.ones((_EPISODE_BUFFER_SIZE,), jnp.int32),
                SampleBatch.EPISODE_REWARD: -jnp.ones((_EPISODE_BUFFER_SIZE,), jnp.float32),
            },
        )

        return cls(
            key=key,
            train_step=0,
            total_timesteps=0,
            total_episodes=0,
            episode_buffer_state=episode_buffer_state,
            params=params,
            opt_state=opt_state,
            time_step=time_step,
            env_state=env_state,
            buffer_state=buffer_state,
        )

    def update(
        self,
        new_key: PRNGKey,
        rollout_data: SampleBatch,
        new_params: hk.Params,
        new_opt_state: optax.OptState,
        new_time_step: TimeStep,
        new_env_state: EnvState,
        new_buffer_state: ReplayBufferState,
        maybe_all_reduce_fn: Callable[[str, ArrayTree], ArrayTree],
    ) -> "AnakinTrainStateWithReplayBuffer":
        """Updates a training state after the completion of a new training iteration.

        Args:
            new_key: A new PRNGKey.
            rollout_data: The rollout data generated in the last training iteration.
            new_params: The new parameters of the agent.
            new_opt_state: The new optimizer state.
            new_time_step: The new environment time step.
            new_env_state: The new environment state.
            new_buffer_state: The new replay buffer state.
            maybe_all_reduce_fn: A function to use for all-reducing relevant metrics.

        Returns:
            The updated training state.
        """
        incremental_episodes = jnp.sum(rollout_data[SampleBatch.STEP_TYPE] == StepType.LAST)
        incremental_timesteps = rollout_data[SampleBatch.STEP_TYPE].shape[0]

        incremental_episodes = maybe_all_reduce_fn("psum", incremental_episodes)
        incremental_timesteps = maybe_all_reduce_fn("psum", incremental_timesteps)

        q = AnakinTrainState.episode_buffer()

        def scan_fn(state, sample):
            state = jax.lax.cond(
                sample[SampleBatch.STEP_TYPE] == StepType.LAST,
                lambda: q.insert(
                    state,
                    jax.tree_map(
                        lambda t: jnp.expand_dims(t, -1),
                        {
                            SampleBatch.EPISODE_LENGTH: sample[SampleBatch.EPISODE_LENGTH],
                            SampleBatch.EPISODE_REWARD: sample[SampleBatch.EPISODE_REWARD],
                        },
                    ),
                ),
                lambda: state,
            )
            return state, None

        new_episode_buffer_state, _ = jax.lax.scan(scan_fn, self.episode_buffer_state, rollout_data)

        return AnakinTrainStateWithReplayBuffer(
            key=new_key,
            train_step=self.train_step + 1,
            total_timesteps=self.total_timesteps + incremental_timesteps,
            total_episodes=self.total_episodes + incremental_episodes,
            episode_buffer_state=new_episode_buffer_state,
            params=new_params,
            opt_state=new_opt_state,
            time_step=new_time_step,
            env_state=new_env_state,
            buffer_state=new_buffer_state,
        )


_DEFAULT_ANAKIN_CONFIG = ConfigDict(
    {
        "num_devices": 1,
        "num_envs_per_device": 1,
    }
)


class AnakinAgent(Agent, ABC):
    batch_axis = "batch_axis"
    device_axis = "device_axis"

    @staticmethod
    def default_config() -> ConfigDict:
        config = super(AnakinAgent, AnakinAgent).default_config()
        config.update(_DEFAULT_ANAKIN_CONFIG)
        return config

    def _maybe_all_reduce(self, fn: str, x: ArrayTree) -> ArrayTree:
        """Performs an all-reduce operation if there are multiple devices or batching.

        Args:
            fn: The name of the all-reduce function to use in the jax.lax namespace.
            x: The array tree to all-reduce.

        Returns:
            The (maybe) all-reduced array tree.
        """
        fn = getattr(jax.lax, fn)

        if self.config.num_envs_per_device > 1:
            x = fn(x, axis_name=self.batch_axis)

        if self.config.num_devices > 1:
            x = fn(x, axis_name=self.device_axis)

        return x

    def train(
        self, key: PRNGKey, num_iterations: int, callback_freq: int = 100, callbacks: Optional[Sequence] = None
    ) -> hk.Params:
        """Trains the agent.

        Args:
            key: The PRNGKey to use to seed and randomness involved in training.
            num_iterations: The number of times to call the train_step function.
            callback_freq: The frequency, in terms of calls to `train_step`, at which to execute any callbacks.
            callbacks: A list of callbacks.

        Returns:
            The final parameters of the agent.
        """
        if num_iterations % callback_freq != 0:
            raise ValueError("num_iterations must be a multiple of callback_freq.")

        pbar = tqdm(total=num_iterations, desc="Training")
        callbacks = callbacks or []

        train_step_fn = self.train_step
        initial_train_state_fn = self.initial_train_state

        init_consistent_key, init_divergent_key = jax.random.split(key)
        init_divergent_key = jax.random.split(key, self.config.num_envs_per_device * self.config.num_devices)
        init_divergent_key = jnp.reshape(
            init_divergent_key, (self.config.num_devices, self.config.num_envs_per_device, 2)
        )

        if self.config.num_envs_per_device > 1:
            train_step_fn = jax.vmap(train_step_fn, axis_name=self.batch_axis)
            initial_train_state_fn = jax.vmap(initial_train_state_fn, axis_name=self.batch_axis, in_axes=(None, 0))
        else:
            init_divergent_key = jnp.squeeze(init_divergent_key, axis=1)

        def scan_fn(train_state, _):
            return train_step_fn(train_state)

        @jax.jit
        def train_fn(initial_train_state):
            new_train_state, metrics = jax.lax.scan(scan_fn, initial_train_state, None, length=callback_freq)

            metrics = jax.tree_map(jnp.mean, metrics)

            return new_train_state, metrics

        if self.config.num_devices > 1:
            train_fn = jax.pmap(train_fn, axis_name=self.device_axis)
            initial_train_state_fn = jax.pmap(initial_train_state_fn, axis_name=self.device_axis, in_axes=(None, 0))
        else:
            init_divergent_key = jnp.squeeze(init_divergent_key, axis=0)

        train_state = initial_train_state_fn(init_consistent_key, init_divergent_key)

        assert (num_iterations // callback_freq) > 0

        for _ in range(num_iterations // callback_freq):
            train_state, metrics = jax.device_get(train_fn(train_state))

            callback_train_state = train_state
            if self.config.num_envs_per_device > 1:
                callback_train_state = jax.tree_map(lambda x: x[0], callback_train_state)

            if self.config.num_devices > 1:
                callback_train_state = jax.tree_map(lambda x: x[0], callback_train_state)
                metrics = jax.tree_map(lambda x: x[0], metrics)

            episode_buffer = AnakinTrainState.episode_buffer()
            sample_fn = episode_buffer.sample_internal

            if self.config.num_envs_per_device > 1:
                sample_fn = jax.vmap(sample_fn)

            if self.config.num_devices > 1:
                sample_fn = jax.vmap(sample_fn)

            _, episodes = sample_fn(train_state.episode_buffer_state)
            episodes = jax.device_get(episodes)

            if self.config.num_envs_per_device > 1 or self.config.num_devices > 1:
                episodes = jax.tree_map(
                    lambda x: einops.rearrange(x, "... t e -> (t ...) e")[:, -_EPISODE_BUFFER_SIZE:], episodes
                )

            episodes = jax.tree_map(lambda x: x[episodes[SampleBatch.EPISODE_LENGTH] != -1], episodes)

            episode_metrics = {
                "min_episode_reward": np.min(episodes[SampleBatch.EPISODE_REWARD]),
                "mean_episode_reward": np.mean(episodes[SampleBatch.EPISODE_REWARD]),
                "max_episode_reward": np.max(episodes[SampleBatch.EPISODE_REWARD]),
                "min_episode_length": np.min(episodes[SampleBatch.EPISODE_LENGTH]),
                "mean_episode_length": np.mean(episodes[SampleBatch.EPISODE_LENGTH]),
                "max_episode_length": np.max(episodes[SampleBatch.EPISODE_LENGTH]),
            }

            metrics = {**metrics, **episode_metrics}

            for callback in callbacks:
                callback(callback_train_state, metrics)

            pbar.update(callback_freq)

        pbar.close()

        return callback_train_state.params
