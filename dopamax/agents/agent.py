from abc import ABC, abstractmethod
from collections import deque
from typing import Tuple, Optional, Sequence, Any

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax.training.replay_buffers import ReplayBufferState
from chex import dataclass, PRNGKey, Array, ArrayTree
from dm_env import StepType
from jax.experimental import host_callback
from ml_collections import ConfigDict
from tqdm import tqdm

from dopamax.environments.environment import Environment, TimeStep, EnvState
from dopamax.rollouts import SampleBatch
from dopamax.spaces import Space
from dopamax.typing import Observation, Action, Metrics


@dataclass(frozen=True)
class TrainState:
    """Dataclass for storing the training state of an agent."""

    key: PRNGKey
    train_step: int
    total_timesteps: int
    total_episodes: int
    params: hk.Params
    opt_state: optax.OptState
    time_step: TimeStep
    env_state: EnvState

    @classmethod
    def initial(
        cls, key: PRNGKey, params: hk.Params, opt_state: optax.OptState, time_step: TimeStep, env_state: EnvState
    ) -> "TrainState":
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
        return cls(
            key=key,
            train_step=0,
            total_timesteps=0,
            total_episodes=0,
            params=params,
            opt_state=opt_state,
            time_step=time_step,
            env_state=env_state,
        )

    def update(
        self,
        new_key: PRNGKey,
        incremental_timesteps: int,
        incremental_episodes: int,
        new_params: hk.Params,
        new_opt_state: optax.OptState,
        new_time_step: TimeStep,
        new_env_state: EnvState,
    ) -> "TrainState":
        """Updates a training state after the completion of a new training iteration.

        Args:
            new_key: A new PRNGKey.
            incremental_timesteps: The number of timesteps collected in the new training iteration.
            incremental_episodes: The number of episodes completed in the new training iteration.
            new_params: The new parameters of the agent.
            new_opt_state: The new optimizer state.
            new_time_step: The new environment time step.
            new_env_state: The new environment state.

        Returns:
            The updated training state.
        """
        return TrainState(
            key=new_key,
            train_step=self.train_step + 1,
            total_timesteps=self.total_timesteps + incremental_timesteps,
            total_episodes=self.total_episodes + incremental_episodes,
            params=new_params,
            opt_state=new_opt_state,
            time_step=new_time_step,
            env_state=new_env_state,
        )


@dataclass(frozen=True)
class TrainStateWithReplayBuffer(TrainState):
    buffer_state: Optional[ReplayBufferState] = None

    @classmethod
    def initial(
        cls,
        key: PRNGKey,
        params: hk.Params,
        opt_state: optax.OptState,
        time_step: TimeStep,
        env_state: EnvState,
        buffer_state: ReplayBufferState,
    ) -> "TrainStateWithReplayBuffer":
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
        return cls(
            key=key,
            train_step=0,
            total_timesteps=0,
            total_episodes=0,
            params=params,
            opt_state=opt_state,
            time_step=time_step,
            env_state=env_state,
            buffer_state=buffer_state,
        )

    def update(
        self,
        new_key: PRNGKey,
        incremental_timesteps: int,
        incremental_episodes: int,
        new_params: hk.Params,
        new_opt_state: optax.OptState,
        new_time_step: TimeStep,
        new_env_state: EnvState,
        new_buffer_state: ReplayBufferState,
    ) -> "TrainStateWithReplayBuffer":
        """Updates a training state after the completion of a new training iteration.

        Args:
            new_key: A new PRNGKey.
            incremental_timesteps: The number of timesteps collected in the new training iteration.
            incremental_episodes: The number of episodes completed in the new training iteration.
            new_params: The new parameters of the agent.
            new_opt_state: The new optimizer state.
            new_time_step: The new environment time step.
            new_env_state: The new environment state.
            new_buffer_state: The new replay buffer state.

        Returns:
            The updated training state.
        """
        return TrainStateWithReplayBuffer(
            key=new_key,
            train_step=self.train_step + 1,
            total_timesteps=self.total_timesteps + incremental_timesteps,
            total_episodes=self.total_episodes + incremental_episodes,
            params=new_params,
            opt_state=new_opt_state,
            time_step=new_time_step,
            env_state=new_env_state,
            buffer_state=new_buffer_state,
        )


class Agent(ABC):
    """Abstract base class for an RL agent.

    Subclasses are intended to be implemented in accordance with the Anakin Podracer architecture.
    See: https://arxiv.org/abs/2104.06272

    This means that we assume the agent's train step can be vectorized across a batch within each device, and can be
    further parallelized across multiple devices.

    Args:
        env: The environment to interact with.
        config: The configuration dictionary for the agent.
    """

    batch_axis = "batch_axis"
    device_axis = "device_axis"

    def __init__(self, env: Environment, config: ConfigDict):
        self._env = env
        self._config = self.default_config()
        self._config.update(config)
        self._config.lock()

        self._reward_buffer = deque(maxlen=100)
        self._length_buffer = deque(maxlen=100)

    @property
    def env(self) -> Environment:
        """The environment that the agent interacts with."""
        return self._env

    @property
    def config(self) -> ConfigDict:
        """The configuration dictionary for the agent."""
        return self._config

    @staticmethod
    def default_config() -> ConfigDict:
        """Returns the default configuration dictionary for all agents."""
        return ConfigDict(
            {
                "num_devices": 1,
                "num_envs_per_device": 1,
            }
        )

    @property
    def observation_space(self) -> Space:
        """The observation space of the agent's environment."""
        return self._env.observation_space

    @property
    def action_space(self) -> Space:
        """The action space of the agent's environment."""
        return self._env.action_space

    @abstractmethod
    def compute_action(
        self, params: hk.Params, key: PRNGKey, observation: Observation, deterministic: bool = True
    ) -> Action:
        pass

    @abstractmethod
    def initial_train_state(self, consistent_key: PRNGKey, divergent_key: PRNGKey) -> TrainState:
        """Initializes the agent's training state.

        Args:
            consistent_key: A PRNGKey that is used to initialize pieces of the train state which should be
                consistent across all devices (e.g. network parameters).
            divergent_key: A PRNGKey that is used to initialize pieces of the train state which should be
                different across all devices (e.g. environment state).

        Returns:
            The initial training state.
        """
        pass

    @abstractmethod
    def train_step(self, train_state: TrainState) -> Tuple[TrainState, Metrics]:
        """Performs a single training step.

        A training step should include experience collection and parameter updates.

        Args:
            train_state: The current training state.

        Returns:
            train_state: The updated training state.
            metrics: A dictionary of metrics to be logged.
        """
        pass

    def _episode_buffer_update_fn(self, args: Tuple[Array, Array], transforms: Any):
        """Updates the episode reward and length buffers with new data."""
        rewards, lengths = args

        rewards = rewards.flatten()
        lengths = lengths.flatten()

        for reward, length in zip(rewards, lengths):
            if length != -np.inf:
                self._reward_buffer.append(reward)
                self._length_buffer.append(length)

    def _send_episode_updates(self, rollout_data: SampleBatch):
        """Sends rollout data to the host for updating the episode metrics for logging.

        Args:
            rollout_data: A batch of rollout data.

        Returns:
            None
        """
        rewards = jnp.where(
            rollout_data[SampleBatch.STEP_TYPE] == StepType.LAST, rollout_data[SampleBatch.EPISODE_REWARD], -jnp.inf
        )
        lengths = jnp.where(
            rollout_data[SampleBatch.STEP_TYPE] == StepType.LAST, rollout_data[SampleBatch.EPISODE_LENGTH], -jnp.inf
        )
        host_callback.id_tap(self._episode_buffer_update_fn, (rewards, lengths))

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
            callback_freq: The frequency, in terms of call to `train_step`, at which to execute any callbacks.
            callbacks: A list of callbacks.

        Returns:
            The final parameters of the agent.
        """
        self._reward_buffer.clear()
        self._length_buffer.clear()

        pbar = tqdm(total=num_iterations, desc="Training")
        callbacks = callbacks or []

        def pbar_update_fn(args, transforms, device):
            if device.id == 0:
                pbar.update(10)

        def callback_fn(args, transforms, device):
            if device.id != 0:
                return

            state, metrics = args

            metrics["min_episode_reward"] = min(self._reward_buffer)
            metrics["max_episode_reward"] = max(self._reward_buffer)
            metrics["mean_episode_reward"] = (
                sum(self._reward_buffer) / len(self._reward_buffer) if self._reward_buffer else 0
            )

            metrics["min_episode_length"] = min(self._length_buffer)
            metrics["max_episode_length"] = max(self._length_buffer)
            metrics["mean_episode_length"] = (
                sum(self._length_buffer) / len(self._length_buffer) if self._length_buffer else 0
            )

            for callback in callbacks:
                callback(state, metrics)

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

        def loop_fn(i, train_state):
            new_train_state, metrics = train_step_fn(train_state)

            callback_train_state = new_train_state

            if self.config.num_envs_per_device > 1:
                metrics = jax.tree_map(lambda x: x[0], metrics)
                callback_train_state = jax.tree_map(lambda x: x[0], callback_train_state)

            new_train_state = jax.lax.cond(
                callback_train_state.train_step % callback_freq == 0,
                lambda _: host_callback.id_tap(
                    callback_fn, (callback_train_state, metrics), result=new_train_state, tap_with_device=True
                ),
                lambda _: new_train_state,
                operand=None,
            )

            new_train_state = jax.lax.cond(
                callback_train_state.train_step % 10 == 0,
                lambda _: host_callback.id_tap(pbar_update_fn, (), result=new_train_state, tap_with_device=True),
                lambda _: new_train_state,
                operand=None,
            )

            return new_train_state

        @jax.jit
        def train_fn(initial_train_state):
            return jax.lax.fori_loop(0, num_iterations, loop_fn, initial_train_state)

        if self.config.num_devices > 1:
            train_fn = jax.pmap(train_fn, axis_name=self.device_axis)
            initial_train_state_fn = jax.pmap(initial_train_state_fn, axis_name=self.device_axis, in_axes=(None, 0))
        else:
            init_divergent_key = jnp.squeeze(init_divergent_key, axis=0)

        initial_train_state = initial_train_state_fn(init_consistent_key, init_divergent_key)
        train_state = train_fn(initial_train_state)

        final_params = jax.device_get(train_state.params)

        if self.config.num_envs_per_device > 1:
            final_params = jax.tree_map(lambda x: x[0], final_params)

        if self.config.num_devices > 1:
            final_params = jax.tree_map(lambda x: x[0], final_params)

        pbar.close()

        return final_params
