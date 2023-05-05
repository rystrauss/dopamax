from abc import ABC
from typing import Optional, Sequence, Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax.training.replay_buffers import ReplayBufferState
from chex import PRNGKey, ArrayTree, dataclass, Scalar
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
    params: hk.Params
    opt_state: optax.OptState
    time_step: TimeStep
    env_state: EnvState

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
    ) -> "AnakinTrainState":
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
        return AnakinTrainState(
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
    ) -> "AnakinTrainStateWithReplayBuffer":
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
        return AnakinTrainStateWithReplayBuffer(
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

    def _get_episode_metrics(self, rollout_data: SampleBatch) -> Tuple[Scalar, Dict[str, Scalar]]:
        incremental_episodes = jnp.sum(rollout_data[SampleBatch.STEP_TYPE] == StepType.LAST)

        episode_metrics = {
            "episode_count": incremental_episodes,
            "sum_episode_reward": jnp.where(
                rollout_data[SampleBatch.STEP_TYPE] == StepType.LAST, rollout_data[SampleBatch.EPISODE_REWARD], 0
            ).sum(),
            "min_episode_reward": jnp.where(
                rollout_data[SampleBatch.STEP_TYPE] == StepType.LAST, rollout_data[SampleBatch.EPISODE_REWARD], jnp.inf
            ).min(),
            "max_episode_reward": jnp.where(
                rollout_data[SampleBatch.STEP_TYPE] == StepType.LAST, rollout_data[SampleBatch.EPISODE_REWARD], -jnp.inf
            ).max(),
            "sum_episode_length": jnp.where(
                rollout_data[SampleBatch.STEP_TYPE] == StepType.LAST, rollout_data[SampleBatch.EPISODE_LENGTH], 0
            ).sum(),
            "min_episode_length": jnp.where(
                rollout_data[SampleBatch.STEP_TYPE] == StepType.LAST, rollout_data[SampleBatch.EPISODE_LENGTH], jnp.inf
            ).min(),
            "max_episode_length": jnp.where(
                rollout_data[SampleBatch.STEP_TYPE] == StepType.LAST, rollout_data[SampleBatch.EPISODE_LENGTH], -jnp.inf
            ).max(),
        }

        return incremental_episodes, episode_metrics

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
            new_train_state, (metrics, episode_metrics) = jax.lax.scan(
                scan_fn, initial_train_state, None, length=callback_freq
            )

            if self.config.num_envs_per_device > 1:
                for k in episode_metrics.keys():
                    if k.startswith("max_"):
                        episode_metrics[k] = jnp.max(episode_metrics[k], axis=-1)
                    elif k.startswith("min_"):
                        episode_metrics[k] = jnp.min(episode_metrics[k], axis=-1)
                    else:
                        episode_metrics[k] = jnp.sum(episode_metrics[k], axis=-1)

            cumulative_episodes = jnp.cumsum(jnp.flip(episode_metrics["episode_count"]))
            cutoff_mask = jnp.flip(cumulative_episodes <= _EPISODE_BUFFER_SIZE)

            for k in episode_metrics.keys():
                if k.startswith("max_"):
                    episode_metrics[k] = jnp.max(jnp.where(cutoff_mask, episode_metrics[k], -jnp.inf))
                elif k.startswith("min_"):
                    episode_metrics[k] = jnp.min(jnp.where(cutoff_mask, episode_metrics[k], jnp.inf))
                else:
                    episode_metrics[k] = jnp.sum(jnp.where(cutoff_mask, episode_metrics[k], 0))

            metrics = jax.tree_map(jnp.mean, metrics)

            return new_train_state, (metrics, episode_metrics)

        if self.config.num_devices > 1:
            train_fn = jax.pmap(train_fn, axis_name=self.device_axis)
            initial_train_state_fn = jax.pmap(initial_train_state_fn, axis_name=self.device_axis, in_axes=(None, 0))
        else:
            init_divergent_key = jnp.squeeze(init_divergent_key, axis=0)

        train_state = initial_train_state_fn(init_consistent_key, init_divergent_key)

        assert (num_iterations // callback_freq) > 0

        for _ in range(num_iterations // callback_freq):
            train_state, (metrics, episode_metrics) = jax.device_get(train_fn(train_state))

            callback_train_state = train_state
            if self.config.num_envs_per_device > 1:
                callback_train_state = jax.tree_map(lambda x: x[0], callback_train_state)

            if self.config.num_devices > 1:
                callback_train_state = jax.tree_map(lambda x: x[0], callback_train_state)
                metrics = jax.tree_map(lambda x: x[0], metrics)

                cumulative_episodes = np.cumsum(np.flip(episode_metrics["episode_count"]))
                cutoff_mask = np.flip(cumulative_episodes <= _EPISODE_BUFFER_SIZE)

                for k in episode_metrics.keys():
                    if k.startswith("max_"):
                        episode_metrics[k] = np.max(np.where(cutoff_mask, episode_metrics[k], -np.inf))
                    elif k.startswith("min_"):
                        episode_metrics[k] = np.min(np.where(cutoff_mask, episode_metrics[k], np.inf))
                    else:
                        episode_metrics[k] = np.sum(np.where(cutoff_mask, episode_metrics[k], 0))

            episode_metrics["mean_episode_reward"] = (
                episode_metrics["sum_episode_reward"] / episode_metrics["episode_count"]
            )
            episode_metrics["mean_episode_length"] = (
                episode_metrics["sum_episode_length"] / episode_metrics["episode_count"]
            )
            del episode_metrics["sum_episode_reward"]
            del episode_metrics["sum_episode_length"]
            del episode_metrics["episode_count"]

            metrics = {**metrics, **episode_metrics}

            for callback in callbacks:
                callback(callback_train_state, metrics)

            pbar.update(callback_freq)

        pbar.close()

        return callback_train_state.params
