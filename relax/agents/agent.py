from abc import ABC, abstractmethod
from typing import Tuple, Optional, Sequence

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from chex import dataclass, PRNGKey
from jax.experimental import host_callback
from ml_collections import ConfigDict
from tqdm import tqdm

from relax.environments.environment import Environment, TimeStep, EnvState
from relax.spaces import Space
from relax.typing import Observation, Action, Metrics


@dataclass(frozen=True)
class TrainState:
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


class Agent(ABC):
    batch_axis = "batch_axis"
    device_axis = "device_axis"

    def __init__(self, env: Environment, config: ConfigDict):
        self._env = env
        self._config = self.default_config()
        self._config.update(config)
        self._config.lock()

    @property
    def env(self) -> Environment:
        return self._env

    @property
    def config(self) -> ConfigDict:
        return self._config

    @staticmethod
    def default_config() -> ConfigDict:
        # Here we define base configurations that are common across all agents.
        return ConfigDict(
            {
                "num_devices": 1,
                "num_envs_per_device": 4,
            }
        )

    @property
    def observation_space(self) -> Space:
        return self._env.observation_space

    @property
    def action_space(self) -> Space:
        return self._env.action_space

    @abstractmethod
    def compute_action(
        self, params: hk.Params, key: PRNGKey, observation: Observation, deterministic: bool = True
    ) -> Action:
        pass

    @abstractmethod
    def initial_train_state(self, consistent_key: PRNGKey, divergent_key: PRNGKey) -> TrainState:
        pass

    @abstractmethod
    def train_step(self, train_state: TrainState) -> Tuple[TrainState, Metrics]:
        pass

    def train(
        self, key: PRNGKey, num_iterations: int, callback_freq: int = 100, callbacks: Optional[Sequence] = None
    ) -> TrainState:
        pbar = tqdm(total=num_iterations, desc="Training")
        callbacks = callbacks or []

        def pbar_update_fn(args, transforms, device):
            if device.id == 0:
                pbar.update(10)

        def callback_fn(args, transforms, device):
            if device.id == 0:
                return

            state, metrics = args

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
        train_state = jax.device_get(train_fn(initial_train_state))

        if self.config.num_envs_per_device > 1:
            train_state = jax.tree_map(lambda x: x[0], train_state)

        if self.config.num_devices > 1:
            train_state = jax.tree_map(lambda x: x[0], train_state)

        pbar.close()

        return train_state
