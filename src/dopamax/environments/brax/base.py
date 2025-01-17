from abc import ABC
from typing import Tuple

import jax
import jax.numpy as jnp
from brax import envs as brax_envs
from chex import dataclass, PRNGKey
from dm_env import StepType

from dopamax.environments.environment import EnvState, Environment, TimeStep
from dopamax.typing import Action


@dataclass(frozen=True)
class BraxEnvState(EnvState):
    episode_reward: float
    episode_length: float
    brax_state: brax_envs.State
    time: int


@dataclass(frozen=True)
class BraxEnvironment(Environment, ABC):
    _brax_env: brax_envs.PipelineEnv

    def reset(self, key: PRNGKey) -> Tuple[TimeStep, BraxEnvState]:
        brax_state = self._brax_env.reset(key)

        time_step = TimeStep.restart(brax_state.obs, brax_state.info)
        env_state = BraxEnvState(
            episode_reward=0.0,
            episode_length=0,
            brax_state=brax_state,
            time=0,
        )

        return time_step, env_state

    def step(self, key: PRNGKey, state: BraxEnvState, action: Action) -> Tuple[TimeStep, BraxEnvState]:
        prev_terminal = jnp.bool_(state.brax_state.done)

        new_brax_state = self._brax_env.step(state.brax_state, action)

        reward = new_brax_state.reward * (1 - prev_terminal)
        length = 1 - prev_terminal

        state = BraxEnvState(
            episode_reward=state.episode_reward + reward,
            episode_length=state.episode_length + length,
            brax_state=new_brax_state,
            time=state.time + 1,
        )

        done = jnp.bool_(new_brax_state.done)
        truncate = jnp.bool_(new_brax_state.info["truncation"])

        time_step = TimeStep(
            observation=new_brax_state.obs,
            reward=reward,
            discount=1.0 - jnp.float32(done & ~truncate),
            step_type=jax.lax.select(done, StepType.LAST, StepType.MID),
            info=new_brax_state.info,
        )

        return time_step, state
