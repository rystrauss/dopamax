from abc import ABC
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import pgx
from chex import dataclass, PRNGKey
from dm_env import StepType

from dopamax.environments.environment import Environment, EnvState, TimeStep
from dopamax.spaces import Box, Space, Discrete
from dopamax.typing import Action


@dataclass(frozen=True)
class PGXEnvState(EnvState):
    episode_reward: float
    episode_length: float
    pgx_state: pgx.State


@dataclass(frozen=True)
class PGXEnvironment(Environment, ABC):
    _pgx_env: pgx.Env

    def reset(self, key: PRNGKey) -> Tuple[TimeStep, PGXEnvState]:
        pgx_state = self._pgx_env.init(key)

        time_step = TimeStep.restart(
            pgx_state.observation.astype(jnp.float32),
            {
                "current_player": pgx_state.current_player,
                "legal_action_mask": pgx_state.legal_action_mask,
            },
        )
        env_state = PGXEnvState(
            episode_reward=0.0,
            episode_length=0,
            pgx_state=pgx_state,
        )

        return time_step, env_state

    @property
    def observation_space(self) -> Space:
        return Box(low=-jnp.inf, high=jnp.inf, shape=self._pgx_env.observation_shape, dtype=jnp.float32)

    @property
    def action_space(self) -> Space:
        return Discrete(self._pgx_env.num_actions)

    def step(self, key: PRNGKey, state: PGXEnvState, action: Action) -> Tuple[TimeStep, PGXEnvState]:
        prev_terminal = jnp.squeeze(jnp.bool_(state.pgx_state.terminated | state.pgx_state.truncated))

        new_pgx_state = self._pgx_env.step(state.pgx_state, action, key)

        reward = jnp.squeeze(new_pgx_state.rewards[new_pgx_state.current_player])
        length = 1 - prev_terminal

        state = PGXEnvState(
            episode_reward=state.episode_reward + reward,
            episode_length=state.episode_length + length,
            pgx_state=new_pgx_state,
        )

        done = jnp.squeeze(jnp.bool_(new_pgx_state.terminated | new_pgx_state.truncated))
        truncate = jnp.squeeze(jnp.bool_(new_pgx_state.truncated))

        time_step = TimeStep(
            observation=new_pgx_state.observation.astype(jnp.float32),
            reward=reward,
            discount=1.0 - jnp.float32(done & ~truncate),
            step_type=jax.lax.select(done, StepType.LAST, StepType.MID),
            info={
                "current_player": new_pgx_state.current_player,
                "legal_action_mask": new_pgx_state.legal_action_mask,
            },
        )

        return time_step, state
