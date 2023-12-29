from abc import ABC
from typing import Tuple

import jax
import jax.numpy as jnp
import pgx
from chex import dataclass, PRNGKey
from dm_env import StepType

from dopamax.environments.environment import Environment, EnvState, TimeStep
from dopamax.spaces import Box, Space, Discrete, Dict
from dopamax.typing import Action


@dataclass(frozen=True)
class PGXEnvState(EnvState):
    episode_reward: float
    episode_length: float
    pgx_state: pgx.State


@dataclass(frozen=True)
class PGXEnvironment(Environment, ABC):
    """Abstract base class for PGX environments.

    PGX is a collection of JAX-native implementations of discrete state space environments like Chess, Shogi, and Go.
    This class serves as a wrapper around PGX environments in order to make them conform to the dopamax environment
    API.

    References:
        https://github.com/sotetsuk/pgx
    """
    _pgx_env: pgx.Env

    def reset(self, key: PRNGKey) -> Tuple[TimeStep, PGXEnvState]:
        pgx_state = self._pgx_env.init(key)

        time_step = TimeStep.restart(
            {
                "observation": pgx_state.observation.astype(jnp.float32),
                "invalid_actions": (~pgx_state.legal_action_mask).astype(jnp.float32),
            },
            {
                "current_player": pgx_state.current_player,
                "player_rewards": pgx_state.rewards,
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
        return Dict(
            {
                "observation": Box(low=-jnp.inf, high=jnp.inf, shape=self._pgx_env.observation_shape),
                "invalid_actions": Box(low=0, high=1, shape=(self._pgx_env.num_actions,)),
            }
        )

    @property
    def action_space(self) -> Space:
        return Discrete(self._pgx_env.num_actions)

    def step(self, key: PRNGKey, state: PGXEnvState, action: Action) -> Tuple[TimeStep, PGXEnvState]:
        prev_terminal = jnp.bool_(state.pgx_state.terminated | state.pgx_state.truncated)

        new_pgx_state = self._pgx_env.step(state.pgx_state, action, key)

        reward = new_pgx_state.rewards[state.pgx_state.current_player]
        length = 1 - prev_terminal

        state = PGXEnvState(
            episode_reward=state.episode_reward + reward,
            episode_length=state.episode_length + length,
            pgx_state=new_pgx_state,
        )

        done = jnp.bool_(new_pgx_state.terminated | new_pgx_state.truncated)
        truncate = jnp.bool_(new_pgx_state.truncated)

        time_step = TimeStep(
            observation={
                "observation": new_pgx_state.observation.astype(jnp.float32),
                "invalid_actions": (~new_pgx_state.legal_action_mask).astype(jnp.float32),
            },
            reward=reward,
            discount=1.0 - jnp.float32(done & ~truncate),
            step_type=jax.lax.select(done, StepType.LAST, StepType.MID),
            info={
                "current_player": new_pgx_state.current_player,
                "player_rewards": new_pgx_state.rewards,
            },
        )

        return time_step, state
