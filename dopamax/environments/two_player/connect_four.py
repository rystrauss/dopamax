from typing import Tuple, Callable

import jax.lax
import jax.numpy as jnp
import numpy as np
from chex import PRNGKey, dataclass, Array
from dm_env import StepType

from dopamax.environments.environment import TimeStep
from dopamax.environments.two_player.base import TwoPlayerZeroSumEnvState, TwoPlayerZeroSumEnvironment
from dopamax.environments.utils import register
from dopamax.spaces import Space, Discrete, Box, Dict
from dopamax.typing import Action, Observation

_NAME = "ConnectFour"


def _build_win_checker():
    weight = np.zeros((4, 4, 1, 6), dtype=np.float32)
    weight[0, :, :, 0] = 1
    weight[:, 0, :, 1] = 1
    weight[-1, :, :, 2] = 1
    weight[:, -1, :, 3] = 1
    for i in range(4):
        weight[i, i, :, 4] = 1
        weight[i, 3 - i, :, 5] = 1

    weight = np.transpose(weight, (3, 2, 0, 1))

    return lambda inputs: jax.lax.conv(
        inputs[None, None, :, :].astype(jnp.float32),
        weight,
        window_strides=(1, 1),
        padding="VALID",
    )


@dataclass(frozen=True)
class ConnectFourEnvState(TwoPlayerZeroSumEnvState):
    episode_reward: float
    episode_length: float
    current_player: int
    time: int
    board: Array
    column_counts: Array

    def to_obs(self) -> Observation:
        return {
            "observation": self.board[::-1, :],
            "invalid_actions": (self.column_counts >= self.board.shape[0]).astype(jnp.float32),
        }


@register(_NAME)
@dataclass(frozen=True)
class ConnectFour(TwoPlayerZeroSumEnvironment):
    _num_rows: int = 6
    _num_cols: int = 7
    _win_checker: Callable = _build_win_checker()

    @property
    def name(self) -> str:
        return _NAME

    @property
    def max_episode_length(self) -> int:
        return self._num_cols * self._num_rows

    @property
    def observation_space(self) -> Space:
        return Dict(
            {
                "observation": Box(low=-1, high=1, shape=(self._num_rows, self._num_cols)),
                "invalid_actions": Box(low=0, high=1, shape=(self._num_cols,)),
            }
        )

    @property
    def action_space(self) -> Space:
        return Discrete(self._num_cols)

    def reset(self, key: PRNGKey) -> Tuple[TimeStep, ConnectFourEnvState]:
        board = jnp.zeros((self._num_rows, self._num_cols), dtype=jnp.int32)
        column_counts = jnp.zeros((self._num_cols,), dtype=jnp.int32)

        state = ConnectFourEnvState(
            episode_reward=0.0,
            episode_length=0,
            time=0,
            board=board,
            column_counts=column_counts,
            current_player=1,
        )
        time_step = TimeStep.restart(state.to_obs())

        return time_step, state

    def _check_winner(self, board: Array) -> int:
        x = self._win_checker(board)
        m = jnp.max(jnp.abs(x))
        m1 = jnp.where(m == jnp.max(x), 1, -1)
        return jnp.where(m == 4, m1, 0)

    def step(self, key: PRNGKey, state: ConnectFourEnvState, action: Action) -> Tuple[TimeStep, ConnectFourEnvState]:
        col_count = state.column_counts[action]
        invalid_move = col_count >= self._num_rows

        prev_winnner = self._check_winner(state.board)
        prev_terminated = prev_winnner != 0

        new_board = jax.lax.select(
            prev_terminated,
            state.board,
            state.board.at[col_count, action].set(state.current_player),
        )

        winner = self._check_winner(new_board)
        reward = winner * state.current_player

        new_column_counts = state.column_counts.at[action].set(state.column_counts[action] + 1)
        current_player = -state.current_player
        count = jnp.sum(new_column_counts)

        terminated = jnp.logical_or(prev_terminated, reward != 0)
        terminated = jnp.logical_or(terminated, count >= self._num_cols * self._num_rows)
        terminated = jnp.logical_or(terminated, invalid_move)

        reward = jnp.where(invalid_move, -1.0, reward) * (1.0 - prev_terminated)
        length = 1 - prev_terminated

        state = ConnectFourEnvState(
            episode_reward=state.episode_reward + reward,
            episode_length=state.episode_length + length,
            time=state.time + 1,
            board=new_board,
            column_counts=new_column_counts,
            current_player=current_player,
        )

        time_step = TimeStep(
            observation=state.to_obs(),
            reward=reward,
            discount=current_player * jnp.float32(~terminated),
            step_type=jax.lax.select(terminated, StepType.LAST, StepType.MID),
        )

        return time_step, state

    def render(self, state: ConnectFourEnvState) -> np.ndarray:
        raise NotImplementedError("Rendering is not implemented for ConnectFour.")
