from typing import Tuple

import gymnax
import jax
from chex import PRNGKey, dataclass
from dm_env import StepType
from gymnax.environments.spaces import Space

from dopamax import spaces
from dopamax.environments.environment import Environment, EnvState, TimeStep
from dopamax.typing import Action


def _convert_space(space: gymnax.environments.spaces.Space) -> spaces.Space:
    if isinstance(space, gymnax.environments.spaces.Box):
        return spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)

    if isinstance(space, gymnax.environments.spaces.Discrete):
        return spaces.Discrete(space.n, dtype=space.dtype)

    if isinstance(space, gymnax.environments.spaces.Dict):
        return spaces.Dict(spaces=space.spaces)

    raise ValueError(f"Unknown space: {space}")


@dataclass(frozen=True)
class GymnaxEnvState(EnvState):
    gymnax_state: gymnax.environments.EnvState


@dataclass(frozen=True)
class GymnaxEnvironment(Environment):
    env: gymnax.environments.environment.Environment
    env_params: gymnax.environments.environment.EnvParams

    @property
    def name(self) -> str:
        return self.env.name

    @property
    def max_episode_length(self) -> int:
        return self.env_params.max_steps_in_episode

    @property
    def observation_space(self) -> Space:
        return _convert_space(self.env.observation_space(self.env_params))

    @property
    def action_space(self) -> Space:
        return _convert_space(self.env.action_space(self.env_params))

    def reset(self, key: PRNGKey) -> Tuple[TimeStep, GymnaxEnvState]:
        obs, gymnax_state = self.env.reset(key, self.env_params)
        state = GymnaxEnvState(episode_length=0, episode_reward=0.0, gymnax_state=gymnax_state)
        ts = TimeStep.restart(self.env.get_obs(gymnax_state))
        return ts, state

    def step(self, key: PRNGKey, state: GymnaxEnvState, action: Action) -> Tuple[TimeStep, GymnaxEnvState]:
        obs, gymnax_state, reward, done, info = self.env.step(key, state.gymnax_state, action, self.env_params)

        done = jax.numpy.bool_(done)

        ts = TimeStep(
            observation=self.env.get_obs(gymnax_state),
            reward=reward,
            discount=info["discount"],
            step_type=jax.lax.select(done, StepType.LAST, StepType.MID),
        )

        new_state = GymnaxEnvState(
            episode_reward=state.episode_reward + reward,
            episode_length=state.episode_length + 1,
            gymnax_state=gymnax_state,
        )

        return ts, new_state
