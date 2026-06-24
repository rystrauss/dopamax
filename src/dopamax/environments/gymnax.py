import gymnax
import jax
import jax.numpy as jnp
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
        return spaces.Dict({k: _convert_space(v) for k, v in space.spaces.items()})

    msg = f"Unknown space: {space}"
    raise ValueError(msg)


@dataclass(frozen=True)
class GymnaxEnvState(EnvState):
    gymnax_state: gymnax.environments.EnvState
    prev_terminal: bool


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

    def reset(self, key: PRNGKey) -> tuple[TimeStep, GymnaxEnvState]:
        obs, gymnax_state = self.env.reset(key, self.env_params)
        state = GymnaxEnvState(
            episode_length=0, episode_reward=0.0, gymnax_state=gymnax_state, prev_terminal=jnp.bool_(False)
        )
        # Use the observation gymnax already returned rather than recomputing get_obs (which, for some gymnax envs,
        # additionally requires params/key and/or is stochastic).
        ts = TimeStep.restart(obs)
        return ts, state

    def step(self, key: PRNGKey, state: GymnaxEnvState, action: Action) -> tuple[TimeStep, GymnaxEnvState]:
        obs, gymnax_state, reward, done, info = self.env.step(key, state.gymnax_state, action, self.env_params)

        done = jnp.bool_(done)

        # gymnax folds the time-limit into is_terminal() (and reports discount=0 there), but a timeout is a truncation,
        # not a true termination: the value bootstrap should be preserved. Detect the timeout from the PRE-step time
        # (gymnax auto-resets its internal state on the terminal step, zeroing the post-step time).
        truncate = jnp.bool_(state.gymnax_state.time + 1 >= self.env_params.max_steps_in_episode)

        # Mask reward/length contributions after the first terminal step so the episode_reward/episode_length
        # accumulators plateau at the true single-episode values across gymnax's internal auto-reset (mirrors the
        # brax/pgx wrappers). rollout_truncated resets the carried state on LAST, so training is unaffected.
        mask = 1.0 - jnp.float32(state.prev_terminal)

        ts = TimeStep(
            observation=obs,
            reward=reward * mask,
            discount=1.0 - jnp.float32(done & ~truncate),
            step_type=jax.lax.select(done, StepType.LAST, StepType.MID),
        )

        new_state = GymnaxEnvState(
            episode_reward=state.episode_reward + reward * mask,
            episode_length=state.episode_length + (1 - jnp.int32(state.prev_terminal)),
            gymnax_state=gymnax_state,
            prev_terminal=state.prev_terminal | done,
        )

        return ts, new_state
