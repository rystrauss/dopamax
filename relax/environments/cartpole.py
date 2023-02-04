from typing import Tuple

import jax
import jax.numpy as jnp
from chex import dataclass, PRNGKey
from dm_env import StepType

from relax.environments.environment import EnvState, Environment, TimeStep
from relax.environments.utils import register
from relax.spaces import Space, Box, Discrete
from relax.typing import Action, Observation

_NAME = "CartPole"


@dataclass(frozen=True)
class CartPoleEnvState(EnvState):
    episode_reward: float
    episode_length: float
    x: float
    x_dot: float
    theta: float
    theta_dot: float
    time: int

    def to_obs(self) -> Observation:
        return jnp.array([self.x, self.x_dot, self.theta, self.theta_dot])


@register(_NAME)
@dataclass(frozen=True)
class CartPole(Environment):
    """The CartPole environment, as defined by Barto, Sutton, and Anderson.

    References:
        This implementation is adapted from:
        https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/classic_control/cartpole.py
    """

    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    total_mass: float = 1.0 + 0.1
    length: float = 0.5
    polemass_length: float = 0.05
    force_mag: float = 10.0
    tau: float = 0.02
    theta_threshold_radians: float = 12 * 2 * jnp.pi / 360
    x_threshold: float = 2.4

    @property
    def name(self) -> str:
        return _NAME

    @property
    def max_episode_length(self) -> int:
        return 500

    @property
    def observation_space(self) -> Space:
        high = jnp.array(
            [
                self.x_threshold * 2,
                jnp.finfo(jnp.float32).max,
                self.theta_threshold_radians * 2,
                jnp.finfo(jnp.float32).max,
            ]
        )
        return Box(low=-high, high=high, shape=(4,))

    @property
    def action_space(self) -> Space:
        return Discrete(2)

    def _is_terminal(self, state: CartPoleEnvState) -> Tuple[bool, bool]:
        done1 = jnp.logical_or(
            state.x < -self.x_threshold,
            state.x > self.x_threshold,
        )

        done2 = jnp.logical_or(
            state.theta < -self.theta_threshold_radians,
            state.theta > self.theta_threshold_radians,
        )

        truncate = state.time >= self.max_episode_length
        done = jnp.logical_or(jnp.logical_or(done1, done2), truncate)

        return done, truncate

    def reset(self, key: PRNGKey) -> Tuple[TimeStep, CartPoleEnvState]:
        x, x_dot, theta, theta_dot = jax.random.uniform(key, minval=-0.05, maxval=0.05, shape=(4,))

        state = CartPoleEnvState(
            episode_reward=0.0,
            episode_length=0,
            x=x,
            x_dot=x_dot,
            theta=theta,
            theta_dot=theta_dot,
            time=0,
        )
        time_step = TimeStep.restart(state.to_obs())

        return time_step, state

    def step(self, key: PRNGKey, state: CartPoleEnvState, action: Action) -> Tuple[TimeStep, CartPoleEnvState]:
        prev_terminal, _ = self._is_terminal(state)

        force = self.force_mag * action - self.force_mag * (1 - action)
        costheta = jnp.cos(state.theta)
        sintheta = jnp.sin(state.theta)

        temp = (force + self.polemass_length * state.theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = state.x + self.tau * state.x_dot
        x_dot = state.x_dot + self.tau * xacc
        theta = state.theta + self.tau * state.theta_dot
        theta_dot = state.theta_dot + self.tau * thetaacc

        reward = 1.0 - prev_terminal

        state = CartPoleEnvState(
            episode_reward=state.episode_reward + reward,
            episode_length=state.episode_length + 1,
            x=x,
            x_dot=x_dot,
            theta=theta,
            theta_dot=theta_dot,
            time=state.time + 1,
        )
        done, truncate = self._is_terminal(state)

        time_step = TimeStep(
            observation=state.to_obs(),
            reward=reward,
            discount=1.0 - jnp.float32(done & ~truncate),
            step_type=jax.lax.select(done, StepType.LAST, StepType.MID),
        )

        return time_step, state
