import math
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from chex import dataclass, PRNGKey
from dm_env import StepType

from dopamax.environments.environment import EnvState, Environment, TimeStep
from dopamax.environments.mountain_car import MountainCarEnvState
from dopamax.environments.utils import register
from dopamax.spaces import Space, Box
from dopamax.typing import Action

_NAME = "MountainCarContinuous"

MountainCarContinuousEnvState = MountainCarEnvState


@register(_NAME)
@dataclass(frozen=True)
class MountainCarContinuous(Environment):
    """The continuous version of the MountainCar environment.

    References:
        This implementation is adapted from:
        https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/classic_control/continuous_mountain_car.py
    """

    min_action: float = -1.0
    max_action: float = 1.0
    min_position: float = -1.2
    max_position: float = 0.6
    max_speed: float = 0.07
    goal_position: float = 0.45
    goal_velocity: float = 0.0
    power: float = 0.0015
    gravity: float = 0.0025

    @property
    def name(self) -> str:
        return _NAME

    @property
    def max_episode_length(self) -> int:
        return 999

    @property
    def observation_space(self) -> Space:
        low = jnp.array([self.min_position, -self.max_speed], dtype=jnp.float32)
        high = jnp.array([self.max_position, self.max_speed], dtype=jnp.float32)
        return Box(low=low, high=high, shape=(2,))

    @property
    def action_space(self) -> Space:
        return Box(low=self.min_action, high=self.max_action, shape=(1,))

    @property
    def renderable(self) -> bool:
        return True

    @property
    def fps(self) -> Optional[int]:
        return 30

    @property
    def render_shape(self) -> Optional[Tuple[int, int, int]]:
        return 400, 600, 3

    def _is_terminal(self, state: MountainCarContinuousEnvState) -> Tuple[bool, bool]:
        done = jnp.logical_and(state.position >= self.goal_position, state.velocity >= self.goal_velocity)
        truncate = state.time >= self.max_episode_length
        done = jnp.logical_or(done, truncate)

        return done, truncate

    def reset(self, key: PRNGKey) -> Tuple[TimeStep, EnvState]:
        position = jax.random.uniform(key, shape=(), minval=-0.6, maxval=-0.4)

        state = MountainCarEnvState(
            episode_reward=0.0,
            episode_length=0,
            position=position,
            velocity=0.0,
            time=0,
        )
        time_step = TimeStep.restart(state.to_obs())

        return time_step, state

    def step(
        self, key: PRNGKey, state: MountainCarContinuousEnvState, action: Action
    ) -> Tuple[TimeStep, MountainCarContinuousEnvState]:
        prev_terminal, _ = self._is_terminal(state)

        action = jnp.squeeze(action, 0)

        force = jnp.clip(action, self.min_action, self.max_action)
        velocity = state.velocity + (force * self.power - jnp.cos(3 * state.position) * self.gravity)
        velocity = jnp.clip(velocity, -self.max_speed, self.max_speed)
        position = state.position + velocity
        position = jnp.clip(position, self.min_position, self.max_position)
        velocity = velocity * (1 - (position >= self.goal_position) * (velocity < 0))

        reward = -0.1 * action**2 + 100 * ((position >= self.goal_position) * (velocity >= self.goal_velocity))
        reward *= ~prev_terminal

        length = 1 - prev_terminal

        state = MountainCarEnvState(
            episode_reward=state.episode_reward + reward,
            episode_length=state.episode_length + length,
            position=position,
            velocity=velocity,
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

    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    def render(self, state: MountainCarContinuousEnvState) -> np.ndarray:
        import pygame
        from pygame import gfxdraw

        screen_width, screen_height = 600, 400

        screen = pygame.Surface((screen_width, screen_height))

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        surf = pygame.Surface((screen_width, screen_height))
        surf.fill((255, 255, 255))

        pos = state.position

        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        xys = list(zip((xs - self.min_position) * scale, ys * scale))

        pygame.draw.aalines(surf, points=xys, closed=False, color=(0, 0, 0))

        clearance = 10

        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            coords.append(
                (
                    c[0] + (pos - self.min_position) * scale,
                    c[1] + clearance + self._height(pos) * scale,
                )
            )

        gfxdraw.aapolygon(surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(surf, coords, (0, 0, 0))

        for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            wheel = (
                int(c[0] + (pos - self.min_position) * scale),
                int(c[1] + clearance + self._height(pos) * scale),
            )

            gfxdraw.aacircle(surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128))
            gfxdraw.filled_circle(surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128))

        flagx = int((self.goal_position - self.min_position) * scale)
        flagy1 = int(self._height(self.goal_position) * scale)
        flagy2 = flagy1 + 50
        gfxdraw.vline(surf, flagx, flagy1, flagy2, (0, 0, 0))

        gfxdraw.aapolygon(
            surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        gfxdraw.filled_polygon(
            surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )

        surf = pygame.transform.flip(surf, False, True)
        screen.blit(surf, (0, 0))

        return np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
