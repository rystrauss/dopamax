import jax.numpy as jnp
from brax import envs as brax_envs
from chex import dataclass

from dopamax.environments.brax.base import BraxEnvironment
from dopamax.environments.utils import register
from dopamax.spaces import Space, Box

_NAME = "InvertedPendulum"


@register(_NAME)
@dataclass(frozen=True)
class InvertedPendulum(BraxEnvironment):
    def __init__(self):
        brax_env = brax_envs.create("inverted_double_pendulum", auto_reset=False)
        super(InvertedPendulum, self).__init__(_brax_env=brax_env)

    @property
    def name(self) -> str:
        return _NAME

    @property
    def max_episode_length(self) -> int:
        return 1000

    @property
    def observation_space(self) -> Space:
        return Box(low=-jnp.inf, high=jnp.inf, shape=(8,))

    @property
    def action_space(self) -> Space:
        return Box(low=-3.0, high=3.0, shape=(1,))
