from abc import ABC

from chex import dataclass

from dopamax.environments.environment import Environment, EnvState


@dataclass(frozen=True)
class TwoPlayerZeroSumEnvState(EnvState):
    episode_reward: float
    episode_length: float
    current_player: int


@dataclass(frozen=True)
class TwoPlayerZeroSumEnvironment(Environment, ABC):
    pass
