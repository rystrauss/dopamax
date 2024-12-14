import pgx
from chex import dataclass

from dopamax.environments.pgx.base import PGXEnvironment
from dopamax.environments.utils import register

_NAME = "Chess"


@register(_NAME)
@dataclass(frozen=True)
class Chess(PGXEnvironment):
    def __init__(self):
        pgx_env = pgx.make("chess")
        super(Chess, self).__init__(_pgx_env=pgx_env)

    @property
    def max_episode_length(self) -> int:
        return 512  # From AlphaZero paper

    @property
    def name(self) -> str:
        return _NAME
