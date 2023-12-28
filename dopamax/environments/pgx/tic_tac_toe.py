import pgx
from chex import dataclass

from dopamax.environments.pgx.base import PGXEnvironment
from dopamax.environments.utils import register

_NAME = "TicTacToe"


@register(_NAME)
@dataclass(frozen=True)
class TicTacToe(PGXEnvironment):
    def __init__(self):
        pgx_env = pgx.make("tic_tac_toe")
        super(TicTacToe, self).__init__(_pgx_env=pgx_env)

    @property
    def max_episode_length(self) -> int:
        return 9

    @property
    def name(self) -> str:
        return _NAME
