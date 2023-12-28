import pgx
from chex import dataclass

from dopamax.environments.pgx.base import PGXEnvironment
from dopamax.environments.utils import register

_NAME = "ConnectFour"


@register(_NAME)
@dataclass(frozen=True)
class ConnectFour(PGXEnvironment):
    def __init__(self):
        pgx_env = pgx.make("connect_four")
        super(ConnectFour, self).__init__(_pgx_env=pgx_env)

    @property
    def max_episode_length(self) -> int:
        return 7 * 6

    @property
    def name(self) -> str:
        return _NAME
