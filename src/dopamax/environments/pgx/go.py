import pgx
from chex import dataclass

from dopamax.environments.pgx.base import PGXEnvironment
from dopamax.environments.utils import register

_NAME_9x9 = "Go9x9"
_NAME_19x19 = "Go19x19"


@register(_NAME_9x9)
@dataclass(frozen=True)
class Go9x9(PGXEnvironment):
    def __init__(self):
        pgx_env = pgx.make("go_9x9")
        super(Go9x9, self).__init__(_pgx_env=pgx_env)

    @property
    def max_episode_length(self) -> int:
        return 9 * 9 * 2

    @property
    def name(self) -> str:
        return _NAME_9x9


@register(_NAME_19x19)
@dataclass(frozen=True)
class Go19x19(PGXEnvironment):
    def __init__(self):
        pgx_env = pgx.make("go_19x19")
        super(Go19x19, self).__init__(_pgx_env=pgx_env)

    @property
    def max_episode_length(self) -> int:
        return 19 * 19 * 2

    @property
    def name(self) -> str:
        return _NAME_19x19
