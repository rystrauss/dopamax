from dopamax.environments.environment import Environment

_registry = {}


def register(name: str):
    """Decorator that registers an environment."""

    def _fn(cls):
        _registry[name] = cls
        return cls

    return _fn


def make_env(env_name: str, **kwargs) -> Environment:
    """Creates an environment.

    Args:
        env_name: The name of the environment to make.
        **kwargs: Keyword args to pass to the environment constructor.

    Returns:
        The environment.
    """
    env_cls = _registry[env_name]
    return env_cls(**kwargs)
