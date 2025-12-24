from dopamax.environments.environment import Environment
from dopamax.environments.gymnax import GymnaxEnvironment

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
    if env_name.startswith("gymnax:"):
        try:
            import gymnax
        except ImportError:
            msg = "Unable to import gymnax. Please install gymnax (e.g. via 'pip install gymnax')."
            raise ImportError(msg)

        env, env_params = gymnax.make(env_name[7:], **kwargs)
        return GymnaxEnvironment(env=env, env_params=env_params)

    env_cls = _registry[env_name]
    return env_cls(**kwargs)
