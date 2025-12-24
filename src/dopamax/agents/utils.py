
_registry = {}


def register(name: str):
    """Decorator that registers an agent."""

    def _fn(cls):
        _registry[name] = cls
        return cls

    return _fn


def get_agent_cls(agent_name: str):
    """Get an agent class by name

    Args:
        agent_name: The name of the agent to create.

    Returns:
        The agent class.
    """
    return _registry[agent_name]
