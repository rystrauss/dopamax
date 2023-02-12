from typing import Type

from dopamax.agents.agent import Agent

_registry = {}


def register(name: str):
    """Decorator that registers an agent."""

    def _fn(cls):
        _registry[name] = cls
        return cls

    return _fn


def get_agent_cls(agent_name: str) -> Type[Agent]:
    """Get an agent class by name

    Args:
        agent_name: The name of the agent to create.

    Returns:
        The agent class.
    """
    return _registry[agent_name]
