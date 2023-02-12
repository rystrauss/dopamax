from abc import abstractmethod, ABC
from dataclasses import field
from typing import Dict, Any, Tuple, Optional

import jax.numpy as jnp
import numpy as np
from chex import PRNGKey, dataclass
from dm_env import StepType

from dopamax.spaces import Space
from dopamax.typing import Action, Observation


@dataclass(frozen=True)
class EnvState:
    """Base class that represents environment state.

    Each environment should have its own state class that inherits from this class.
    """

    episode_reward: float
    episode_length: float


@dataclass(frozen=True)
class TimeStep:
    """Container for environment time step data.

    This class contains data returned by the environment at each time step, such as the reward and the next observation.

    References:
        https://github.com/deepmind/dm_env/blob/master/dm_env/_environment.py#L25
    """

    observation: Observation
    reward: Optional[float]
    discount: Optional[float]
    step_type: StepType
    info: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def restart(cls, observation: Observation, info: Optional[Dict[str, Any]] = None) -> "TimeStep":
        """Creates a time step for the first step in an episode.

        Note that `reward` and `discount` are set to `jnp.nan` to indicate that they are undefined for the first step
        of the episode.

        Args:
            observation: The observation from the environment.
            info: Optional dictionary with extra information.

        Returns:
            A `TimeStep` for the first step in an episode.
        """
        return cls(
            observation=observation,
            reward=jnp.nan,
            discount=jnp.nan,
            step_type=StepType.FIRST,
            info=info or {},
        )

    @classmethod
    def transition(
        cls, observation: Observation, reward: float, discount: float = 1.0, info: Optional[Dict[str, Any]] = None
    ) -> "TimeStep":
        """Creates a time step for a transition in the middle of an episode.

        Args:
            observation: The observation from the environment.
            reward: The reward from the environment.
            discount: The discount from the environment.
            info: Optional dictionary with extra information.

        Returns:
            A `TimeStep` for a transition in the middle of an episode.
        """
        return cls(
            observation=observation,
            reward=reward,
            discount=discount,
            step_type=StepType.MID,
            info=info or {},
        )

    @classmethod
    def termination(cls, observation: Observation, reward: float, info: Optional[Dict[str, Any]] = None) -> "TimeStep":
        """Creates a time step for the last step in an episode due to a terminal environment state.

        The discount will be set to 0.0, and the step type will be set to `StepType.LAST`.

        Args:
            observation: The observation from the environment.
            reward: The reward from the environment.
            info: Optional dictionary with extra information.

        Returns:
            A `TimeStep` for the last step in an episode due to a terminal environment state.
        """
        return cls(
            observation=observation,
            reward=reward,
            discount=0.0,
            step_type=StepType.LAST,
            info=info or {},
        )

    @classmethod
    def truncation(
        cls, observation: Observation, reward: float, discount: float = 1.0, info: Optional[Dict[str, Any]] = None
    ) -> "TimeStep":
        """Creates a time step for the last step in an episode due to a truncation.

        A truncation scenario is when we want to end the episode, but the environment was not necessarily in a terminal
        state. Thus, the discount will be set to 1.0 by default, and the step type will be set to `StepType.LAST`.

        Args:
            observation: The observation from the environment.
            reward: The reward from the environment.
            discount: The discount from the environment.
            info: Optional dictionary with extra information.

        Returns:
            A `TimeStep` for the last step in an episode due to a truncation.
        """
        return cls(
            observation=observation,
            reward=reward,
            discount=discount,
            step_type=StepType.LAST,
            info=info or {},
        )


@dataclass(frozen=True)
class Environment(ABC):
    """Abstract base class for environments."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the environment."""
        pass

    @property
    @abstractmethod
    def max_episode_length(self) -> int:
        """The maximum number of allowed steps in an episode."""
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Space:
        """The observation space of the environment."""
        pass

    @property
    @abstractmethod
    def action_space(self) -> Space:
        """The action space of the environment."""
        pass

    @property
    def renderable(self) -> bool:
        """Whether the environment can be rendered."""
        return False

    @property
    def fps(self) -> Optional[int]:
        """The frames per second of rendered frames."""
        return None

    @property
    def render_shape(self) -> Optional[Tuple[int, int, int]]:
        """The shape of rendered frames."""
        return None

    @abstractmethod
    def reset(self, key: PRNGKey) -> Tuple[TimeStep, EnvState]:
        """Resets the environment to the start of a new episode.

        Args:
            key: A `PRNGKey` used to sample the initial state.

        Returns:
            A tuple containing the initial time step and the initial environment state.
        """
        pass

    @abstractmethod
    def step(self, key: PRNGKey, state: EnvState, action: Action) -> Tuple[TimeStep, EnvState]:
        """Steps the environment forward by one time step.

        Args:
            key: A `PRNGKey` used to seed any randomness involved in the environment transition.
            state: A `EnvState` representing the current state of the environment.
            action: The action to take in the environment.

        Returns:
            A tuple containing the time step and the next environment state.
        """
        pass

    def render(self, state: EnvState) -> np.ndarray:
        """Renders the current state of the environment as an RGB frame.

        Args:
            state: A `EnvState` representing the current state of the environment.

        Returns:
            An RGB frame of the current state of the environment.
        """
        raise NotImplementedError("This environment does not support rendering.")
