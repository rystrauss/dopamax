from abc import ABC, abstractmethod
from typing import Tuple, Optional, Sequence

import haiku as hk
from brax.training.replay_buffers import ReplayBufferState
from chex import dataclass, PRNGKey
from ml_collections import ConfigDict

from dopamax.environments.environment import Environment
from dopamax.spaces import Space
from dopamax.typing import Observation, Action, Metrics

# The number of most recent episodes to average over when logging performance.
_EPISODE_BUFFER_SIZE = 128


@dataclass(frozen=True)
class TrainState:
    """The training state of an agent."""

    key: PRNGKey
    train_step: int
    total_timesteps: int
    total_episodes: int
    episode_buffer_state: ReplayBufferState


class Agent(ABC):
    """Abstract base class for an RL agent.

    Subclasses are intended to be implemented in accordance with the Anakin Podracer architecture.
    See: https://arxiv.org/abs/2104.06272

    This means that we assume the agent's train step can be vectorized across a batch within each device, and can be
    further parallelized across multiple devices.

    Args:
        env: The environment to interact with.
        config: The configuration dictionary for the agent.
    """

    def __init__(self, env: Environment, config: ConfigDict):
        self._env = env
        self._config = self.default_config()
        self._config.update(config)
        self._config.lock()

    @property
    def env(self) -> Environment:
        """The environment that the agent interacts with."""
        return self._env

    @property
    def config(self) -> ConfigDict:
        """The configuration dictionary for the agent."""
        return self._config

    @staticmethod
    def default_config() -> ConfigDict:
        """Returns the default configuration dictionary for all agents."""
        return ConfigDict()

    @property
    def observation_space(self) -> Space:
        """The observation space of the agent's environment."""
        return self._env.observation_space

    @property
    def action_space(self) -> Space:
        """The action space of the agent's environment."""
        return self._env.action_space

    @abstractmethod
    def compute_action(self, params: hk.Params, key: PRNGKey, observation: Observation, **kwargs) -> Action:
        """Computes an action for given observations.

        Args:
            params: The agent's params.
            key: A PRNG key.
            observation: Observations to compute actions for. It is assumed that the observations will have a batch
                dimension.
            **kwargs: Additional keyword arguments that some agents may provide for customization of how actions
                are computed.

        Returns:
            An array of actions corresponding to the given observations.
        """
        pass

    @abstractmethod
    def initial_train_state(self, consistent_key: PRNGKey, divergent_key: PRNGKey) -> TrainState:
        """Initializes the agent's training state.

        Args:
            consistent_key: A PRNGKey that is used to initialize pieces of the train state which should be
                consistent across all devices (e.g. network parameters).
            divergent_key: A PRNGKey that is used to initialize pieces of the train state which should be
                different across all devices (e.g. environment state).

        Returns:
            The initial training state.
        """
        pass

    @abstractmethod
    def train_step(self, train_state: TrainState) -> Tuple[TrainState, Metrics]:
        """Performs a single training step.

        A training step should include experience collection and parameter updates.

        Args:
            train_state: The current training state.

        Returns:
            train_state: The updated training state.
            metrics: A dictionary of metrics to be logged.
        """
        pass

    @abstractmethod
    def train(
        self, key: PRNGKey, num_iterations: int, callback_freq: int = 100, callbacks: Optional[Sequence] = None
    ) -> hk.Params:
        """Trains the agent.

        Args:
            key: The PRNGKey to use to seed and randomness involved in training.
            num_iterations: The number of times to call the train_step function.
            callback_freq: The frequency, in terms of call to `train_step`, at which to execute any callbacks.
            callbacks: A list of callbacks.

        Returns:
            The final parameters of the agent.
        """
        pass
