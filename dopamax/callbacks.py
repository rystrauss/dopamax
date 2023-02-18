from abc import ABC, abstractmethod
from copy import copy

import jax
from wandb.sdk.wandb_run import Run

from dopamax.agents.agent import TrainState
from dopamax.typing import Metrics


class Callback(ABC):
    """Base class for callbacks that can be passed to an agent's train method."""

    def __call__(self, train_state: TrainState, metrics: Metrics):
        self.on_train_step(train_state, metrics)

    @abstractmethod
    def on_train_step(self, train_state: TrainState, metrics: Metrics):
        """Executes at the end of an agent's train step.

        Args:
            train_state: The current training state.
            metrics: Metrics collected by the agent during the last training step.

        Returns:
            None
        """
        pass


class WandbCallback(Callback):
    """Callback that logs metrics to Weights & Biases.

    Args:
        run: The wandb run to log to.
    """

    def __init__(self, run: Run):
        self._run = run

    def on_train_step(self, train_state: TrainState, metrics: Metrics):
        to_log = copy(metrics)

        to_log["timesteps"] = train_state.total_timesteps
        to_log["episodes"] = train_state.total_episodes

        to_log = jax.tree_map(lambda x: x.item(), to_log)

        self._run.log(to_log, step=int(train_state.train_step), commit=True)
