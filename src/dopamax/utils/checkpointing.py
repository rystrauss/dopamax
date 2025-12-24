"""Utilities for checkpointing and resuming training."""

import os
import pickle
from pathlib import Path
from typing import Any

import haiku as hk
from loguru import logger

from dopamax.agents.base import TrainState


def save_checkpoint(
    checkpoint_dir: str | Path,
    train_state: TrainState,
    step: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a training checkpoint.

    Args:
        checkpoint_dir: Directory to save the checkpoint in.
        train_state: The current training state to save.
        step: The training step number.
        metadata: Optional metadata dictionary to save with the checkpoint.

    Returns:
        The path to the saved checkpoint file.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pkl"

    checkpoint_data = {
        "train_state": train_state,
        "step": step,
        "metadata": metadata or {},
    }

    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint_data, f)

    logger.info(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path: str | Path) -> tuple[TrainState, int, dict[str, Any]]:
    """Load a training checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.

    Returns:
        A tuple of (train_state, step, metadata).

    Raises:
        FileNotFoundError: If the checkpoint file doesn't exist.
        ValueError: If the checkpoint file is corrupted or invalid.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        msg = f"Checkpoint file not found: {checkpoint_path}"
        raise FileNotFoundError(msg)

    try:
        with open(checkpoint_path, "rb") as f:
            checkpoint_data = pickle.load(f)
    except Exception as e:
        msg = f"Failed to load checkpoint from {checkpoint_path}: {e}"
        raise ValueError(msg) from e

    if not isinstance(checkpoint_data, dict) or "train_state" not in checkpoint_data:
        msg = f"Invalid checkpoint format in {checkpoint_path}"
        raise ValueError(msg)

    train_state = checkpoint_data["train_state"]
    step = checkpoint_data.get("step", 0)
    metadata = checkpoint_data.get("metadata", {})

    logger.info(f"Loaded checkpoint from {checkpoint_path} (step {step})")
    return train_state, step, metadata


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Find the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory to search for checkpoints.

    Returns:
        Path to the latest checkpoint, or None if no checkpoints found.
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pkl"))
    if not checkpoints:
        return None

    # Sort by step number (extracted from filename)
    def get_step(path: Path) -> int:
        try:
            return int(path.stem.split("_")[-1])
        except (ValueError, IndexError):
            return -1

    checkpoints.sort(key=get_step, reverse=True)
    return checkpoints[0]


def save_params(params_path: str | Path, params: hk.Params) -> Path:
    """Save model parameters to a file.

    Args:
        params_path: Path to save the parameters to.
        params: The parameters to save.

    Returns:
        The path to the saved parameters file.
    """
    params_path = Path(params_path)
    params_path.parent.mkdir(parents=True, exist_ok=True)

    with open(params_path, "wb") as f:
        pickle.dump(params, f)

    logger.info(f"Saved parameters to {params_path}")
    return params_path


def load_params(params_path: str | Path) -> hk.Params:
    """Load model parameters from a file.

    Args:
        params_path: Path to the parameters file.

    Returns:
        The loaded parameters.

    Raises:
        FileNotFoundError: If the parameters file doesn't exist.
        ValueError: If the parameters file is corrupted.
    """
    params_path = Path(params_path)

    if not params_path.exists():
        msg = f"Parameters file not found: {params_path}"
        raise FileNotFoundError(msg)

    try:
        with open(params_path, "rb") as f:
            params = pickle.load(f)
    except Exception as e:
        msg = f"Failed to load parameters from {params_path}: {e}"
        raise ValueError(msg) from e

    logger.info(f"Loaded parameters from {params_path}")
    return params

