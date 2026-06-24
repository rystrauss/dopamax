import importlib.util
import inspect
import os
import pickle
import random
from functools import partial

import click
import haiku as hk
import jax
import jax.random
import numpy as np
import yaml
from chex import PRNGKey
from dm_env import StepType
from loguru import logger
from ml_collections import ConfigDict
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

import wandb
from dopamax.agents.utils import get_agent_cls
from dopamax.callbacks import WandbCallback
from dopamax.environments import make_env
from dopamax.environments.visualization import render_trajectory
from dopamax.rollouts import SampleBatch, rollout_episode
from dopamax.typing import Observation

console = Console()


@click.group()
def cli():
    """Dopamax provides a collection of reinforcement learning agents and environments implemented in pure JAX."""


@cli.command(short_help="Train an agent.")
@click.option(
    "--config",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    required=True,
    help="Path to the training config YAML file.",
)
@click.option("--offline", is_flag=True, help="Run in offline mode, without logging to W&B.")
@click.option(
    "--profiler_port",
    type=click.INT,
    default=None,
    help="The port number on which to run a JAX profiler server during training.",
)
def train(config, offline, profiler_port):
    """Trains an agent using the provided config file, and logs results to W&B."""
    with open(config) as f:
        config = ConfigDict(yaml.safe_load(f))

    if "seed" not in config:
        config.seed = random.randint(0, 2**32 - 1)

    env = make_env(config.env_name)
    agent_cls = get_agent_cls(config.agent_name)
    agent = agent_cls(env, config.agent_config)

    run = wandb.init(
        project="dopamax",
        job_type="train",
        config=config.to_dict(),
        mode="disabled" if offline else "online",
    )

    key = jax.random.PRNGKey(config.seed)

    if profiler_port is not None:
        jax.profiler.start_server(profiler_port)

    params = agent.train(
        key,
        config.train_iterations,
        callback_freq=config.get("callback_freq", 100),
        callbacks=[WandbCallback(run)],
    )

    if profiler_port is not None:
        jax.profiler.stop_server()

    if not offline:
        params_path = os.path.join(run.dir, "params.pkl")

        with open(params_path, "wb") as f:
            pickle.dump(params, f)

        params_artifact = wandb.Artifact(
            config.env_name.replace(":", "-") + "-" + config.agent_name + "-agent", type="agent"
        )
        params_artifact.add_file(params_path)
        run.log_artifact(params_artifact)


@cli.command(short_help="Evaluate an agent.")
@click.option(
    "--agent_artifact",
    type=click.STRING,
    required=True,
    help="W&B artifact reference for the trained agent (e.g. entity/project/<env_name>-<agent_name>-agent:version).",
)
@click.option("--num_episodes", type=click.INT, required=True, help="The number of episodes to evaluate.")
@click.option(
    "--render",
    type=click.BOOL,
    default=False,
    help="Whether to render the episodes. Note that this will usually significantly slow down the evaluation process, "
    "since environment rendering is not a pure JAX function and requires callbacks to the host.",
)
@click.option(
    "--seed",
    type=click.INT,
    default=None,
    help="Random seed for reproducible evaluation. Randomly drawn if not provided.",
)
def evaluate(agent_artifact, num_episodes, render, seed):
    """Evaluates a trained agent and logs the results to W&B."""
    run = wandb.init(project="dopamax", job_type="evaluate", config={"num_episodes": num_episodes})
    artifact = run.use_artifact(agent_artifact, type="agent")
    artifact_dir = artifact.download()

    train_run_config = ConfigDict(artifact.logged_by().config)
    run.config["env_name"] = train_run_config.env_name
    run.config["agent_name"] = train_run_config.agent_name

    env = make_env(train_run_config.env_name)
    agent_cls = get_agent_cls(train_run_config.agent_name)
    agent = agent_cls(env, train_run_config.agent_config)

    params_path = os.path.join(artifact_dir, "params.pkl")
    with open(params_path, "rb") as f:
        params = pickle.load(f)

    # Some agents (e.g. AlphaZero) require the environment state in compute_action to seed their search.
    needs_env_state = "env_state" in inspect.signature(agent.compute_action).parameters

    rollout_fn = jax.jit(rollout_episode, static_argnums=(0, 1, 4, 5))

    if needs_env_state:

        def policy_fn(params: hk.Params, key: PRNGKey, observation: Observation, env_state):
            action = hk.expand_apply(lambda obs, es: agent.compute_action(params, key, obs, es))(observation, env_state)
            return action, {}
    else:

        def policy_fn(params: hk.Params, key: PRNGKey, observation: Observation):
            return hk.expand_apply(partial(agent.compute_action, params, key))(observation), {}

    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    run.config["eval_seed"] = seed
    prng = hk.PRNGSequence(seed)

    rewards, lengths, renders = [], [], []

    for _ in tqdm(range(num_episodes), unit="episodes"):
        rollout_data = rollout_fn(
            env, policy_fn, params, prng.next(), return_env_states=render, pass_env_state_to_policy=needs_env_state
        )

        # rollout_episode scans the full horizon and (for auto-resetting envs) may contain multiple episodes; read the
        # accumulators at the FIRST terminal step so we report the first episode's return/length, not a sum. Falls
        # back to the last step when the episode never terminates within max_episode_length.
        step_types = np.asarray(rollout_data[SampleBatch.STEP_TYPE])
        last_indices = np.argwhere(step_types == StepType.LAST)
        first_last = int(last_indices[0][0]) if len(last_indices) else -1
        rewards.append(rollout_data[SampleBatch.EPISODE_REWARD][first_last])
        lengths.append(rollout_data[SampleBatch.EPISODE_LENGTH][first_last])

        if render:
            env_states = jax.tree.map(lambda x: x[: first_last + 1], rollout_data[SampleBatch.ENVIRONMENT_STATE])
            renders.append(render_trajectory(env, env_states))

    to_log = {
        "mean_reward": np.mean(rewards),
        "min_reward": np.min(rewards),
        "max_reward": np.max(rewards),
        "mean_length": np.mean(lengths),
        "min_length": np.min(lengths),
        "max_length": np.max(lengths),
    }

    if render:
        to_log["renders"] = [wandb.Video(data, format=format) for data, format in renders]

    run.log(to_log)


@cli.command(short_help="Print the default config for a given agent.")
@click.argument("agent", type=click.STRING)
def agent_config(agent):
    """Prints the default config for AGENT."""
    agent_cls = get_agent_cls(agent)
    config_yaml = yaml.dump(agent_cls.default_config().to_dict(), sort_keys=False)
    console.print(config_yaml)


@cli.command(short_help="Lists all available agents.")
def list_agents():
    """Lists all available agents."""
    from dopamax.agents.utils import _registry

    table = Table(title="Available Agents", show_header=True, header_style="bold magenta")
    table.add_column("Agent Name", style="cyan")

    for agent_name in sorted(_registry.keys()):
        table.add_row(agent_name)

    console.print(table)


@cli.command(short_help="Lists all available environments.")
def list_environments():
    """Lists all available environments."""
    from dopamax.environments.utils import _registry

    table = Table(title="Available Environments", show_header=True, header_style="bold magenta")
    table.add_column("Environment Name", style="cyan")
    table.add_column("Type", style="green")

    for env_name in sorted(_registry.keys()):
        table.add_row(env_name, "Native")

    # Also list gymnax environments
    if importlib.util.find_spec("gymnax") is not None:
        table.add_row("", "", style="dim")
        table.add_row("gymnax:CartPole-v1", "Gymnax", style="dim")
        table.add_row("gymnax:MountainCar-v0", "Gymnax", style="dim")
        table.add_row("gymnax:Acrobot-v1", "Gymnax", style="dim")
        table.add_row("(and many more - see gymnax docs)", "Gymnax", style="dim italic")
    else:
        logger.debug("Gymnax not available, skipping gymnax environment listing")

    console.print(table)
