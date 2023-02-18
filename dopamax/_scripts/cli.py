import os
import pickle
import random

import click
import einops
import haiku as hk
import jax
import jax.random
import numpy as np
import wandb
import yaml
from dm_env import StepType
from ml_collections import ConfigDict
from tqdm import tqdm

from dopamax.agents.utils import get_agent_cls
from dopamax.callbacks import WandbCallback
from dopamax.environments import make_env
from dopamax.rollouts import rollout_episode, SampleBatch


@click.group()
def cli():
    """Dopamax provides a collection of reinforcement learning agents and environments implemented in pure JAX."""
    pass


@cli.command(short_help="Train an agent.")
@click.option(
    "--config",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    required=True,
    help="Path to the training config YAML file.",
)
@click.option("--offline", is_flag=True, help="Run in offline mode, without logging to W&B.")
def train(config, offline):
    """Trains an agent using the provided config file, and logs results to W&B."""
    with open(config, "r") as f:
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

    params = agent.train(
        key,
        config.train_iterations,
        callback_freq=config.get("callback_freq", 100),
        callbacks=[WandbCallback(run)],
    )

    if not offline:
        params_path = os.path.join(run.dir, "params.pkl")

        with open(params_path, "wb") as f:
            pickle.dump(params, f)

        params_artifact = wandb.Artifact(config.env_name + "-" + config.agent_name + "-agent", type="agent")
        params_artifact.add_file(params_path)
        run.log_artifact(params_artifact)


@cli.command(short_help="Evaluate an agent.")
@click.option("--agent_artifact", type=click.STRING, required=True, help="Path to the agent's config YAML file.")
@click.option("--num_episodes", type=click.INT, required=True, help="The number of episodes to evaluate.")
@click.option(
    "--render",
    type=click.BOOL,
    default=False,
    help="Whether to render the episodes. Note that this will usually significantly slow down the evaluation process, "
    "since environment rendering is not a pure JAX function and requires callbacks to the host.",
)
def evaluate(agent_artifact, num_episodes, render):
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

    rollout_fn = jax.jit(rollout_episode, static_argnums=(0, 1, 4))

    def policy_fn(*args):
        return agent.compute_action(*args, deterministic=True), {}

    prng = hk.PRNGSequence(random.randint(0, 2**32 - 1))

    rewards, lengths, renders = [], [], []

    for _ in tqdm(range(num_episodes), unit="episodes"):
        rollout_data = rollout_fn(env, policy_fn, params, prng.next(), render)
        rewards.append(rollout_data[SampleBatch.EPISODE_REWARD][-1])
        lengths.append(rollout_data[SampleBatch.EPISODE_LENGTH][-1])

        if render:
            last_index = np.argwhere(rollout_data[SampleBatch.STEP_TYPE] == StepType.LAST)[0][0]
            renders.append(rollout_data[SampleBatch.RENDER][: last_index + 1])

    to_log = {
        "mean_reward": np.mean(rewards),
        "min_reward": np.min(rewards),
        "max_reward": np.max(rewards),
        "mean_length": np.mean(lengths),
        "min_length": np.min(lengths),
        "max_length": np.max(lengths),
    }

    if render:
        to_log["renders"] = [
            wandb.Video(einops.rearrange(np.array(data), "t h w c -> t c h w"), fps=env.fps) for data in renders
        ]

    run.log(to_log)


@cli.command(short_help="Print the default config for a given agent.")
@click.argument("agent", type=click.STRING)
def agent_config(agent):
    """Prints the default config for AGENT."""
    agent_cls = get_agent_cls(agent)
    print(yaml.dump(agent_cls.default_config().to_dict(), sort_keys=False))


@cli.command(short_help="Lists all available agents.")
def list_agents():
    """Lists all available agents."""
    from dopamax.agents.utils import _registry

    print("Available agents:")
    for agent_name in sorted(_registry.keys()):
        print("  -", agent_name)


@cli.command(short_help="Lists all available environments.")
def list_environments():
    """Lists all available environments."""
    from dopamax.environments.utils import _registry

    print("Available environments:")
    for agent_name in sorted(_registry.keys()):
        print("  -", agent_name)
