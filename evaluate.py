import os
import pickle
import random

import click
import einops
import haiku as hk
import jax
import numpy as np
import wandb
from dm_env import StepType
from ml_collections import ConfigDict
from tqdm import tqdm

from dopamax.agents.utils import get_agent_cls
from dopamax.environments import make_env
from dopamax.rollouts import rollout_episode, SampleBatch


@click.command()
@click.option("--agent_artifact", type=click.STRING, required=True, help="Path to the agent's config YAML file.")
@click.option("--num_episodes", type=click.INT, required=True, help="The number of episodes to evaluate.")
@click.option("--render", type=click.BOOL, default=False, help="Whether to render the episodes.")
def main(agent_artifact, num_episodes, render):
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


if __name__ == "__main__":
    main()
