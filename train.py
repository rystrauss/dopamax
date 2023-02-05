import os
import pickle
import random

import click
import jax.random
import yaml
from ml_collections import ConfigDict

import wandb
from relax.agents.utils import get_agent_cls
from relax.callbacks import WandbCallback
from relax.environments import make_env


@click.command()
@click.option(
    "--config",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    help="Path to the training config YAML file.",
)
@click.option("--offline", is_flag=True, help="Run in offline mode.")
def main(config, offline):
    with open(config, "r") as f:
        config = ConfigDict(yaml.safe_load(f))

    if "seed" not in config:
        config.seed = random.randint(0, 2**32 - 1)

    env = make_env(config.env_name)
    agent_cls = get_agent_cls(config.agent_name)
    agent = agent_cls(env, config.agent_config)

    run = wandb.init(
        project="relax",
        job_type="train",
        config=config.to_dict(),
        mode="disabled" if offline else "online",
    )

    key = jax.random.PRNGKey(config.seed)

    train_state = agent.train(
        key,
        config.train_iterations,
        callbacks=[WandbCallback(run)],
    )

    train_state_path = os.path.join(wandb.run.dir, "train_state.pkl")

    with open(train_state_path, "wb") as f:
        pickle.dump(train_state, f)

    train_state_artifact = wandb.Artifact("train_state", type="train_state")
    train_state_artifact.add_file(train_state_path)
    run.log_artifact(train_state_artifact)


if __name__ == "__main__":
    main()
