import os
import pickle
import random

import click
import jax.random
import wandb
import yaml
from ml_collections import ConfigDict

from dopamax.agents.utils import get_agent_cls
from dopamax.callbacks import WandbCallback
from dopamax.environments import make_env


@click.command()
@click.option(
    "--config",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    required=True,
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


if __name__ == "__main__":
    main()
