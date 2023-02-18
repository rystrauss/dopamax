[1]: https://github.com/google/jax

[2]: https://arxiv.org/abs/2104.06272

# Dopamax

<p>
       <a href="https://pypi.python.org/pypi/dopamax">
        <img src="https://img.shields.io/pypi/pyversions/dopamax.svg?style=flat-square" /></a>
       <a href= "https://badge.fury.io/py/dopamax">
        <img src="https://badge.fury.io/py/dopamax.svg" /></a>
       <a href= "https://github.com/rystrau/dopamax/blob/master/LICENSE.md">
        <img src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
       <a href= "https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" /></a>
</p>

Dopamax is a library containing pure [JAX][1] implementations of common reinforcement learning algorithms. _Everything_
is implemented in JAX, including the environments. This allows for extremely fast training and evaluation of agents,
because the entire loop of environment simulation, agent interaction, and policy updates can be compiled as a single
XLA program and executed on CPUs, GPUs, or TPUs. More specifically, rhe implementations in Dopamax follow the
Anakin Podracer architecture -- see [this paper][2] for more details.

## Supported Algorithms

- [Proximal Policy Optimization (PPO)](dopamax/agents/ppo.py)
- [Deep Q-Network (DQN)](dopamax/agents/dqn.py)

## Installation

Dopamax can be installed with:

```bash
pip install git+https://github.com/rystrauss/dopamax.git
```

This will install the `dopamax` Python package, as well as a command-line interface (CLI) for training and evaluation.

## Usage

After installation, the Dopamax CLI can be used to train and evaluate agents:

```bash
dopamax --help
```

Dopamax uses [Weights and Biases (W&B)](https://wandb.ai/site) for logging and artifact management. Before using the CLI
for training and evaluation, you must first make sure you have a W&B account (it's free) and have authenticated
with `wandb login`.

### Training

Agent's can be trained using the `dopamax train` command, to which you must provide a configuration file. The
configuration file is a YAML file that specifies the agent, environment, and training hyperparameters. You can find
examples in the [configs](configs) directory. For example, to train a PPO agent on the CartPole environment, you would
run:

```bash
dopamax train --config examples/ppo-cartpole/config.yaml
```

Note that all of the example config files have a random seed specified, so you will get the same result every time you
run the command. The seeds provided in the examples are known to result in a successful run (with the given
hyperparameters). To get different results on each run, you can remove the seed from the config file.

### Evaluation

Once you have trained some agents, you can evaluate them using the `dopamax evaluate` command. This will allow you to
specify a W&B agent artifact that you'd like to evaluate (these artifacts are produced by the training runs and
contain the agent hyperparameters and weights from the end of training). For example, to evaluate a PPO agent trained
on CartPole, you might use a command like:

```bash
dopamax evaluate --agent_artifact CartPole-PPO-agent:v0 --num_episodes 100
```

where `--num_episodes 100` signals that you would like to rollout the agent's policy for 100 episodes. The minimum,
mean, and maximum episode reward will be logged back to W&B. If you would additionally like to render the episodes and
have then logged back to W&B, you can provide the `--render` flag. But note that this will usually significantly slow
down the evaluation process since environment rendering is not a pure JAX function and requires callbacks to the host.
You should usually only use the `--render` flag with a small number of episodes.