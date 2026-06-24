# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Dopamax is a library of **pure JAX** reinforcement learning algorithms — including the environments. The entire loop (env simulation, agent interaction, policy updates) compiles to a single XLA program, so it must stay JAX-traceable end to end. Implementations follow the [Anakin Podracer architecture](https://arxiv.org/abs/2104.06272): vmap a train step across a batch within each device, then pmap across devices.

## Commands

This project uses `uv` (with `ruff` for lint/format). Python 3.11 / 3.12.

```bash
uv sync --all-extras --dev          # install deps + dev group
uv run pytest tests                 # run all tests
uv run pytest tests/test_agents.py  # single file
uv run pytest tests/test_agents.py::test_name   # single test
uv run ruff format .                # format (line-length 120, double quotes)
uv run ruff check .                 # lint

# CLI (entry point: dopamax = dopamax._scripts.cli:cli)
dopamax list-agents
dopamax list-environments
dopamax agent-config PPO            # print an agent's default config as YAML
dopamax train --config examples/ppo-cartpole/config.yaml [--offline]
dopamax evaluate --agent_artifact CartPole-PPO-agent:v0 --num_episodes 100 [--render]
```

CI (`.github/workflows/python-package.yml`) runs pytest plus short `--offline` training smoke tests for PPO/DQN CartPole and AlphaZero TicTacToe. Run those `train --offline` commands locally to reproduce CI failures. Training/eval log to Weights & Biases — `wandb login` first, or pass `--offline` / `mode="disabled"`.

## Architecture

### The three registries

Agents, environments, and networks each live behind a string-keyed registry populated by a `@register("Name")` decorator and read via a `make`/`get` lookup. **Registration only happens on import**, so a new class must be imported in the relevant `__init__.py` or it won't be found:

- **Agents** — `agents/utils.py` (`register`, `get_agent_cls`); imported in `agents/__init__.py`.
- **Environments** — `environments/utils.py` (`register`, `make_env`); imported in `environments/__init__.py`. `make_env` also special-cases the `gymnax:` prefix, wrapping gymnax envs in `GymnaxEnvironment`.
- **Networks** — `networks.py` (`register`, `get_network_build_fn`); `"mlp"` and `"cnn"` are built in.

### Agent class hierarchy

`Agent` (`agents/base.py`, abstract) → `AnakinAgent` (`agents/anakin/base.py`) → concrete agents (`ppo.py`, `dqn.py`, `ddpg.py` which defines both DDPG and TD3, `sac.py`, `alphazero.py`).

- `AnakinAgent.train()` is the shared driver. It conditionally `jax.vmap`s the train step over `num_envs_per_device` (axis `batch_axis`) and `jax.pmap`s over `num_devices` (axis `device_axis`), wraps `callback_freq` train steps in a `jax.lax.scan` inside one `@jax.jit`, then runs callbacks between scan blocks on host. **`num_iterations` must be a multiple of `callback_freq`.**
- Concrete agents implement `default_config()`, `compute_action()`, `initial_train_state()`, and `train_step()`. Each defines a module-level `_DEFAULT_<AGENT>_CONFIG` `ConfigDict` and merges it onto the parent's via the `super(Cls, Cls).default_config()` pattern.
- **`_maybe_all_reduce(fn, x)`** (e.g. `pmean` on grads, `psum` on counts) is a no-op with a single env/device and the correct collective otherwise — always route cross-replica reductions through it rather than calling `jax.lax.p*` directly, so single-device runs stay correct.

### Config flow

Training config is a YAML file with top-level keys `env_name`, `agent_name`, `train_iterations`, optional `seed`, optional `callback_freq`, and a nested `agent_config:`. `Agent.__init__` merges `agent_config` onto `default_config()` then **locks** the `ConfigDict` (no new keys after construction). Examples live in `examples/<agent>-<env>/config.yaml`.

### Train state

`TrainState` (base) carries `key`, `train_step`, `total_timesteps`, `total_episodes`, and an `episode_buffer_state` (a flashbax ring buffer of the last 128 episodes' length/reward, used purely for logging). `AnakinTrainState` adds `params`, `opt_state`, `time_step`, `env_state`. Off-policy agents (DQN/DDPG/SAC) use `AnakinTrainStateWithReplayBuffer`, which adds `buffer_state`. All are frozen chex dataclasses (immutable JAX pytrees) — produce new state via `.update(...)`, never mutate.

A common JAX-traceable idiom for conditional logic (e.g. DQN's "don't learn until `learning_starts`"): compute both the real `next_train_state` and a `warmup_train_state`, then `jax.tree.map` a `jax.lax.select` between them on a predicate. Don't use Python `if` on traced values.

### Environments

`Environment` (`environments/environment.py`, frozen dataclass, abstract) exposes `name`, `max_episode_length`, `observation_space`, `action_space`, `reset(key) -> (TimeStep, EnvState)`, `step(key, state, action) -> (TimeStep, EnvState)`. `TimeStep` mirrors `dm_env` (`restart`/`transition`/`termination`/`truncation` constructors, with `StepType`). `EnvState` subclasses track `episode_reward`/`episode_length` plus env-specific fields.

Three families: **brax** (continuous control, `environments/brax/`), **pgx** (board games, `environments/pgx/` — `PGXEnvironment` wraps `pgx.Env`; observations are a `Dict` space with `observation` + `invalid_actions` mask; used by AlphaZero), and **gymnax** (classic control, via the `gymnax:` prefix).

### Rollouts and data

`rollouts.py` provides `rollout_episode` (full episode via `jax.lax.scan` to `max_episode_length`) and `rollout_truncated` (fixed N steps, auto-resetting across episode boundaries — the workhorse for on-policy collection), plus `create_minibatches`. All trajectory data is a `SampleBatch` (a `dict` subclass with string-constant keys like `OBSERVATION`, `ACTION`, `ADVANTAGE`). Agents add computed keys (e.g. PPO writes `RETURN`/`ADVANTAGE` via `rlax`) onto the dict.

### Spaces, networks, distributions

- `spaces.py`: `Discrete`, `Box`, `Dict` — each with `sample(key)` and `contains(item)`.
- `networks.py`: build functions return Haiku modules. `get_actor_critic_model_fn` (PPO/AlphaZero, `value_network` ∈ `"copy"`/`"shared"`), `get_deterministic_actor_model_fn` (DDPG/TD3), `get_discrete_q_network_model_fn` (DQN, with `dueling`/`use_twin`), `get_continuous_q_network_model_fn` (SAC/TD3). Models are `hk.transform`ed in the agent constructor.
- Policies are `distrax` distributions. `distributions.py` adds a `Transformed` wrapper (used for tanh-squashed Gaussians in continuous control).
- `utils/jax_utils.py::expand_apply` is a distrax-compatible variant of `hk.expand_apply` for applying batched models to a single timestep.

## Conventions and gotchas

- **Everything is traced.** No Python control flow on array values inside `train_step`/`step` — use `jax.lax.cond`/`select`/`scan`. No `.item()` or host callbacks except in callbacks (which run on host between jitted blocks) and rendering.
- Multi-device/batched reductions must go through `_maybe_all_reduce` to remain valid in the single-replica case.
- Adding to a registry without importing the module in `__init__.py` makes it invisible to the CLI and `make_env`/`get_agent_cls`.
- Trained agents are saved as a W&B artifact (`<env>-<agent>-agent`) containing a pickled `params.pkl`; `evaluate` reconstructs the agent from the originating run's logged config.
