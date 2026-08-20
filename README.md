# Curious-Agent-RL

An experimental reinforcement-learning project comparing two classical
curiosity formulations against a no-curiosity baseline. The project implements:

- **Schmidhuber (1991)** — *Curious Model-Building Control Systems*: intrinsic
  reward from improvement in prediction ability, via a world model M and a
  confidence model C (`r_curiosity = C_before - C_after`).
- **Pathak et al. (2017)** — *Curiosity-driven Exploration by Self-supervised
  Prediction*: the Intrinsic Curiosity Module (ICM), where curiosity is the
  forward-model prediction error in a feature space learned by an inverse
  dynamics model.
- **Vanilla DQN** — an external-reward-only ablation that constructs neither
  curiosity module.

All agents learn inside a configurable grid world containing deterministic,
static, noisy, and changing regions. This makes it possible to study whether
learning progress draws an agent toward useful novelty while allowing it to
lose interest in transitions that are already understood or remain
unpredictable.

## Curiosity mechanisms

### Schmidhuber (1991) — confidence-based learning progress

Each Schmidhuber agent contains three learning components:

- **World model (M):** predicts the next state from the current state and action.
- **Confidence model (C):** predicts the world model's error.
- **Controller (Q):** learns action values from external and intrinsic rewards.

For a state-action pair, the confidence model's prediction is recorded before
and after the online model update:

```text
r_curiosity = C_before - C_after
r_total     = r_external + beta * r_curiosity
```

A decrease in predicted model error produces positive curiosity reward. The
combined reward is then used by either tabular Q-learning or DQN.

### Pathak et al. (2017) — Intrinsic Curiosity Module (ICM)

The ICM agent replaces the world model and confidence model with three networks:

- **Encoder (theta_E):** maps a state to a compact feature vector.
- **Inverse model (theta_I):** predicts the action from consecutive features;
  it shapes the encoder so features only carry action-relevant information.
- **Forward model (theta_F):** predicts the next feature vector; its error is
  the curiosity reward.

```text
r_curiosity = (eta / 2) * ||phi_hat(s') - phi(s')||^2
r_total     = r_external + beta * r_curiosity
```

The ICM is trained online on current transitions (as in the paper) with the
joint objective `(1 - inverse_weight)*L_I + inverse_weight*L_F`, while the
Q-network trains from a replay buffer. The curiosity reward is detached so no
gradient flows from the controller into the forward model, satisfying the
paper's "do not backpropagate the policy gradient loss to the forward model".

### Fair cross-agent curiosity comparison

The two formulations produce curiosity signals on wildly different raw scales:
Schmidhuber's is the change in a sigmoid-bounded confidence prediction
(typically ~0.001 per step), while ICM's is an unbounded feature-space MSE
(typically ~0.25 per step). To make `beta` and the comparison plots meaningful
across agents, both neural agents normalize their curiosity reward with a
running mean of recent magnitudes:

```text
running_mean = 0.99 * running_mean + 0.01 * |r_curiosity_raw|
r_curiosity  = (r_curiosity_raw / (running_mean + eps)) * target_intrinsic_scale
r_total      = r_external + beta * r_curiosity
```

The sign is preserved, so negative (boredom / model-worsening) signals still
register. Set `normalize_curiosity: false` in either config to use the raw
signal. `target_intrinsic_scale` (default 1.0) together with `beta` sets the
intrinsic pressure relative to the sparse +1 external goal.

At each environment step, the pipeline:

1. selects an action with an epsilon-greedy policy;
2. observes the environment transition and external reward;
3. measures and updates the curiosity models (M/C or encoder/inverse/forward);
4. computes curiosity and total reward;
5. updates the tabular Q-function or neural Q-network.

## Implementations

| Component | Tabular agent | Schmidhuber DQN | ICM DQN |
| --- | --- | --- | --- |
| World model | State-action lookup table | Multilayer perceptron | Feature encoder + forward model |
| Confidence model | Predicted-error lookup table | Multilayer perceptron | Inverse model (shapes features) |
| Curiosity reward | `C_before - C_after` | `C_before - C_after` | `(eta/2) * ||phi_hat - phi||^2` |
| Controller | Q-table | Q-network or dueling Q-network | Q-network or dueling Q-network |
| Stabilization | Direct Q-learning updates | Replay buffer and a target-network copy | Replay buffer and a target-network copy |
| Checkpoint | NumPy `.npz` | PyTorch `.pt` | PyTorch `.pt` |

The unified runner executes the tabular experiment first and the curious DQN
experiment second by default. A separate vanilla DQN provides an
external-reward-only ablation, and either neural implementation can be run
independently or as a pair.

## Environment

The default environment is a `10 x 10` grid with four actions: up, down, left,
and right. Its cells are divided among five configurable zone types:

| Zone | Transition behavior | Experimental purpose |
| --- | --- | --- |
| Static | Selected action only; no zone modification | Already-boring behavior |
| Deterministic A | Selected action plus a fixed learnable shift | Primary learnable region |
| Deterministic B | Selected action plus a different fixed shift | Alternative learnable region |
| Noisy | Selected action plus a random local perturbation | Persistently unpredictable region |
| Dynamic | Selected action plus a periodically switching shift | Tests renewed curiosity |

The agent begins at the top-left corner. The default external goal is at the
bottom-right corner and provides a reward of `1.0`.

## Requirements

- Python 3.10 or newer
- NumPy
- PyTorch
- Pygame
- PyYAML
- Matplotlib
- TensorBoard

Development and test tools are provided by the `dev` optional dependency.

## Installation

Clone the repository and create a local virtual environment:

```bash
git clone https://github.com/Nahom32/Curious-Agent-RL
cd Curious-Agent-RL

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate the environment with:

```powershell
venv\Scripts\Activate.ps1
```

The remaining examples use the virtual environment's Python executable
directly, so activating it is optional.

## Quick start

Inspect the resolved experiment plan without starting training:

```bash
venv/bin/python main.py --dry-run
```

Run a small end-to-end experiment:

```bash
venv/bin/python main.py \
  --episodes 5 \
  --max-steps 20 \
  --output-dir runs/smoke
```

Run the complete experiment using the defaults from both configuration files:

```bash
venv/bin/python main.py
```

The default run is substantially longer: 1,000 tabular episodes followed by
2,000 DQN episodes, with at most 200 steps per episode. After both agents
finish, a coupled comparison plot is automatically generated and saved to
`rl_comparison.png`. Pass `--no-plot` to skip this.

## Command-line options

```text
--agent {tabular,dqn,vanilla-dqn,icm,dqn-pair,curious-pair,both,all}
                              Select an experiment; default is both
--tabular-config PATH       Use a custom tabular YAML configuration
--dqn-config PATH           Use a custom DQN YAML configuration
--vanilla-dqn-config PATH   Use a custom vanilla DQN YAML configuration
--icm-config PATH           Use a custom ICM DQN YAML configuration
--episodes N                Override episodes for all selected agents
--max-steps N               Override maximum steps per episode
--seed N                    Override the configured random seed
--output-dir PATH           Place agent artifacts below a common directory
--render                    Render training through Pygame
--dry-run                   Resolve and print the plan without training
--no-plot                   Skip automatic comparison plot after multi-agent runs
```

For the complete built-in help:

```bash
venv/bin/python main.py --help
```

### Run one agent

```bash
venv/bin/python main.py --agent tabular
venv/bin/python main.py --agent dqn
venv/bin/python main.py --agent vanilla-dqn
venv/bin/python main.py --agent icm
venv/bin/python main.py --agent dqn-pair
venv/bin/python main.py --agent curious-pair   # Schmidhuber DQN then ICM DQN
```

### Use custom configurations

```bash
venv/bin/python main.py \
  --tabular-config configs/tabular.yaml \
  --dqn-config configs/dqn.yaml
```

Command-line values for episodes, maximum steps, seed, and output directory
override the corresponding YAML values without changing the files.

### Render training

```bash
venv/bin/python main.py --agent tabular --render
```

Rendering requires a graphical desktop and is considerably slower than
headless training.

## Configuration

The default experiment definitions are:

- [`configs/tabular.yaml`](configs/tabular.yaml)
- [`configs/dqn.yaml`](configs/dqn.yaml)
- [`configs/vanilla_dqn.yaml`](configs/vanilla_dqn.yaml)
- [`configs/icm.yaml`](configs/icm.yaml)

Both files configure:

- grid size, goal, zone proportions, colors, and transition rules;
- exploration rate, decay, curiosity weight, and discount factor;
- episode count, maximum episode length, logging interval, and save interval;
- checkpoint, log, and result paths;
- rendering size and frame rate.

The DQN configuration additionally controls the world-model, confidence-model,
and Q-network architectures, optimizer learning rates, replay buffer, and
target-network updates. The ICM configuration controls the encoder/inverse/
forward-model architectures, the feature dimension, the reward scaling `eta`,
and the `inverse_weight` that balances the inverse and forward losses.

The most important curiosity parameter is:

```yaml
agent:
  beta: 1.0
```

`beta` controls the contribution of intrinsic reward to the controller's total
reward. A value of `0.0` disables curiosity and provides a useful external-only
reward check, but the dedicated vanilla DQN is the preferred ablation because
it does not construct or train the curiosity models at all.

## Outputs

Without `--output-dir`, final checkpoints are written to:

```text
checkpoints/
├── tabular/
│   └── agent_final.npz
├── dqn/
│   └── agent_final.pt
├── icm/
│   └── agent_final.pt
└── vanilla-dqn/
    └── agent_final.pt
```

With `--output-dir runs/experiment-01`, they are written to:

```text
runs/experiment-01/
├── tabular/checkpoints/agent_final.npz
├── dqn/checkpoints/agent_final.pt
├── icm/checkpoints/agent_final.pt
└── vanilla-dqn/checkpoints/agent_final.pt
```

Periodic checkpoints are controlled by `training.save_interval`. Training
progress and summary metrics are emitted to the console.

After a multi-agent pipeline (`both` or `dqn-pair`), the runner automatically
generates a coupled comparison plot saved as `rl_comparison.png`. You can skip
this with `--no-plot`, or generate it later from existing log files:

```bash
venv/bin/python scripts/visualize.py \
  --mode compare \
  --log-files runs/experiment-01/*/logs/training.log
```

## Standalone training scripts

The original per-agent entry points remain available:

```bash
venv/bin/python scripts/train_tabular.py \
  --config configs/tabular.yaml \
  --seed 42

venv/bin/python scripts/train_dqn.py \
  --config configs/dqn.yaml \
  --seed 42

venv/bin/python scripts/train_vanilla_dqn.py \
  --config configs/vanilla_dqn.yaml \
  --seed 42

venv/bin/python scripts/train_dqn_icm.py \
  --config configs/icm.yaml \
  --seed 42
```

Add `--render` to either command to open the Pygame environment.

## Visualization

The visualization script provides live rendering, heatmap analysis, training
curves, and coupled comparison plots:

```bash
# Train and display a live tabular agent
venv/bin/python scripts/visualize.py \
  --config configs/tabular.yaml \
  --mode live \
  --episodes 10

# Load the default final tabular checkpoint and plot C/Q heatmaps
venv/bin/python scripts/visualize.py \
  --config configs/tabular.yaml \
  --mode heatmap

# Plot metrics parsed from a standalone tabular training log
venv/bin/python scripts/visualize.py \
  --mode curves \
  --log-file training_tabular.log

# Plot DQN metrics after training
venv/bin/python scripts/visualize.py \
  --config configs/dqn.yaml \
  --mode curves \
  --log-file training_dqn.log

# Coupled comparison across multiple agents on the same axes
venv/bin/python scripts/visualize.py \
  --mode compare \
  --log-files training_dqn.log training_tabular.log \
  --label "Curious DQN" "Tabular"

# Compare curious DQN against the vanilla ablation
venv/bin/python scripts/visualize.py \
  --mode compare \
  --log-files logs/vanilla_dqn/training.log training_dqn.log \
  --label "Vanilla DQN" "Curious DQN"

# Compare the two curiosity formulations directly
venv/bin/python scripts/visualize.py \
  --mode compare \
  --log-files training_dqn.log training_icm.log \
  --label "Schmidhuber DQN" "ICM DQN (Pathak)"
```

The heatmap command looks for the checkpoint path specified in the selected
configuration. Matplotlib modes create PNG files and open an interactive plot
window. Training curves require the periodic episode records in the log; model
checkpoints do not store historical episode metrics.

The `compare` mode auto-detects the log format (curious-agent or vanilla DQN)
and overlays the shared metrics — **Average Reward**, **Episode Length**, and
**Epsilon** — plus **Curiosity Reward** for curious agents, generating a single
coupled figure saved to `rl_comparison.png`.

## Testing

Run the complete suite with the repository virtual environment:

```bash
venv/bin/pytest
```

Run an individual test module:

```bash
venv/bin/pytest tests/test_agents.py
venv/bin/pytest tests/test_main.py
```

The suite covers the environment, tabular curiosity behavior, neural models,
DQN target-network construction, and unified experiment orchestration.

## Project structure

```text
Curious-Agent-RL/
├── main.py                         # Unified experiment pipeline
├── configs/
│   ├── tabular.yaml                # Tabular experiment configuration
│   ├── dqn.yaml                    # Curious neural experiment
│   └── vanilla_dqn.yaml            # External-reward-only DQN
├── scripts/
│   ├── train_tabular.py            # Standalone tabular trainer
│   ├── train_dqn.py                # Standalone Schmidhuber DQN trainer
│   ├── train_dqn_icm.py            # Standalone ICM DQN trainer
│   ├── train_vanilla_dqn.py        # Vanilla DQN ablation trainer
│   └── visualize.py                # Live and analytical visualization
├── src/curious_agent/
│   ├── agents/
│   │   ├── tabular_curious.py      # Tabular M, C, and Q agent
│   │   ├── dqn_curious.py          # Neural M, C, and Q agent (Schmidhuber)
│   │   ├── dqn_icm.py              # Encoder/inverse/forward ICM + Q agent (Pathak)
│   │   └── dqn.py                  # External-reward-only DQN
│   ├── env/grid_world.py           # Multi-zone grid environment
│   ├── models/
│   │   ├── world_model.py
│   │   ├── confidence_net.py
│   │   ├── icm.py                  # Encoder, inverse model, forward model
│   │   └── q_network.py
│   └── utils/replay_buffer.py
└── tests/                           # Unit and integration tests
```

## Current scope and limitations

This repository is an early experimental implementation rather than a
benchmark-ready RL framework.

- The curiosity signals are online approximations: Schmidhuber's is based on
  the confidence model's change in predicted error; Pathak's is the
  forward-model prediction error in the learned feature space.
- The tabular world model stores one next-state prediction per state-action
  pair, so it cannot represent a full stochastic transition distribution.
- Both DQN variants bootstrap TD targets from a soft-updated target-network
  copy.
- Visualization and checkpoint analysis are currently more complete for the
  tabular agent than for DQN.
- Zone-distribution visualization and tabular model-accuracy reporting still
  contain placeholder logic.
- The unified runner saves checkpoints and log files, and a multi-agent
  comparison plot can be produced from logs; it does not yet aggregate
  repeated seeds or evaluate trained policies.

These constraints make the project most useful for learning, prototyping, and
extending curiosity-driven exploration experiments.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
