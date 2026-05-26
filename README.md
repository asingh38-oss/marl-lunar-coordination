# MARL Lunar Coordination Framework

Multi-agent reinforcement learning for autonomous rover coordination under lunar communication constraints. Three agents learn to cooperatively cover landmark targets using a shared PPO policy, evaluated under degraded comms (packet loss, observation latency) and mid-mission agent failure.

---

## What this actually does

The moon has no GPS and no reliable radio infrastructure. Rovers communicating with each other or a base station will experience real packet loss and signal delay. Most MARL research assumes perfect observation. This doesn't.

Three agents learn to cover three landmark positions using `simple_spread_v3` from PettingZoo MPE. All agents share one policy since they're homogeneous -- training on one agent's experience applies to all of them. A comms wrapper sits between the environment and the agents and randomly drops or delays observations. A digital twin logs full trajectories per episode and runs kinematic lookahead to predict collisions before they happen.

The core question: **how much does coordination degrade as comms get worse, and is comms reliability or fleet size the bigger bottleneck when things go wrong?**

---

## Results

| Condition | Mean Total Reward | Completion Rate |
|-----------|:-----------------:|:---------------:|
| No degradation (0%) | -140.95 | 100% |
| 20% packet loss | -135.16 | 100% |
| 40% packet loss | -139.70 | 100% |
| Agent failure (N-1) | -136.99 | 100% |

Chart saved to `plots/results_comparison.png` after running `plot_results.py`.

**Key finding:** the trained policy is surprisingly robust to comms degradation — reward variance across all conditions is under 6 points. All episodes completed regardless of packet loss or agent failure. This suggests that for short-horizon cooperative navigation tasks, the shared policy generalizes well even when agents operate on stale or missing observations. The more interesting degradation likely emerges at higher loss rates (>60%) or longer latency windows.

---

## Quick start (no training needed)

The trained model is included in the repo. Clone and run immediately:

```bash
git clone https://github.com/asingh38-oss/marl-lunar-coordination
cd marl-lunar-coordination
python -m venv venv
source venv/bin/activate   # windows: venv\Scripts\activate
pip install -r requirements.txt
python demo.py --episodes 5
```

---

## Demo usage

`demo.py` is the main entry point. Load the trained model and run any scenario:

```bash
# baseline -- no degradation
python demo.py --episodes 5

# 20% packet loss
python demo.py --loss 0.2 --episodes 5

# 40% packet loss with replay GIF
python demo.py --loss 0.4 --episodes 5 --replay

# kill agent_0 at step 15
python demo.py --fault-agent agent_0 --episodes 5

# combine -- packet loss AND agent failure AND replay
python demo.py --loss 0.2 --fault-agent agent_0 --fault-step 15 --episodes 5 --replay
```

### Demo flags

| Flag | Default | What it does |
|------|---------|-------------|
| `--loss` | `0.0` | packet loss rate (0.0 to 1.0) |
| `--latency` | `0` | observation delay in steps |
| `--episodes` | `5` | number of episodes to run |
| `--fault-agent` | `None` | agent to kill mid-episode (e.g. `agent_0`) |
| `--fault-step` | `15` | step the fault happens |
| `--replay` | off | save animated GIF replay to `plots/demo_replay.gif` |
| `--checkpoint` | `checkpoints/policy_final.pt` | path to model weights |

---

## Training from scratch

If you want to retrain instead of using the included model:

```bash
python train.py
```

Trains for 2000 episodes using PPO with a shared actor-critic network. Takes ~15 minutes on CPU. Saves checkpoints every 100 episodes to `checkpoints/` and logs reward/loss to `logs/training_log.csv`.

Watch progress while it trains (separate terminal):

```bash
python -c "
import pandas as pd
df = pd.read_csv('logs/training_log.csv')
print(df.tail(20)[['episode','reward']].to_string(index=False))
"
```

---

## Full evaluation (reproducing the results table)

```bash
python eval.py --loss 0.0 --episodes 30 --tag baseline
python eval.py --loss 0.2 --episodes 30 --tag loss_20
python eval.py --loss 0.4 --episodes 30 --tag loss_40
python eval.py --loss 0.0 --fault-agent agent_0 --fault-step 15 --episodes 30 --tag fault
python plot_results.py
```

---

## File structure
marl-lunar-coordination/
├── demo.py             -- main entry point, load model and run any scenario
├── train.py            -- PPO training loop + ActorCritic network definition
├── eval.py             -- full evaluation with logging (for reproducing results)
├── lunar_comms.py      -- comms degradation wrapper (packet loss + latency)
├── twin.py             -- digital twin: trajectory logging, collision prediction, replay
├── plot_results.py     -- generates comparison charts from eval CSVs
├── requirements.txt
├── checkpoints/
│   └── policy_final.pt -- trained model weights (included, no training needed)
├── logs/               -- CSVs and twin JSONs generated during eval
└── plots/              -- output charts and replay GIFs

---

## How the PPO works

Single actor-critic network shared across all three agents. Each episode:

1. Collect rollout -- each agent runs a forward pass to get an action, log-prob, and value estimate
2. Compute discounted returns (gamma=0.99) and normalize advantages
3. PPO update -- 4 gradient steps, clipped surrogate objective (eps=0.2), value coefficient 0.5, entropy bonus 0.01
4. Gradient clipping at 0.5 to prevent exploding gradients

All three agents contribute to the same update batch, so each episode generates 3x the data of training separate policies. This is parameter sharing -- standard for homogeneous agent fleets.

## How the comms wrapper works

`LunarCommsWrapper` wraps the PettingZoo env. On each step:

- New observations go into a per-agent circular buffer (size = `latency_steps + 1`)
- If `latency_steps > 0`, the agent gets the observation from that many steps ago
- With probability `packet_loss`, the delivered observation is replaced with the agent's last known observation -- not zeros, because a real rover would use stale data, not nothing

## How the digital twin works

Every step, the twin logs position, velocity, action, and reward for each agent. It also runs kinematic lookahead -- projects each agent forward 3 steps using current velocity and flags any pair predicted to collide before it happens. Trajectories save to JSON and replay as GIFs.

---

## Acknowledgments

Built with assistance from Claude (Anthropic) for code structure, debugging, and project framing. All experiments, results, and analysis are my own.
