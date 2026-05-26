# MARL Lunar Coordination Framework

Multi-agent reinforcement learning for autonomous rover coordination under lunar communication constraints. Three rovers learn to cooperatively cover target positions using a shared PPO policy, tested under degraded comms and mid-mission failures.

---

## What this actually does

The moon has no GPS and no reliable radio. Rovers communicating with each other or a base station will experience real packet loss and signal delay. Most MARL research assumes perfect observation. This doesn't.

Three agents learn to cover three landmark positions using `simple_spread_v3` from PettingZoo MPE. In the live render, the **large blue dots are the landmarks** (fixed target positions) and the **small gray dots are the rovers** — your policy controls the gray ones and tries to get each one near a blue one.

All three agents share one policy since they're homogeneous — training on one agent's experience applies to all of them. A comms wrapper sits between the environment and the agents and randomly drops or delays observations to simulate what a lunar radio link actually looks like. A digital twin logs full trajectories per episode and runs kinematic lookahead to predict collisions before they happen.

The core question: **how much does coordination degrade as comms get worse, and is comms reliability or fleet size the bigger bottleneck when things go wrong?**

---

## Results

| Condition | Mean Total Reward | Completion Rate |
|-----------|:-----------------:|:---------------:|
| No degradation (0%) | -140.95 | 100% |
| 20% packet loss | -135.16 | 100% |
| 40% packet loss | -139.70 | 100% |
| Agent failure (N-1) | -136.99 | 100% |

**Key finding:** the trained policy is surprisingly robust to comms degradation — reward variance across all four conditions is under 6 points and completion rate stays at 100% regardless of packet loss or agent failure. For short-horizon cooperative navigation, the shared policy generalizes well even when agents are acting on stale or missing observations.

Chart saved to `plots/results_comparison.png` after running `plot_results.py`.

---

## Setup

```bash
git clone https://github.com/asingh38-oss/marl-lunar-coordination
cd marl-lunar-coordination

python -m venv venv
source venv/bin/activate   # windows: venv\Scripts\activate

pip install -r requirements.txt
```

Six packages: `pettingzoo[mpe]`, `torch`, `matplotlib`, `pandas`, `numpy`, `pygame`. No Ray, no Stable Baselines, no dependency conflicts.

---

## Quick demo (no training needed)

The trained model ships with the repo. Clone, install, and run immediately:

```bash
# live pygame window -- large blue dots are landmarks, small gray dots are rovers
# press Q to quit
python demo.py --render --episodes 3

# no window, just prints results to terminal
python demo.py --episodes 5

# simulate bad lunar radio -- 40% of observations dropped
python demo.py --loss 0.4 --episodes 5

# one rover dies at step 15, see if the other two recover
python demo.py --fault-agent agent_0 --episodes 5

# save an animated GIF replay to plots/demo_replay.gif
python demo.py --loss 0.2 --episodes 5 --replay

# everything at once
python demo.py --render --loss 0.2 --fault-agent agent_0 --fault-step 15 --episodes 3 --replay
```

### Demo flags

| Flag | Default | What it does |
|------|---------|-------------|
| `--render` | off | open live pygame window, press Q to quit |
| `--loss` | `0.0` | packet loss rate (0.0 to 1.0) |
| `--latency` | `0` | observation delay in steps |
| `--episodes` | `5` | number of episodes to run |
| `--fault-agent` | `None` | which rover to kill mid-episode (e.g. `agent_0`) |
| `--fault-step` | `15` | step the fault happens |
| `--replay` | off | save animated GIF to `plots/demo_replay.gif` |
| `--checkpoint` | `checkpoints/policy_final.pt` | path to model weights |

---

## Training from scratch

If you want to retrain instead of using the included model:

```bash
python train.py
```

Trains for 3000 episodes using PPO with a shared actor-critic network. Updates every 10 episodes. Takes around 15-20 minutes on CPU. Saves checkpoints every 500 episodes to `checkpoints/` and a final `checkpoints/policy_final.pt` at the end. Training reward and loss log to `logs/training_log.csv`.

Watch the reward trend while it trains (open a second terminal):

```bash
python -c "
import pandas as pd
df = pd.read_csv('logs/training_log.csv')
print(df.dropna().tail(20)[['episode','reward']].to_string(index=False))
"
```

Reward starts very negative (random behavior) and slowly climbs. If it's improving by episode 500, it's working.

---

## Full evaluation (reproducing the results table)

```bash
python eval.py --loss 0.0  --episodes 30 --tag baseline
python eval.py --loss 0.2  --episodes 30 --tag loss_20
python eval.py --loss 0.4  --episodes 30 --tag loss_40
python eval.py --loss 0.0  --fault-agent agent_0 --fault-step 15 --episodes 30 --tag fault

python plot_results.py
```

Each run saves a CSV to `logs/eval_{tag}.csv` and a twin trajectory log to `logs/twin_{tag}.json`.

Replay any episode as a GIF:

```bash
python twin.py 0 logs/twin_baseline.json
python twin.py 0 logs/twin_baseline.json plots/replay_ep0.gif
```

---

## File structure

```
marl-lunar-coordination/
├── demo.py             -- main entry point, load model and run any scenario
├── train.py            -- PPO training loop + ActorCritic network definition
├── eval.py             -- full evaluation with logging (for reproducing results)
├── lunar_comms.py      -- comms degradation wrapper (packet loss + latency)
├── twin.py             -- digital twin: trajectory logging, collision prediction, replay
├── plot_results.py     -- generates comparison charts from eval CSVs
├── requirements.txt
├── .gitignore
├── checkpoints/
│   └── policy_final.pt -- trained model weights (included, no training needed)
├── logs/               -- CSVs and twin JSONs generated during eval
└── plots/              -- output charts and replay GIFs
```

---

## How the PPO works

Single actor-critic network shared across all three agents. Each update cycle (every 10 episodes):

1. Collect rollouts -- each agent runs forward passes through the shared policy to get actions, log-probs, and value estimates. Per-agent returns are computed separately before being pooled into one shared batch.
2. Normalize advantages and returns -- keeps the value loss from exploding
3. PPO update -- 4 gradient steps with clipped surrogate objective (eps=0.2), value coefficient 0.5, entropy bonus 0.01 to keep exploration alive
4. Gradient clipping at 0.5

All three agents contribute to the same update batch, so each cycle generates 3x the data compared to training separate policies. This is parameter sharing -- the standard approach for homogeneous agent fleets.

## How the comms wrapper works

`LunarCommsWrapper` wraps the PettingZoo env. On each step:

- New observations go into a per-agent circular buffer (size = `latency_steps + 1`)
- If `latency_steps > 0`, the agent receives the observation from that many steps ago instead of the current one
- With probability `packet_loss`, the delivered observation is swapped out for the agent's last known observation -- not zeros, because a real rover would use stale data, not nothing

An agent with 40% packet loss and 2-step latency is acting on information that's at minimum 2 steps old, and 40% of the time even older. That's close to what you'd actually see on a low-bandwidth lunar relay link.

## How the digital twin works

Every step during eval, the twin logs position, velocity, action, and reward for each agent. It also runs kinematic lookahead -- projects each agent 3 steps forward using current velocity and flags any pair predicted to get within the collision threshold before it actually happens. Trajectories save to JSON and replay as GIFs.

---

## Acknowledgments

Built with assistance from Claude (Anthropic) for code structure, debugging, and project framing. All experiments, results, and analysis are my own.