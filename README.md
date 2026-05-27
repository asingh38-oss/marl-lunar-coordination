# marl-lunar-coordination

PPO-trained agents that coordinate to cover target positions, tested under degraded comms and rover failures. Built as research for NASA Ames internship applications focused on autonomous multi-vehicle coordination.

---

## What it does

Three rovers learn to spread out and cover three target positions (landmarks) using multi-agent PPO with a shared policy. The interesting part is the communication wrapper — you can simulate packet loss, observation delay, and mid-mission rover failure to see how coordination holds up.

In the live render: **big blue dots = landmarks (stationary targets)**, **small gray dots = the rovers you're controlling**.

The main question this tries to answer: is comms reliability or fleet size the bigger bottleneck when things go wrong on a lunar surface mission?

---

## Results

| Condition | Mean Reward | Completion |
|-----------|:-----------:|:----------:|
| No degradation | -140.95 | 100% |
| 20% packet loss | -135.16 | 100% |
| 40% packet loss | -139.70 | 100% |
| Agent failure (N-1) | -136.99 | 100% |

The policy ended up being pretty robust — less than 6 points of reward variance across all conditions, 100% completion rate even with a dead rover. Probably because the task horizon is short enough that stale observations don't hurt much. Higher loss rates (>60%) or longer latency windows would likely show more degradation.

---

## Setup

```bash
git clone https://github.com/asingh38-oss/marl-lunar-coordination
cd marl-lunar-coordination

python -m venv venv
source venv/bin/activate   # windows: venv\Scripts\activate

pip install -r requirements.txt
```

Requirements: `pettingzoo[mpe]`, `torch`, `matplotlib`, `pandas`, `numpy`, `pygame`

---

## Running it

Trained model is already in the repo, no training needed.

```bash
# live pygame window -- press Q to quit
python demo.py --render --episodes 3

# just print results, no window
python demo.py --episodes 5

# simulate 40% packet loss
python demo.py --loss 0.4 --episodes 5

# kill rover 0 at step 15
python demo.py --fault-agent agent_0 --episodes 5

# save a gif replay
python demo.py --loss 0.2 --episodes 5 --replay

# worst case
python demo.py --render --loss 0.2 --fault-agent agent_0 --fault-step 15 --episodes 3
```

Flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--render` | off | open live pygame window |
| `--loss` | 0.0 | packet loss rate (0.0 to 1.0) |
| `--latency` | 0 | observation delay in steps |
| `--episodes` | 5 | number of episodes |
| `--fault-agent` | None | rover to kill mid-episode (e.g. agent_0) |
| `--fault-step` | 15 | step the fault happens |
| `--replay` | off | save gif to plots/demo_replay.gif |
| `--checkpoint` | checkpoints/policy_final.pt | model weights path |

---

## MARL vs Stigmergy

Ran a head-to-head between the trained PPO policy and a rule-based pheromone agent (stigmergy) across three conditions. Stigmergy works by having each rover deposit a scent trail on cells it visits — trails decay over time and rovers move toward the least-visited areas. No neural net, no training, just local rules.

| Condition | MARL (PPO) | Stigmergy (rule-based) |
|-----------|:----------:|:----------------------:|
| No degradation | -137.77 | -241.47 |
| 40% packet loss | -141.41 | -244.95 |
| Agent failure (N-1) | -136.46 | -252.11 |

Key findings:
- MARL outperforms stigmergy by ~75% across all conditions
- Both degrade almost identically under 40% packet loss (~2% drop each) — the task is short enough that stale observations don't hurt either approach much
- Agent failure hits stigmergy harder (-252 vs -136) because stigmergy needs all three rovers actively covering separate areas — lose one and coverage drops noticeably
- The comms robustness advantage for stigmergy (expected, since it doesn't use radio at all) didn't show up at this loss rate — would likely emerge at >60% loss or with longer latency windows

```bash
python stigmergy.py
# outputs logs/stigmergy_benchmark.csv and plots/stigmergy_vs_marl.png
```

---

## Training from scratch

```bash
python train.py
```

3000 episodes, updates every 10. Takes around 15-20 minutes on CPU. Logs to `logs/training_log.csv`, saves checkpoints every 500 episodes.

Check progress mid-training:
```bash
python -c "import pandas as pd; df = pd.read_csv('logs/training_log.csv'); print(df.dropna().tail(20)[['episode','reward']].to_string(index=False))"
```

---

## Reproducing the full benchmark

```bash
python eval.py --loss 0.0  --episodes 30 --tag baseline
python eval.py --loss 0.2  --episodes 30 --tag loss_20
python eval.py --loss 0.4  --episodes 30 --tag loss_40
python eval.py --loss 0.0  --fault-agent agent_0 --fault-step 15 --episodes 30 --tag fault
python plot_results.py
```

Replay any episode as a GIF:
```bash
python twin.py 0 logs/twin_baseline.json
python twin.py 0 logs/twin_baseline.json plots/replay_ep0.gif
```

---

## Files

```
marl-lunar-coordination/
├── demo.py             # main entry point
├── train.py            # PPO training + ActorCritic network
├── eval.py             # evaluation with logging
├── lunar_comms.py      # packet loss + latency wrapper
├── twin.py             # digital twin: logging, collision prediction, replay
├── plot_results.py     # bar charts from eval CSVs
├── stigmergy.py        # rule-based pheromone agent (MARL vs stigmergy benchmark)
├── requirements.txt
├── .gitignore
├── checkpoints/
│   └── policy_final.pt # trained weights, no training needed
├── logs/               # generated during eval, gitignored
└── plots/              # generated during eval, gitignored
```

---

## How the PPO works

All three agents share one network (parameter sharing — they're identical so there's no reason to train separate policies). Each update cycle collects 10 episodes of experience, computes per-agent discounted returns separately, then does 4 gradient steps with the PPO clip objective. Normalizing both advantages and returns was the key fix that stopped the value loss from exploding early in training.

## How the comms wrapper works

Wraps the PettingZoo env and intercepts observations before they reach the agents. On each step: the new obs goes into a per-agent buffer, the agent gets the delayed version if latency > 0, and with probability `packet_loss` the delivered obs gets swapped for the agent's last known observation — not zeros, because a real rover would use stale data, not nothing.

## How the digital twin works

Logs position, velocity, action, and reward for every agent every step. Also runs kinematic lookahead each step — projects each agent 3 steps forward using current velocity and flags any pair predicted to collide before it actually happens. Saves to JSON, replays as GIF.

---

Built with help from Claude (Anthropic) for code structure and debugging. Experiments and analysis are my own.