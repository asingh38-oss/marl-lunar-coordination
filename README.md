# marl-lunar-coordination

PPO-trained agents that coordinate to cover target positions, tested under degraded comms and rover failures. Built as research for NASA Ames internship applications focused on autonomous multi-vehicle coordination.

---

## what it does

Three rovers learn to spread out and cover three target positions (landmarks) using multi-agent PPO with a shared policy. The interesting part is the communication wrapper -- you can simulate packet loss, observation delay, and mid-mission rover failure to see how coordination holds up.

In the live render: **big blue dots = landmarks (stationary targets)**, **small gray dots = the rovers you're controlling**.

The main question this tries to answer: is comms reliability or fleet size the bigger bottleneck when things go wrong on a lunar surface mission?

---

## results

| condition | mean reward | completion |
|-----------|:-----------:|:----------:|
| no degradation | -140.95 | 100% |
| 20% packet loss | -135.16 | 100% |
| 40% packet loss | -139.70 | 100% |
| agent failure (N-1) | -136.99 | 100% |

the policy ended up being pretty robust -- less than 6 points of reward variance across all conditions, 100% completion rate even with a dead rover. probably because the task horizon is short enough that stale observations don't matter much. higher loss rates (>60%) or longer latency windows would likely show more degradation.

---

## setup

```bash
git clone https://github.com/asingh38-oss/marl-lunar-coordination
cd marl-lunar-coordination

python -m venv venv
source venv/bin/activate   # windows: venv\Scripts\activate

pip install -r requirements.txt
```

requirements: `pettingzoo[mpe]`, `torch`, `matplotlib`, `pandas`, `numpy`, `pygame`

---

## running it

trained model is already in the repo, no training needed

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

# worst case scenario
python demo.py --render --loss 0.2 --fault-agent agent_0 --fault-step 15 --episodes 3
```

flags:

| flag | default | description |
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

## training from scratch

```bash
python train.py
```

3000 episodes, updates every 10. takes ~15-20 min on cpu. logs to `logs/training_log.csv`, saves checkpoints every 500 episodes.

check progress mid-training:
```bash
python -c "import pandas as pd; df = pd.read_csv('logs/training_log.csv'); print(df.dropna().tail(20)[['episode','reward']].to_string(index=False))"
```

---

## reproducing the benchmark

```bash
python eval.py --loss 0.0  --episodes 30 --tag baseline
python eval.py --loss 0.2  --episodes 30 --tag loss_20
python eval.py --loss 0.4  --episodes 30 --tag loss_40
python eval.py --loss 0.0  --fault-agent agent_0 --fault-step 15 --episodes 30 --tag fault
python plot_results.py
```

replay any episode:
```bash
python twin.py 0 logs/twin_baseline.json
python twin.py 0 logs/twin_baseline.json plots/replay_ep0.gif
```

---

## files

```
marl-lunar-coordination/
├── demo.py             # main entry point
├── train.py            # ppo training + actorcritic network
├── eval.py             # evaluation with logging
├── lunar_comms.py      # packet loss + latency wrapper
├── twin.py             # digital twin: logging, collision prediction, replay
├── plot_results.py     # bar charts from eval csvs
├── stigmergy.py        # rule-based pheromone agent (marl vs stigmergy benchmark)
├── stigmergy.py        # rule-based pheromone agent (marl vs stigmergy benchmark)
├── requirements.txt
├── .gitignore
├── checkpoints/
│   └── policy_final.pt # trained weights, no training needed
├── logs/
└── plots/
```

---


---

## marl vs stigmergy

ran a head-to-head between the trained ppo policy and a rule-based pheromone agent (stigmergy) across three conditions. stigmergy works by having each rover deposit a "scent trail" on cells it visits -- trails decay over time and rovers move toward the least-visited areas. no neural net, no training, just local rules.

| condition | marl (ppo) | stigmergy (rule-based) |
|-----------|:----------:|:----------------------:|
| no degradation | -137.77 | -241.47 |
| 40% packet loss | -141.41 | -244.95 |
| agent failure (N-1) | -136.46 | -252.11 |

findings:
- marl outperforms stigmergy by ~75% across all conditions
- both degrade almost identically under 40% packet loss (~2% drop each) -- the task is short-horizon enough that stale observations don't matter much for either approach
- agent failure hits stigmergy harder (-252 vs -136) because stigmergy depends on all three rovers covering separate areas -- lose one and coverage drops noticeably
- the comms robustness advantage for stigmergy (expected, since it doesn't use radio at all) didn't show up at this loss rate -- would likely emerge at >60% loss or with longer latency windows

reproduce it:
```bash
python stigmergy.py
```

outputs `logs/stigmergy_benchmark.csv` and `plots/stigmergy_vs_marl.png`

## how the ppo works

all three agents share one network (parameter sharing -- they're identical so there's no reason to train separate policies). each update cycle collects 10 episodes worth of experience, computes per-agent discounted returns separately, then does 4 gradient steps with ppo clip objective. normalizing both advantages and returns was the key fix to stop the value loss from exploding.

## how the comms wrapper works

wraps the pettingzoo env and intercepts observations before they reach the agents. on each step: new obs goes into a per-agent buffer, agent receives the delayed version if latency > 0, and with probability `packet_loss` the delivered obs gets swapped for the agent's last known observation. not zeros -- stale data, which is what a real rover would actually have.

## how the digital twin works

logs position, velocity, action, and reward for every agent every step. also runs kinematic lookahead each step -- projects agents forward 3 steps using current velocity and flags any pair predicted to collide before it happens. saves to json, replays as gif.

---

built with help from claude (anthropic) for code structure and debugging. experiments and analysis are my own.