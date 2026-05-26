import argparse
import numpy as np
import pandas as pd
import torch
from pettingzoo.mpe import simple_spread_v3
from lunar_comms import LunarCommsWrapper
from twin import DigitalTwin
from train import ActorCritic
import os

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",  type=str,   default="checkpoints/policy_final.pt")
parser.add_argument("--loss",        type=float, default=0.0,   help="packet loss 0.0-1.0")
parser.add_argument("--latency",     type=int,   default=0,     help="obs delay in steps")
parser.add_argument("--episodes",    type=int,   default=30)
parser.add_argument("--fault-agent", type=str,   default=None,  help="e.g. agent_0")
parser.add_argument("--fault-step",  type=int,   default=15,    help="step the fault kicks in")
parser.add_argument("--tag",         type=str,   default="run", help="label for output files")
parser.add_argument("--random",      action="store_true",        help="skip checkpoint, use random actions")
args = parser.parse_args()

os.makedirs("logs", exist_ok=True)

tmp = simple_spread_v3.parallel_env(N=3, max_cycles=50, continuous_actions=False)
tmp.reset()
obs_dim   = tmp.observation_space(tmp.agents[0]).shape[0]
n_actions = tmp.action_space(tmp.agents[0]).n
tmp.close()

policy = None
if not args.random:
    policy = ActorCritic(obs_dim, n_actions)
    policy.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    policy.eval()
    print(f"loaded checkpoint: {args.checkpoint}")
else:
    print("using random actions (no checkpoint)")

twin = DigitalTwin()
rows = []

for ep in range(args.episodes):
    base_env = simple_spread_v3.parallel_env(N=3, max_cycles=50, continuous_actions=False)
    env      = LunarCommsWrapper(base_env, packet_loss=args.loss, latency_steps=args.latency)

    obs, _       = env.reset(seed=ep)
    total_reward = 0
    completed    = False
    dead_agents  = set()

    for step in range(50):
        if args.fault_agent and step == args.fault_step:
            dead_agents.add(args.fault_agent)
            print(f"  [ep {ep:02d}] {args.fault_agent} failed at step {step}")

        actions = {}
        for agent in env.agents:
            if agent in dead_agents:
                actions[agent] = 0
            elif policy is not None:
                a, _, _ = policy.get_action(obs[agent])
                actions[agent] = a
            else:
                actions[agent] = env.action_space(agent).sample()

        obs, rewards, terms, truncs, _ = env.step(actions)

        agent_states = {}
        for agent, ob in obs.items():
            agent_states[agent] = {
                "pos":    ob[:2].tolist(),
                "vel":    ob[2:4].tolist() if len(ob) >= 4 else [0.0, 0.0],
                "action": int(actions.get(agent, 0)),
                "reward": float(rewards.get(agent, 0.0))
            }

        for w in twin.predict_collision(agent_states):
            print(f"  [twin] {w['agents']} collision warning -- {w['steps_until_collision']} steps out")

        twin.log_step(step, agent_states)
        total_reward += sum(rewards.values())

        if all(terms.values()) or all(truncs.values()):
            completed = True
            break

    twin.commit_episode(ep, metadata={
        "packet_loss": args.loss,
        "latency":     args.latency,
        "fault_agent": args.fault_agent,
        "completed":   completed
    })

    rows.append({
        "episode":       ep,
        "total_reward":  total_reward,
        "completed":     int(completed),
        "packet_loss":   args.loss,
        "latency_steps": args.latency,
        "fault_agent":   args.fault_agent or "none"
    })
    print(f"ep {ep+1:02d}/{args.episodes}  |  reward: {total_reward:8.2f}  |  done: {completed}")

df = pd.DataFrame(rows)
df.to_csv(f"logs/eval_{args.tag}.csv", index=False)
twin.save(f"logs/twin_{args.tag}.json")

print(f"\n--- {args.tag} ---")
print(f"mean reward:     {df['total_reward'].mean():.3f}")
print(f"completion rate: {df['completed'].mean() * 100:.1f}%")
