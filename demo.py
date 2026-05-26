#!/usr/bin/env python3
# pull the repo, run this, done.
# example: python demo.py --loss 0.2 --episodes 5 --replay
# live window: python demo.py --render --episodes 3

import argparse
import numpy as np
import torch
from pettingzoo.mpe import simple_spread_v3
from lunar_comms import LunarCommsWrapper
from twin import DigitalTwin
from train import ActorCritic

parser = argparse.ArgumentParser(description="run trained lunar MARL agents")
parser.add_argument("--loss",        type=float, default=0.0,  help="packet loss rate (0.0-1.0)")
parser.add_argument("--latency",     type=int,   default=0,    help="obs delay in steps")
parser.add_argument("--episodes",    type=int,   default=5,    help="number of episodes to run")
parser.add_argument("--fault-agent", type=str,   default=None, help="agent to kill mid-episode (e.g. agent_0)")
parser.add_argument("--fault-step",  type=int,   default=15,   help="step the fault happens")
parser.add_argument("--replay",      action="store_true",      help="save a gif replay to plots/demo_replay.gif")
parser.add_argument("--render",      action="store_true",      help="open live pygame window (press Q to quit)")
parser.add_argument("--checkpoint",  type=str,   default="checkpoints/policy_final.pt")
args = parser.parse_args()

# setup
tmp = simple_spread_v3.parallel_env(N=3, max_cycles=50, continuous_actions=False)
tmp.reset()
obs_dim   = tmp.observation_space(tmp.agents[0]).shape[0]
n_actions = tmp.action_space(tmp.agents[0]).n
tmp.close()

policy = ActorCritic(obs_dim, n_actions)
policy.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
policy.eval()
print(f"loaded: {args.checkpoint}")
print(f"running {args.episodes} episodes | loss={args.loss:.0%} | latency={args.latency} steps | fault={args.fault_agent or 'none'}\n")

twin    = DigitalTwin()
results = []

for ep in range(args.episodes):
    render_mode = "human" if args.render else "rgb_array"
    base_env    = simple_spread_v3.parallel_env(N=3, max_cycles=50, continuous_actions=False, render_mode=render_mode)
    env         = LunarCommsWrapper(base_env, packet_loss=args.loss, latency_steps=args.latency)
    obs, _      = env.reset(seed=ep)
    total_r     = 0
    completed   = False
    dead_agents = set()

    for step in range(50):
        if args.fault_agent and step == args.fault_step:
            dead_agents.add(args.fault_agent)

        actions = {}
        for agent in env.agents:
            if agent in dead_agents:
                actions[agent] = 0
            else:
                a, _, _ = policy.get_action(obs[agent])
                actions[agent] = a

        obs, rewards, terms, truncs, _ = env.step(actions)

        if args.render:
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    pygame.quit()
                    exit()

        agent_states = {
            a: {
                "pos":    ob[:2].tolist(),
                "vel":    ob[2:4].tolist() if len(ob) >= 4 else [0.0, 0.0],
                "action": int(actions.get(a, 0)),
                "reward": float(rewards.get(a, 0.0))
            }
            for a, ob in obs.items()
        }
        twin.log_step(step, agent_states)
        total_r += sum(rewards.values())

        if all(terms.values()) or all(truncs.values()):
            completed = True
            break

    base_env.close()
    twin.commit_episode(ep, metadata={
        "packet_loss": args.loss, "latency": args.latency,
        "fault_agent": args.fault_agent, "completed": completed
    })
    results.append((total_r, completed))
    print(f"ep {ep+1}/{args.episodes}  |  reward: {total_r:8.2f}  |  done: {completed}")

mean_r    = np.mean([r for r, _ in results])
comp_rate = np.mean([int(c) for _, c in results]) * 100
print(f"\nmean reward:     {mean_r:.3f}")
print(f"completion rate: {comp_rate:.1f}%")

if args.replay:
    import os
    os.makedirs("plots", exist_ok=True)
    twin.save("logs/demo_twin.json")
    twin.replay(episode_id=0, save_path="plots/demo_replay.gif")
    print("replay saved to plots/demo_replay.gif")