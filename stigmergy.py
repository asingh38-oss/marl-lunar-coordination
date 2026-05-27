# stigmergy.py
# rule-based pheromone agent benchmarked against trained PPO policy
# no training, no neural net -- just local rules and indirect coordination
#
# how it works:
#   each agent deposits a "scent trail" on cells it visits
#   the trail decays over time
#   agents move toward the least-visited (lowest pheromone) areas
#   this is stigmergy -- coordination through environment, not direct comms
#
# key comparison: does pheromone beat PPO? and which holds up better
# when comms break down or a rover dies mid-mission?

import numpy as np
import pandas as pd
import torch
import os
from pettingzoo.mpe import simple_spread_v3
from lunar_comms import LunarCommsWrapper
from train import ActorCritic

os.makedirs("logs", exist_ok=True)
os.makedirs("plots", exist_ok=True)

GRID_SIZE = 20
WORLD_MIN = -1.5
WORLD_MAX =  1.5


class PheromoneGrid:
    # shared grid all agents read from and write to
    # this is the stigmergy mechanism -- indirect comms through the environment
    # no radio needed, no packet loss -- just local sensing

    def __init__(self, size=GRID_SIZE, decay=0.92):
        self.size  = size
        self.decay = decay
        self.grid  = np.zeros((size, size))

    def reset(self):
        self.grid = np.zeros((self.size, self.size))

    def _to_grid(self, x, y):
        gx = int((x - WORLD_MIN) / (WORLD_MAX - WORLD_MIN) * self.size)
        gy = int((y - WORLD_MIN) / (WORLD_MAX - WORLD_MIN) * self.size)
        return np.clip(gx, 0, self.size - 1), np.clip(gy, 0, self.size - 1)

    def deposit(self, x, y, amount=1.0):
        gx, gy = self._to_grid(x, y)
        self.grid[gx, gy] += amount

    def decay_step(self):
        self.grid *= self.decay

    def neighbor_pheromones(self, x, y):
        # pheromone level in each direction + current cell
        # action mapping: 0=stay, 1=left, 2=right, 3=down, 4=up
        gx, gy = self._to_grid(x, y)
        s = self.size - 1
        return {
            0: self.grid[gx, gy],
            1: self.grid[max(0, gx-1), gy],
            2: self.grid[min(s, gx+1), gy],
            3: self.grid[gx, max(0, gy-1)],
            4: self.grid[gx, min(s, gy+1)],
        }


def get_pos(obs):
    # obs[2:4] is agent position in simple_spread_v3
    return float(obs[2]), float(obs[3])


def stigmergy_action(obs, grid, epsilon=0.15):
    x, y = get_pos(obs)
    grid.deposit(x, y)

    # small random chance to explore so agents don't get completely stuck
    if np.random.random() < epsilon:
        return np.random.randint(0, 5)

    # go toward the least-visited neighboring cell
    neighbors = grid.neighbor_pheromones(x, y)
    return min(neighbors, key=neighbors.get)


def run_episode(strategy, policy=None, grid=None, packet_loss=0.0,
                fault_agent=None, fault_step=15, seed=0):
    base_env = simple_spread_v3.parallel_env(
        N=3, max_cycles=50, continuous_actions=False
    )

    if packet_loss > 0:
        env = LunarCommsWrapper(base_env, packet_loss=packet_loss)
    else:
        env = base_env

    obs, _ = env.reset(seed=seed)

    if grid is not None:
        grid.reset()

    total_reward = 0
    dead_agents  = set()

    for step in range(50):
        if fault_agent and step == fault_step:
            dead_agents.add(fault_agent)

        actions = {}
        for agent in env.agents:
            if agent in dead_agents:
                actions[agent] = 0
            elif strategy == "stigmergy":
                actions[agent] = stigmergy_action(obs[agent], grid)
            elif strategy == "marl":
                a, _, _ = policy.get_action(obs[agent])
                actions[agent] = a
            else:
                actions[agent] = env.action_space(agent).sample()

        obs, rewards, terms, truncs, _ = env.step(actions)

        if grid is not None:
            grid.decay_step()

        total_reward += sum(rewards.values())
        if all(terms.values()) or all(truncs.values()):
            break

    base_env.close()
    return total_reward


def benchmark(n_episodes=50):
    # load trained MARL policy
    tmp = simple_spread_v3.parallel_env(N=3, max_cycles=50, continuous_actions=False)
    tmp.reset()
    obs_dim   = tmp.observation_space(tmp.agents[0]).shape[0]
    n_actions = tmp.action_space(tmp.agents[0]).n
    tmp.close()

    policy = ActorCritic(obs_dim, n_actions)
    policy.load_state_dict(torch.load("checkpoints/policy_final.pt", map_location="cpu"))
    policy.eval()
    print("loaded policy")

    grid = PheromoneGrid()
    rows = []

    conditions = [
        # (label, strategy, packet_loss, fault_agent)
        ("MARL -- no degradation",      "marl",      0.0,  None),
        ("Stigmergy -- no degradation",  "stigmergy", 0.0,  None),
        ("MARL -- 40% packet loss",      "marl",      0.4,  None),
        ("Stigmergy -- 40% packet loss", "stigmergy", 0.4,  None),
        ("MARL -- agent failure",        "marl",      0.0,  "agent_0"),
        ("Stigmergy -- agent failure",   "stigmergy", 0.0,  "agent_0"),
    ]

    for label, strategy, loss, fault in conditions:
        rewards = []
        for ep in range(n_episodes):
            r = run_episode(
                strategy    = strategy,
                policy      = policy if strategy == "marl" else None,
                grid        = grid   if strategy == "stigmergy" else None,
                packet_loss = loss,
                fault_agent = fault,
                seed        = ep
            )
            rewards.append(r)

        mean_r = np.mean(rewards)
        std_r  = np.std(rewards)
        rows.append({
            "condition":    label,
            "strategy":     strategy,
            "packet_loss":  loss,
            "fault":        fault or "none",
            "mean_reward":  round(mean_r, 2),
            "std_reward":   round(std_r, 2),
        })
        print(f"{label:<40} mean: {mean_r:8.2f}  std: {std_r:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv("logs/stigmergy_benchmark.csv", index=False)
    print("\nsaved to logs/stigmergy_benchmark.csv")
    return df


def plot_benchmark(df=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.style as mplstyle
    mplstyle.use("dark_background")

    if df is None:
        df = pd.read_csv("logs/stigmergy_benchmark.csv")

    # split by strategy
    marl_rows  = df[df["strategy"] == "marl"]
    stig_rows  = df[df["strategy"] == "stigmergy"]

    labels = ["No degradation", "40% packet loss", "Agent failure"]
    marl_means  = marl_rows["mean_reward"].tolist()
    stig_means  = stig_rows["mean_reward"].tolist()
    marl_stds   = marl_rows["std_reward"].tolist()
    stig_stds   = stig_rows["std_reward"].tolist()

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width/2, marl_means, width, yerr=marl_stds,
                label="MARL (trained PPO)", color="#448aff", alpha=0.85,
                edgecolor="white", linewidth=0.4, capsize=4)
    b2 = ax.bar(x + width/2, stig_means, width, yerr=stig_stds,
                label="Stigmergy (rule-based)", color="#00e676", alpha=0.85,
                edgecolor="white", linewidth=0.4, capsize=4)

    ax.set_ylabel("mean total reward", fontsize=11)
    ax.set_title("MARL vs Stigmergy: coordination under degraded conditions", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                f"{h:.0f}", ha="center", va="bottom", fontsize=8, color="white")

    plt.tight_layout()
    out = "plots/stigmergy_vs_marl.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"chart saved: {out}")
    plt.close()


if __name__ == "__main__":
    print(f"running benchmark ({50} episodes per condition)...")
    df = benchmark(n_episodes=50)

    print("\n--- summary ---")
    print(df[["condition", "mean_reward", "std_reward"]].to_string(index=False))

    plot_benchmark(df)