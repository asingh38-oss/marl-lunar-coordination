import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import os

mplstyle.use("dark_background")
os.makedirs("plots", exist_ok=True)

conditions = {
    "baseline": "No degradation (0%)",
    "loss_20":  "20% packet loss",
    "loss_40":  "40% packet loss",
    "fault":    "Agent failure (N-1)",
}
colors = ["#00e676", "#ffab40", "#ff5252", "#448aff"]

def load(tag):
    path = f"logs/eval_{tag}.csv"
    if not os.path.exists(path):
        print(f"  skipping {tag} -- not found")
        return None
    return pd.read_csv(path)

dfs   = {tag: load(tag) for tag in conditions}
valid = {tag: df for tag, df in dfs.items() if df is not None}

if not valid:
    print("no eval CSVs in logs/ -- run eval.py first")
    exit()

labels = [conditions[t] for t in valid]
means  = [df["total_reward"].mean()    for df in valid.values()]
rates  = [df["completed"].mean() * 100 for df in valid.values()]
clrs   = colors[:len(valid)]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Lunar MARL: Coordination Under Degraded Communications",
             fontsize=13, fontweight="bold", y=1.01)

ax = axes[0]
bars = ax.bar(labels, means, color=clrs, alpha=0.85, edgecolor="white", linewidth=0.4)
ax.set_ylabel("mean total reward", fontsize=11)
ax.set_title("Reward by condition", fontsize=11)
ax.tick_params(axis="x", rotation=18, labelsize=9)
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9)

ax = axes[1]
bars = ax.bar(labels, rates, color=clrs, alpha=0.85, edgecolor="white", linewidth=0.4)
ax.set_ylabel("mission completion (%)", fontsize=11)
ax.set_ylim(0, 115)
ax.set_title("Mission completion rate", fontsize=11)
ax.tick_params(axis="x", rotation=18, labelsize=9)
for bar, val in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.0f}%", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
out = "plots/results_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved: {out}")

print(f"\n{'condition':<28} {'mean reward':>12}  {'completion':>10}")
print("-" * 54)
for tag, df in valid.items():
    print(f"{conditions[tag]:<28} {df['total_reward'].mean():>12.2f}  {df['completed'].mean()*100:>9.1f}%")
