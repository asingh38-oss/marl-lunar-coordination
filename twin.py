import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class DigitalTwin:
    # logs full agent trajectories per episode and lets you:
    #   - predict_collision: kinematic lookahead, flags upcoming collisions
    #   - replay:            animated gif of any stored episode
    #   - save/load:         persist to json between runs

    def __init__(self):
        self.episodes     = []
        self.current_traj = []

    def log_step(self, step, agent_data):
        self.current_traj.append({"step": step, "agents": agent_data})

    def commit_episode(self, ep_id, metadata=None):
        self.episodes.append({
            "episode":    ep_id,
            "trajectory": self.current_traj,
            "metadata":   metadata or {}
        })
        self.current_traj = []

    def save(self, path="logs/twin_log.json"):
        with open(path, "w") as f:
            json.dump(self.episodes, f, indent=2)
        print(f"twin saved: {path}")

    @classmethod
    def load(cls, path):
        t = cls()
        with open(path) as f:
            t.episodes = json.load(f)
        return t

    def predict_collision(self, agent_data, steps_ahead=3, threshold=0.15):
        warnings = []
        agents   = list(agent_data.keys())

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agents[i], agents[j]
                p1 = np.array(agent_data[a]["pos"])
                p2 = np.array(agent_data[b]["pos"])
                v1 = np.array(agent_data[a].get("vel", [0.0, 0.0]))
                v2 = np.array(agent_data[b].get("vel", [0.0, 0.0]))

                for t in range(1, steps_ahead + 1):
                    dist = float(np.linalg.norm((p1 + v1 * t * 0.1) - (p2 + v2 * t * 0.1)))
                    if dist < threshold:
                        warnings.append({
                            "agents":                (a, b),
                            "steps_until_collision": t,
                            "predicted_dist":        round(dist, 4)
                        })
                        break

        return warnings

    def replay(self, episode_id=0, save_path=None):
        if episode_id >= len(self.episodes):
            print(f"episode {episode_id} not in log ({len(self.episodes)} stored)")
            return

        traj = self.episodes[episode_id]["trajectory"]
        meta = self.episodes[episode_id].get("metadata", {})
        n    = len(traj)

        fig, ax = plt.subplots(figsize=(7, 7))
        fig.patch.set_facecolor("#0d1117")
        colors = ["#00e676", "#ff5252", "#448aff"]

        def update(frame):
            ax.clear()
            ax.set_facecolor("#0d1117")
            ax.set_xlim(-1.8, 1.8)
            ax.set_ylim(-1.8, 1.8)
            ax.grid(True, alpha=0.12, color="white")
            ax.tick_params(colors="gray")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a2a2a")

            loss_str = f"{meta.get('packet_loss', 0):.0%} loss"
            fault    = meta.get("fault_agent") or "none"
            ax.set_title(
                f"ep {episode_id}  |  step {frame}  |  {loss_str}  |  fault: {fault}",
                color="white", fontsize=9, pad=8
            )

            for idx, (agent_id, state) in enumerate(traj[frame]["agents"].items()):
                x, y  = state["pos"]
                color = colors[idx % len(colors)]
                label = agent_id.replace("agent_", "A")

                ax.scatter(x, y, c=color, s=150, zorder=5, edgecolors="white", linewidths=0.4)
                ax.annotate(
                    f"{label}  r={state.get('reward', 0):.2f}",
                    (x, y), textcoords="offset points", xytext=(7, 7),
                    color=color, fontsize=8, fontweight="bold"
                )

                vx, vy = state.get("vel", [0, 0])
                if abs(vx) + abs(vy) > 0.01:
                    ax.annotate("",
                        xy=(x + vx * 0.35, y + vy * 0.35), xytext=(x, y),
                        arrowprops=dict(arrowstyle="->", color=color, alpha=0.45)
                    )

            ax.text(0.02, 0.02, f"{frame + 1}/{n}", transform=ax.transAxes, color="gray", fontsize=8)

        ani = animation.FuncAnimation(fig, update, frames=n, interval=120, repeat=True)

        out = save_path or f"logs/replay_ep{episode_id}.gif"
        ani.save(out, writer="pillow", fps=10)
        print(f"saved: {out}")
        plt.close(fig)
        return ani


if __name__ == "__main__":
    import sys
    ep_id    = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    log_path = sys.argv[2] if len(sys.argv) > 2 else "logs/twin_baseline.json"
    out_path = sys.argv[3] if len(sys.argv) > 3 else None

    twin = DigitalTwin.load(log_path)
    twin.replay(episode_id=ep_id, save_path=out_path)
