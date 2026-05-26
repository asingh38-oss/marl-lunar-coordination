import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pettingzoo.mpe import simple_spread_v3
import pandas as pd
import os


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh()
        )
        self.actor  = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def get_action(self, obs_np):
        with torch.no_grad():
            obs = torch.FloatTensor(obs_np).unsqueeze(0)
            logits, value = self(obs)
            dist   = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()


def ppo_update(policy, optimizer, obs_b, act_b, old_lp_b, ret_b, adv_b):
    obs_t    = torch.FloatTensor(np.array(obs_b))
    act_t    = torch.LongTensor(act_b)
    old_lp_t = torch.FloatTensor(old_lp_b)
    ret_t    = torch.FloatTensor(ret_b)
    adv_t    = torch.FloatTensor(adv_b)

    # normalize both -- keeps value loss from exploding
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)

    for _ in range(4):
        logits, values = policy(obs_t)
        dist      = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(act_t)
        entropy   = dist.entropy().mean()

        ratio = torch.exp(log_probs - old_lp_t)
        surr  = torch.min(
            ratio * adv_t,
            torch.clamp(ratio, 0.8, 1.2) * adv_t
        )
        loss = -surr.mean() + 0.5 * (ret_t - values).pow(2).mean() - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

    return loss.item()


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = simple_spread_v3.parallel_env(
        N=3, local_ratio=0.5, max_cycles=50, continuous_actions=False
    )
    env.reset()
    sample_agent = env.agents[0]
    obs_dim      = env.observation_space(sample_agent).shape[0]
    n_actions    = env.action_space(sample_agent).n
    print(f"env ready -- obs_dim={obs_dim}, n_actions={n_actions}")

    policy    = ActorCritic(obs_dim, n_actions)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    n_episodes   = 3000
    gamma        = 0.99
    update_every = 10
    rows         = []

    buf_obs, buf_acts, buf_lps, buf_rets, buf_advs = [], [], [], [], []

    print(f"training {n_episodes} episodes (update every {update_every})...")

    for ep in range(n_episodes):
        obs, _ = env.reset()

        agent_obs  = {a: [] for a in env.agents}
        agent_acts = {a: [] for a in env.agents}
        agent_lps  = {a: [] for a in env.agents}
        agent_vals = {a: [] for a in env.agents}
        agent_rews = {a: [] for a in env.agents}
        total_reward = 0

        for step in range(50):
            actions = {}
            for agent in env.agents:
                a, lp, v = policy.get_action(obs[agent])
                actions[agent] = a
                agent_obs[agent].append(obs[agent])
                agent_acts[agent].append(a)
                agent_lps[agent].append(lp)
                agent_vals[agent].append(v)

            obs, rewards, terms, truncs, _ = env.step(actions)

            for agent in actions:
                agent_rews[agent].append(float(rewards.get(agent, 0.0)))

            total_reward += sum(rewards.values())
            if all(terms.values()) or all(truncs.values()):
                break

        for agent in list(agent_rews.keys()):
            if not agent_rews[agent]:
                continue
            R    = 0
            rets = []
            for r in reversed(agent_rews[agent]):
                R = r + gamma * R
                rets.insert(0, R)

            advs = [r - v for r, v in zip(rets, agent_vals[agent])]
            buf_obs.extend(agent_obs[agent])
            buf_acts.extend(agent_acts[agent])
            buf_lps.extend(agent_lps[agent])
            buf_rets.extend(rets)
            buf_advs.extend(advs)

        rows.append({"episode": ep + 1, "reward": total_reward, "loss": None})

        if (ep + 1) % update_every == 0:
            loss = ppo_update(policy, optimizer, buf_obs, buf_acts, buf_lps, buf_rets, buf_advs)
            buf_obs, buf_acts, buf_lps, buf_rets, buf_advs = [], [], [], [], []

            for r in rows[-update_every:]:
                r["loss"] = loss

            mean_r = np.mean([r["reward"] for r in rows[-100:]])
            print(f"ep {ep+1:4d}/{n_episodes}  |  mean reward (last 100): {mean_r:.2f}  |  loss: {loss:.4f}")

            if (ep + 1) % 500 == 0:
                torch.save(policy.state_dict(), f"checkpoints/policy_ep{ep+1}.pt")

    pd.DataFrame(rows).to_csv("logs/training_log.csv", index=False)
    torch.save(policy.state_dict(), "checkpoints/policy_final.pt")
    print("done. saved to checkpoints/policy_final.pt")
