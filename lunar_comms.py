import numpy as np
from collections import defaultdict, deque


class LunarCommsWrapper:
    # wraps a pettingzoo parallel env and degrades observations
    # to simulate what actually happens on a lunar comms link:
    # packets get dropped and signals arrive late
    #
    # packet_loss:   0.0 = perfect, 1.0 = everything dropped
    # latency_steps: how many steps behind the delivered obs is
    #
    # when a packet drops we fall back to the last known obs
    # (not zeros -- a real agent would use stale data, not nothing)

    def __init__(self, env, packet_loss=0.0, latency_steps=0):
        self.env           = env
        self.packet_loss   = packet_loss
        self.latency_steps = latency_steps
        self.last_obs      = {}

        lt = latency_steps
        self.obs_buffer = defaultdict(lambda: deque(maxlen=lt + 1))

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_obs = {a: ob.copy() for a, ob in obs.items()}

        for agent, ob in obs.items():
            self.obs_buffer[agent].clear()
            for _ in range(self.latency_steps):
                self.obs_buffer[agent].append(ob.copy())

        return obs, info

    def step(self, actions):
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        degraded = {}

        for agent, ob in obs.items():
            self.obs_buffer[agent].append(ob.copy())

            if self.latency_steps > 0 and len(self.obs_buffer[agent]) > 0:
                delivered = self.obs_buffer[agent][0]
            else:
                delivered = ob

            if np.random.random() < self.packet_loss:
                degraded[agent] = self.last_obs.get(agent, ob).copy()
            else:
                degraded[agent] = delivered.copy()
                self.last_obs[agent] = delivered.copy()

        return degraded, rewards, terms, truncs, infos

    def __getattr__(self, name):
        return getattr(self.env, name)
