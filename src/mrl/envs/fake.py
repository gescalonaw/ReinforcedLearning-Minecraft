"""Minecraft simulado para probar el código sin MineRL (segundos, sin Java ni ventana).

Imita la interfaz de MineRL 0.4.4: observación Dict con "pov" (64, 64, 3) uint8
(y "compass" en Navigate), acción Dict con teclas + "camera", y action_space.noop().
La recompensa es aleatoria: sirve para verificar que todo corre, NO para aprender.
"""
from collections import OrderedDict

import gym
import numpy as np

_KEYS_ALL = ["forward", "back", "left", "right", "jump", "sneak", "sprint", "attack"]


class _NoopDict(gym.spaces.Dict):
    def noop(self):
        action = OrderedDict((k, 0) for k in _KEYS_ALL)
        action["camera"] = np.zeros(2, dtype=np.float32)
        return action


class FakeMineRL(gym.Env):
    def __init__(self, compass=False, episode_len=200):
        spaces = {k: gym.spaces.Discrete(2) for k in _KEYS_ALL}
        spaces["camera"] = gym.spaces.Box(-180.0, 180.0, (2,), dtype=np.float32)
        self.action_space = _NoopDict(spaces)
        obs = {"pov": gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)}
        if compass:
            obs["compass"] = gym.spaces.Dict(
                {"angle": gym.spaces.Box(-180.0, 180.0, (), dtype=np.float32)})
        self.observation_space = gym.spaces.Dict(obs)
        self.compass = compass
        self.episode_len = episode_len
        self.rng = np.random.default_rng(0)
        self.t = 0

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        return [seed]

    def _obs(self):
        obs = {"pov": self.rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)}
        if self.compass:
            obs["compass"] = {"angle": np.array(self.rng.uniform(-180, 180), dtype=np.float32)}
        return obs

    def reset(self):
        self.t = 0
        return self._obs()

    def step(self, action):
        self.t += 1
        reward = float(action.get("attack", 0) and self.rng.random() < 0.02)
        return self._obs(), reward, self.t >= self.episode_len, {}

    def close(self):
        pass
