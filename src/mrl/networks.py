"""Codificador de imágenes compartido por todos los agentes.

Es la CNN "Nature" de DQN (Mnih et al., 2015), la misma que usan los ejemplos de
Atari de Tianshou para PPO y SAC discreto, adaptada a imágenes de 64x64.

Sigue la interfaz que esperan las redes de Tianshou (tianshou.utils.net.discrete):
    forward(obs, state=None, info={}) -> (features, state)
    atributo output_dim
"""
import numpy as np
import torch
from torch import nn


class NatureCNN(nn.Module):
    def __init__(self, in_channels, output_dim=512, device="cpu"):
        super().__init__()
        self.device = device
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),  # 64 -> 15
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),           # 15 -> 6
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),           # 6 -> 4
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, output_dim), nn.ReLU(inplace=True),
        )

    def forward(self, obs, state=None, info={}):
        # La observación llega como uint8 [0, 255] (así ocupa 4 veces menos memoria
        # en el buffer). Se pasa a float [0, 1] recién aquí.
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
        obs = obs.to(self.device, dtype=torch.float32) / 255.0
        return self.net(obs), state
