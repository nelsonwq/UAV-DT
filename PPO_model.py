import torch
import torch.nn as nn
from torch.distributions import Normal


class PPOModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, act_dim))
        self.critic = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def get_action_dist(self, obs):
        return Normal(self.actor(obs), torch.ones_like(self.actor(obs)))  # 连续动作