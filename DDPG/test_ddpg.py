import random

import numpy as np
import torch
from Env_UAV import Environment
from agent_ddpg import DDPGAgent
import os
import torch
import torch.nn as nn


env = Environment()
STATE_DIM = env.state_dim
ACTION_DIM = env.action_dim

current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'
actor_path = model + 'ddpg_actor_20250422210807.pth'
agent = DDPGAgent(STATE_DIM, ACTION_DIM)

# 超参数
NUM_EPISODE = 1000
NUM_STEP = 200
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000

REWARD_BUFFER = np.empty(shape=NUM_EPISODE)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # TODO
        # 使用 sigmoid 函数将第一个动作（任务分配比例）限制在 [0, 1] 之间
        x = torch.sigmoid(x)
        return x


actor = Actor(STATE_DIM, ACTION_DIM)
actor.load_state_dict(torch.load(actor_path))


for episode_i in range(NUM_EPISODE):
    state, state_normalization = env.reset()
    episode_reward = 0
    for step_i in range(NUM_STEP):
        state_normalization = torch.FloatTensor(state_normalization).unsqueeze(0)
        action = actor(state_normalization)
        action = action.detach().cpu().numpy()[0][0]  # TODO
        next_state, next_state_normalization, reward, done = env.step(state, action)
        state = next_state
        state_normalization = next_state_normalization
        episode_reward += reward

        if done:
            break

    REWARD_BUFFER[episode_i] = episode_reward
    print(f'Episode:{episode_i+1}, Reward:{round(episode_reward, 2)}')  # 保留2位小数





