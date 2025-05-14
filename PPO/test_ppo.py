import os

import gym
import torch
import torch.nn as nn
import pygame
import numpy as np
from torch.distributions import Normal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

env = gym.make(id='Pendulum-v1', render_mode='rgb_array')
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'
actor_path = model + 'ppo_actor_20250423144455.pth'


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mean = self.tanh(self.fc_mean(x)) * 2
        std = self.softplus(self.fc_std(x)) + 1e-3
        return mean, std

    def select_action(self, s):
        with torch.no_grad():
            mu, sigma = self.forward(s)
            normal_dist = Normal(mu, sigma)
            action = normal_dist.sample()
            action = action.clamp(-2.0, 2.0)
        return action


actor = Actor(STATE_DIM, ACTION_DIM).to(device)
actor.load_state_dict(torch.load(actor_path))


def process_frame(frame):
    frame = np.transpose(frame, (1, 0, 2))
    frame = pygame.surfarray.make_surface(frame)
    return pygame.transform.scale(frame, (width, height))


pygame.init()
width, height = 600, 600
screen = pygame.display.set_mode((width, height))

NUM_EPISODE = 30
NUM_STEP = 200
for episode_i in range(NUM_EPISODE):
    state, other = env.reset()
    episode_reward = 0
    for tep_i in range(NUM_STEP):
        action = actor.select_action(torch.FloatTensor(state).unsqueeze(0).to(device)).detach().cpu().numpy()[0]
        next_state, reward, done, truncation, info = env.step(action)
        state = next_state
        episode_reward += reward

        frame = env.render()
        frame = process_frame(frame)
        screen.blit(frame, (0, 0))
        pygame.display.flip()
        clock = pygame.time.Clock()
        clock.tick(60)  # ftp

    print(f'Episode Reward: {episode_i}, Reward: {episode_reward}')
pygame.quit()
env.close()
