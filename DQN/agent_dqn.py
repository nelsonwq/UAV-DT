import random

import numpy as np
import torch
from torch import nn


class Replaymemory:
    def __init__(self, n_state, n_action):
        self.n_state = n_state
        self.n_action = n_action
        self.MEMORY_SIZE = 1000
        self.BATCH_SIZE = 64

        self.all_state = np.empty(shape=(self.MEMORY_SIZE, self.n_state), dtype=np.float32)
        self.all_action = np.random.randint(low=0, high=n_action, size=self.MEMORY_SIZE, dtype=np.uint8)
        self.all_reward = np.empty(self.MEMORY_SIZE, dtype=np.float32)
        self.all_done = np.random.randint(low=0, high=2, size=self.MEMORY_SIZE, dtype=np.uint8)
        self.all_next_state = np.empty(shape=(self.MEMORY_SIZE, self.n_state), dtype=np.float32)
        self.t_memo = 0
        self.t_max = 0

    def add_memo(self, state, action, reward, next_state, done):
        self.all_state[self.t_memo] = state
        self.all_action[self.t_memo] = action
        self.all_reward[self.t_memo] = reward
        self.all_next_state[self.t_memo] = next_state
        self.all_done[self.t_memo] = done
        self.t_max = max(self.t_max, self.t_memo + 1)
        self.t_memo = (self.t_memo + 1) % self.MEMORY_SIZE

    def sample(self):

        if self.t_max > self.BATCH_SIZE:
            idxes = random.sample(range(0, self.t_max), self.BATCH_SIZE)
        else:
            idxes = range(0, self.t_max)

        batch_state = []
        batch_action = []
        batch_reward = []
        batch_done = []
        batch_next_state = []

        for idx in idxes:
            batch_state.append(self.all_state[idx])
            batch_action.append(self.all_action[idx])
            batch_reward.append(self.all_reward[idx])
            batch_done.append(self.all_done[idx])
            batch_next_state.append(self.all_next_state[idx])

        batch_state_tensor = torch.as_tensor(np.asarray(batch_state), dtype=torch.float32)
        batch_action_tensor = torch.as_tensor(np.asarray(batch_action), dtype=torch.int64).unsqueeze(-1)
        batch_reward_tensor = torch.as_tensor(np.asarray(batch_reward), dtype=torch.float32).unsqueeze(-1)
        batch_next_state_tensor = torch.as_tensor(np.asarray(batch_next_state), dtype=torch.float32)
        batch_done_tensor = torch.as_tensor(np.asarray(batch_done), dtype=torch.float32).unsqueeze(-1)

        return batch_state_tensor, batch_action_tensor, batch_reward_tensor, batch_next_state_tensor, batch_done_tensor


class DQN(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=n_input, out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=n_output)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        q_value = self(obs_tensor.unsqueeze(0))
        # print(f'q_value:{q_value}')
        max_q_idx = torch.argmax(input=q_value)
        # print(f'max_q_idx:{max_q_idx}')
        action = max_q_idx.detach().item()
        return action


class Agent:
    def __init__(self, n_input, n_output):
        self.n_input = n_input
        self.n_output = n_output

        self.GAMMA = 0.99
        self.learning_rate = 3e-3

        self.memo = Replaymemory(n_state=self.n_input, n_action=self.n_output)  # TODO

        self.online_net = DQN(self.n_input, self.n_output)  # TODO
        self.target_net = DQN(self.n_input, self.n_output)  # TODO

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)  # TODO
