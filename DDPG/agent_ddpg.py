import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random

import torch.optim as optim
import torch.nn.functional as F
# 超参数
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
MEMORY_SIZE = 100000
BATCH_SIZE = 64
TAU = 5e-3

# Set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(f'Device type:{device}')


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
        # print(f"action:{x.shape}")
        # 使用 tanh 函数将第二个动作（功率）的输出限制在 [-1, 1] 之间，然后映射到 [1, 5] 之间
        # x[:, 1] = (torch.tanh(x[:, 1]) + 1) * 2 + 1
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # 输出具体的Q值

class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add_memo(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)  # 升维
        next_state = np.expand_dims(next_state, 0)  # 升维
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        # np.concatenate() 降维
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())  # copy
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())  # copy
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.replay_buffer = ReplayMemory(MEMORY_SIZE)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        # print(f'state.shape:{state.shape}')
        action = self.actor(state)
        # print(f'action.shape:{action.shape}')
        return action.detach().cpu().numpy()[0]

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        # print(f'actions:{actions},\nnp.vstack(actions){np.vstack(actions)}')
        # print(f'rewards:{rewards},\ntorch.FloatTensor(rewards).unsqueeze(1){torch.FloatTensor(rewards).unsqueeze(1)}')
        # print(f'states.shape:{np.array(states).shape}')
        # print(f'actions.shape:{np.array(actions).shape}')
        # print(f'rewards.shape:{np.array(rewards).shape}')
        # print(f'next_states.shape:{np.array(next_states).shape}')
        # print(f'dones.shape:{np.array(dones).shape}')

        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(np.vstack(actions)).to(device)  # 纵向堆叠
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)  # 横向升维，(64,)-->(64,1)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)  # 横向升维，(64,)-->(64,1)

        # Update critic
        next_actions = self.actor_target(next_states)
        target_Q = self.critic_target(next_states, next_actions.detach())
        target_Q = rewards + (GAMMA * target_Q * (1 - dones))
        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)  # TODO
        self.critic_optimizer.zero_grad()  # 梯度清零
        critic_loss.backward()  # 计算loss的梯度
        self.critic_optimizer.step()  # 更新参数

        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()  # 梯度清零
        actor_loss.backward()  # 计算loss的梯度
        self.actor_optimizer.step()  # 更新参数

        # Update target networks of critic and actor
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1-TAU) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1-TAU) * target_param.data)



