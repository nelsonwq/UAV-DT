import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

class DDPG:
    def __init__(self, model, gamma=None, tau=None, actor_lr=None, critic_lr=None):

        self.model = model
        self.target_model = deepcopy(model)
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.actor_optimizer = optim.Adam(self.model.get_actor_params(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.model.parameters(), lr=critic_lr)

    def predict(self, obs):
        # print(f'DDPG----{obs}')
        with torch.no_grad():
            action =self.model.forward(obs)
            # print(f"action===={action}")
            return action

    def learn(self, obs, act, reward, next_obs, done):
        actor_loss = self._actor_learn(obs)
        critic_loss = self._critic_learn(obs, act, reward, next_obs, done)
        return actor_loss.item(), critic_loss.item()

    def _actor_learn(self, obs):
        self.actor_optimizer.zero_grad()

        action = self.model.forward(obs)
        Q = self.model.forward(obs, action)
        loss = -Q.mean()

        loss.backward()
        self.actor_optimizer.step()
        return loss

    def _critic_learn(self, obs, act, reward, next_obs, done):
        self.critic_optimizer.zero_grad()

        with torch.no_grad():
            next_action = self.target_model.forward(next_obs)
            next_Q = self.target_model.forward(next_obs, next_action)
            target_Q = reward.view(-1, 1) + (1 - done.view(-1, 1).float()) * self.gamma * next_Q

        current_Q = self.model.forward(obs, act)
        loss = nn.functional.mse_loss(current_Q, target_Q)
        loss.backward()
        self.critic_optimizer.step()
        return loss

    def sync_target(self):
        with torch.no_grad():
            for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

