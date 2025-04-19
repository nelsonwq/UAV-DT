import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np


class PPO:
    def __init__(self,
                 model,
                 clip_epsilon=0.2,  # 截断范围(通常0.1~0.3)
                 gamma=0.99,  # 折扣因子
                 gae_lambda=0.95,  # GAE系数
                 actor_lr=3e-4,  # 策略网络学习率
                 critic_lr=1e-3,  # 价值网络学习率
                 epochs=4,  # 每批数据训练轮次
                 entropy_coef=0.01  # 熵正则化系数
                 ):
        self.model = model  # 策略网络(actor) + 价值网络(critic)
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epochs = epochs
        self.entropy_coef = entropy_coef

        # 优化器分离
        self.actor_optimizer = optim.Adam(self.model.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.model.critic.parameters(), lr=critic_lr)

    def predict(self, obs, deterministic=False):
        """ 预测动作（带探索） """
        with torch.no_grad():
            obs = torch.FloatTensor(obs)
            if hasattr(self.model, 'get_action_dist'):  # 连续动作空间
                dist = self.model.get_action_dist(obs)
                if deterministic:
                    action = dist.mean
                else:
                    action = dist.sample()
                log_prob = dist.log_prob(action)
                return action.numpy(), log_prob.numpy()
            else:  # 离散动作空间
                probs = self.model.actor(obs)
                dist = Categorical(probs)
                action = dist.sample()
                return action.item(), dist.log_prob(action).item()

    def compute_gae(self, rewards, dones, values):
        """ 计算GAE优势估计 """
        advantages = np.zeros_like(rewards)
        # print(rewards.shape)
        # print(advantages.shape)
        last_advantage = 0
        next_value = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + (1 - dones[t]) * self.gamma * next_value - values[t]
            advantages[t] = last_advantage = delta + (1 - dones[t]) * self.gamma * self.gae_lambda * last_advantage
            next_value = values[t]
        # print(advantages.shape)
        return advantages

    def learn(self, batch_obs, batch_actions, batch_log_probs, batch_rewards, batch_dones):
        """ PPO核心训练逻辑 """
        batch_obs = torch.FloatTensor(batch_obs)
        batch_actions = torch.FloatTensor(batch_actions)
        batch_old_log_probs = torch.FloatTensor(batch_log_probs)
        batch_rewards = torch.FloatTensor(batch_rewards)
        batch_dones = torch.FloatTensor(batch_dones)
        # print(f'batch_obs.shape:{batch_obs.shape}')
        # print(f'batch_actions.shape:{batch_actions.shape}')
        # print(f'batch_rewards.shape:{batch_rewards.shape}')
        # print(f'batch_dones.shape:{batch_dones.shape}')

        # 计算GAE和回报
        with torch.no_grad():
            values = self.model.critic(batch_obs).squeeze()
            advantages = self.compute_gae(batch_rewards.numpy(), batch_dones.numpy(), values.numpy()).squeeze()
            # print(values.shape, advantages.shape)
            targets = advantages + values.numpy()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = torch.FloatTensor(advantages)
            targets = torch.FloatTensor(targets)

        # 多轮次优化
        for _ in range(self.epochs):
            # 1. Critic更新（价值函数拟合）
            self.critic_optimizer.zero_grad()
            current_values = self.model.critic(batch_obs).squeeze()
            critic_loss = nn.functional.mse_loss(current_values, targets)
            critic_loss.backward()
            self.critic_optimizer.step()

            # 2. Actor更新（策略优化）
            self.actor_optimizer.zero_grad()
            dist = self.model.get_action_dist(batch_obs)
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            # 重要性采样比率
            ratios = (new_log_probs - batch_old_log_probs).exp()

            # 截断式目标函数
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            actor_loss.backward()
            self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()