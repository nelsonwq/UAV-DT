import numpy as np
from torch import nn
import torch
from torch.distributions import Normal
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Computing device:{device}')


class
    (nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mean = self.sigmoid(self.fc_mean(x))  # [0, 1]
        std = self.softplus(self.fc_std(x)) + 1e-3
        # std = torch.clamp(std, max=10)

        # mean0 = self.sigmod(mean[:, 0])  # [0,1]
        #
        # mean1 = (self.tanh(mean[:, 1]) + 1) * 2 + 1  # [1,5]
        # 增加标准差的下限，确保有足够的随机性
        # std0 = self.softplus(std[:, 0]) + 1e-3
        # std1 = self.softplus(std[:, 1]) + 1e-3
        # mean = torch.stack([mean0, mean1], dim=1)
        # std = torch.stack([std0, std1], dim=1)
        return mean, std

    def select_action(self, s):
        with torch.no_grad():
            mu, sigma = self.forward(s)
            # print(f'mu:{mu}, sigma:{sigma}')
            normal_dist = Normal(mu, sigma)
            action = normal_dist.sample()
            # action = torch.clamp(action, 0, 1)
            # 使用 tanh 函数平滑处理后映射到 [0, 1]
            action = torch.sigmoid(action)
            # print(f'action:{action}')
            # print(f'mu:{mu}, sigma:{sigma}')
            # normal_dist0 = Normal(mu[:, 0], sigma[:, 0])
            # normal_dist1 = Normal(mu[:, 1], sigma[:, 1])
            #
            # action0 = normal_dist0.sample()
            # action1 = normal_dist1.sample()

            # 对动作进行裁剪，确保动作在指定范围内
            # print(f'action0:{action0}, action1:{action1}')
            # action0 = torch.clamp(action0, min=0, max=1)
            # action1 = torch.clamp(action1, min=1, max=5)
            # print(f'action0:{action0}, action1:{action1}')
            # action = action.clamp(-2.0, 2.0)
            # 使用 sigmoid 函数将第一个动作（任务分配比例）限制在 [0, 1] 之间
            # action[:, 0] = torch.sigmoid(action[:, 0])
            # # 使用 tanh 函数将第二个动作（功率）的输出限制在 [-1, 1] 之间，然后映射到 [1, 5] 之间
            # action[:, 1] = (torch.tanh(action[:, 1]) + 1) * 2 + 1
            # action = torch.stack([action0, action1], dim=1)
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        value = self.fc4(x)
        return value


class ReplayMemory:
    def __init__(self, batch_size):
        self.state_cap = []
        self.action_cap = []
        self.reward_cap = []
        self.value_cap = []
        self.done_cap = []
        self.BATCH_SIZE = batch_size

    def add_memo(self, state, action, reward, value, done):
        self.state_cap.append(state)
        self.action_cap.append(action)
        self.reward_cap.append(reward)
        self.value_cap.append(value)
        self.done_cap.append(done)

    def sample(self):
        num_state = len(self.state_cap)
        batch_start_points = np.arange(0, num_state, self.BATCH_SIZE)
        memory_indices = np.arange(num_state, dtype=np.int32)
        np.random.shuffle(memory_indices)
        batches = [memory_indices[i: i + self.BATCH_SIZE] for i in batch_start_points]

        return np.array(self.state_cap), \
            np.array(self.action_cap), \
            np.array(self.reward_cap), \
            np.array(self.value_cap), \
            np.array(self.done_cap), \
            batches

    def clear_memo(self):
        self.state_cap = []
        self.action_cap = []
        self.reward_cap = []
        self.value_cap = []
        self.done_cap = []


class PPOAgent:
    def __init__(self, state_dim, action_dim, batch_size):
        self.LR_ACTOR = 3e-4
        self.LR_CRITIC = 3e-4
        self.GAMMA = 0.98
        self.LAMBDA = 0.95
        self.EPOCH = 20
        self.EPSILON_CLIP = 0.15

        self.actor = Actor(state_dim, action_dim).to(device)
        self.old_actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.LR_CRITIC)
        self.replay_buffer = ReplayMemory(batch_size)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor.select_action(state)
        value = self.critic.forward(state)
        return action.detach().cpu().numpy()[0], value.detach().cpu().numpy()[0]

    def update(self):
        self.old_actor.load_state_dict(self.actor.state_dict())
        for epoch_i in range(self.EPOCH):
            memo_states, memo_actions, memo_rewards, memo_values, memo_dones, batches = self.replay_buffer.sample()
            T = len(memo_rewards)
            memo_advantages = np.zeros(T, dtype=np.float32)

            for t in range(T):
                discount = 1
                a_t = 0
                for k in range(t, T - 1):
                    a_t += memo_rewards[k] + self.GAMMA * memo_values[k + 1] * (1 - int(memo_dones[k])) - memo_values[k]
                    discount *= self.GAMMA * self.LAMBDA
                memo_advantages[t] = a_t

            with torch.no_grad():
                memo_advantages_tensor = torch.tensor(memo_advantages).unsqueeze(1).to(device)
                memo_values_tensor = torch.tensor(memo_values).to(device)

            memo_states_tensor = torch.FloatTensor(memo_states).to(device)
            memo_actions_tensor = torch.FloatTensor(memo_actions).to(device)
            for batch in batches:
                with torch.no_grad():
                    old_mu, old_sigma = self.old_actor(memo_states_tensor[batch])
                    old_pi = Normal(old_mu, old_sigma)
                batch_old_probs_tensor = old_pi.log_prob(memo_actions_tensor[batch])

                mu, sigma = self.actor(memo_states_tensor[batch])
                pi = Normal(mu, sigma)

                batch_prob_tensor = pi.log_prob(memo_actions_tensor[batch])

                ratio = torch.exp(batch_prob_tensor - batch_old_probs_tensor)
                surr1 = ratio * memo_advantages_tensor[batch]
                surr2 = torch.clamp(ratio, 1 - self.EPSILON_CLIP, 1 + self.EPSILON_CLIP) * memo_advantages_tensor[batch]

                actor_loss = -torch.min(surr1, surr2).mean()

                batch_returns = memo_advantages_tensor[batch] + memo_values_tensor[batch]

                batch_old_values = self.critic(memo_states_tensor[batch])

                critic_loss = nn.MSELoss()(batch_old_values, batch_returns)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

        self.replay_buffer.clear_memo()

    def save_policy(self):
        torch.save(self.actor.state_dict(), 'ppo_policy_pendulum_v1.para')
