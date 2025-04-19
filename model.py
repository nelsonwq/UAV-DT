import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, obs_dim, act_dim): # (5,2)
        super().__init__()
        self.actor_model = ActorModel(obs_dim, act_dim)
        self.critic_model = CriticModel(obs_dim, act_dim)

    def forward(self, obs, act=None):
        if act is None:
            return self.actor_model(obs)
        return self.critic_model(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()

class ActorModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hid_size = 100
        self.fc1 = nn.Linear(obs_dim, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, act_dim)

    def forward(self, obs):
        hid = F.relu(self.fc1(obs))
        raw_output = torch.sigmoid(self.fc2(hid))
        means = 0.1 + 0.8 * torch.sigmoid(raw_output)  # 输出约束为[0.1, 0.9]
        # print(f"means--{means}")
        return means

class CriticModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.hid_size = 100
        self.obs_dim = obs_dim
        self.concat_dim = obs_dim + act_dim

        self.fc1 = nn.Linear(self.concat_dim, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, 1)

    def forward(self, obs, act):
        # print(obs.shape, act.shape)
        obs = obs.view(-1, self.obs_dim)
        act = act.view(-1, 1)
        # print(obs.shape, act.shape)
        concat = torch.cat([obs, act], dim=1)
        hid = F.relu(self.fc1(concat))

        Q = self.fc2(hid).squeeze(1)
        # print(Q.shape)
        return Q
