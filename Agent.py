import torch
import numpy as np

class Agent:
    def __init__(self, algorithm, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.algorithm = algorithm
        self._sync_target()

    def _sync_target(self):
        self.algorithm.target_model.load_state_dict(self.algorithm.model.state_dict())

    def predict(self, obs):
        # print(f'Agent----{obs}')
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs)
            action = self.algorithm.predict(obs_tensor)
            # print(f"action===={action}")
        return action

    def learn(self, obs, act, reward, next_obs, done):
        # print(f'obs.shape:{obs.shape}')
        # print(f'act.shape:{act.shape}')
        # print(f'reward.shape:{reward.shape}')
        # print(f'next_obs.shape:{next_obs.shape}')
        # print(f'done.shape:{done.shape}')
        obs = torch.FloatTensor(obs)
        act = torch.FloatTensor(np.vstack(act))
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_obs = torch.FloatTensor(next_obs)
        done = torch.FloatTensor(done).unsqueeze(1)

        _, loss = self.algorithm.learn(obs, act, reward, next_obs, done)
        self._sync_target()
        return loss
