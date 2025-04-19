import torch

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
        return action.cpu().detach().numpy()

    def learn(self, obs, act, reward, next_obs, done):
        obs = torch.FloatTensor(obs)
        act = torch.FloatTensor(act)
        reward = torch.FloatTensor(reward)
        next_obs = torch.FloatTensor(next_obs)
        done = torch.FloatTensor(done)

        _, loss = self.algorithm.learn(obs, act, reward, next_obs, done)
        self._sync_target()
        return loss
