import torch
import torch.nn as nn
import torch.nn.functional as F


class UAVCommModel(nn.Module):
    def __init__(self, obs_dim=5, act_dim=2):
        super().__init__()
        # 观测编码器（处理空间坐标+速率）
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64)
        )

        # 策略网络
        self.actor = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, act_dim),
            PowerAndRatioConstraint()
        )

        # 价值网络
        self.critic = nn.Sequential(
            nn.Linear(64 + act_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs, act=None):
        obs = obs.reshape(-1, 5)
        encoded = self.obs_encoder(obs)
        # print(f'encoded.shape=={encoded.shape}')

        if act is None:
            return self.actor(encoded)
        else:
            act = act.reshape(-1, 2)
            return self.critic(torch.cat([encoded, act], dim=1))

    def get_actor_params(self):
        return self.actor.parameters()

class PowerAndRatioConstraint(nn.Module):
    """动作空间物理约束层"""

    def forward(self, x):

        x = x.reshape(-1, 2)
        # 使用tanh激活+线性变换（梯度更稳定）
        ratio_trans = 0.4 * torch.tanh(x[:, 0]) + 0.5  # 中心在0.5
        safe_ratio = 0.1 + 0.8 * ratio_trans  # [0.1,0.9]

        # 采用Softplus保证正梯度
        power_trans = torch.nn.functional.softplus(x[:, 1])
        safe_power = 1 + 4 * (power_trans / (1 + power_trans))  # [1,5]

        return torch.stack([safe_ratio, safe_power], dim=1)


# 示例用法
if __name__ == "__main__":
    model = UAVCommModel()
    dummy_obs = torch.randn(3, 5)  # batch_size=3

    # 测试动作生成
    actions = model(dummy_obs)
    print(f"生成动作示例:\n{actions}")

    # 测试价值估计
    q_values = model(dummy_obs, actions)
    print(f"价值估计:\n{q_values}")

