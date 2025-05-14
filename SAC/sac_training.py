import os.path
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sac_agent import SACAgent
from Env_UAV import Environment

env = Environment()
STATE_DIM = env.state_dim
ACTION_DIM = env.action_dim
MEMORY_SIZE = 1000000

agent = SACAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, memo_capacity=MEMORY_SIZE,
                 alpha=1e-5, beta=1e-5, gamma=0.98, tau=0.001, layer1_dim=256, layer2_dim=256, batch_size=128)  # TODO

PLOT_REWARD = True

NUM_EPISODE = 200
NUM_STEP = 25
REWARD_BUFFER = []
best_reward = -np.inf

print(best_reward)
# 保存模型地址
current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'

if not os.path.exists(model):
    os.mkdir(model)

timestamp = time.strftime('%Y%m%d%H%M%S')


for episode_i in range(NUM_EPISODE):
    state, state_normalization = env.reset()
    episode_reward = 0

    for step_i in range(NUM_STEP):
        action = agent.get_action(state_normalization)
        next_state, next_state_normalization, reward, done = env.step(state, action)
        agent.add_memo(state, action, reward, next_state, done)
        episode_reward += reward
        state = next_state
        agent.update()

        if done:
            break
    REWARD_BUFFER.append(episode_reward)
    avg_reward = np.mean(REWARD_BUFFER)

    # 保存模型
    if avg_reward > best_reward:
        best_reward = avg_reward
        torch.save(agent.actor.state_dict(), model + f'sac_actor_{timestamp}.pth')
        print(f'...保存最优奖励的模型：{best_reward}')

    print(f'Episode {episode_i}', 'reward %.1f' % episode_reward, 'avg_reward: {:.1f}'.format(avg_reward))


if PLOT_REWARD:
    plt.plot(np.arange(len(REWARD_BUFFER)), REWARD_BUFFER, color='purple', alpha=0.5, label='Reward')
    plt.plot(np.arange(len(REWARD_BUFFER)), gaussian_filter1d(REWARD_BUFFER, sigma=5), color='green', linewidth=2)
    plt.title('Reward')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.savefig(f'Reward-{timestamp}.png', format='png')
    plt.show()