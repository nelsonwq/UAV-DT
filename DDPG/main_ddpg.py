import random

import gym
import numpy as np
import torch
from Env_UAV import Environment
from agent_ddpg import DDPGAgent
import os
import time

# 初始化环境
# env = gym.make(id='Pendulum-v1')
env = Environment()
# STATE_DIM = env.observation_space.shape[0]
# ACTION_DIM = env.action_space.shape[0]
STATE_DIM = env.state_dim
ACTION_DIM = env.action_dim
# print(f'STATE_DIM:{STATE_DIM}, ACTION_DIM:{ACTION_DIM}')
agent = DDPGAgent(STATE_DIM, ACTION_DIM)

# 超参数
NUM_EPISODE = 1000
NUM_STEP = 200
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000

REWARD_BUFFER = np.empty(shape=NUM_EPISODE)

for episode_i in range(NUM_EPISODE):
    state, state_normalization = env.reset()
    # print(f'state:{state}')
    episode_reward = 0
    for step_i in range(NUM_STEP):
        epsilon = np.interp(x=episode_i*NUM_STEP+step_i, xp=[0, EPSILON_DECAY], fp=[EPSILON_START, EPSILON_END])  # 探索开发的阈值
        random_sample = random.random()
        action = np.empty(shape=(2,))
        if random_sample <= epsilon:  # 探索
            action = np.random.uniform(low=0, high=1)
            # action[0] = round(np.random.uniform(low=0, high=1), 6)
            # action[1] = round(np.random.uniform(low=1, high=5), 6)
        else:  # 开发
            action = agent.get_action(state_normalization)[0]
            # action[0] = round(action[0], 6)  # 确保任务分配比例在 [0, 1] 之间
            # action[1] = round(action[1], 6)  # 确保功率在 [1, 5] 之间

        next_state, next_state_normalization, reward, done = env.step(state, action)
        agent.replay_buffer.add_memo(state, action, reward, next_state, done)

        state = next_state
        state_normalization = next_state_normalization
        episode_reward += reward

        agent.update()

        if done:
            break

    REWARD_BUFFER[episode_i] = episode_reward
    print(f'Episode:{episode_i+1}, Reward:{round(episode_reward, 2)}')  # 保留2位小数

current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'
if not os.path.exists(model):
    os.mkdir(model)
timestamp = time.strftime('%Y%m%d%H%M%S')

# Save models
torch.save(agent.actor.state_dict(), model + f'ddpg_actor_{timestamp}.pth')
torch.save(agent.critic.state_dict(), model + f'ddpg_critic_{timestamp}.pth')


