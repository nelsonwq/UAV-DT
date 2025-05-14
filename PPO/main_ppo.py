import os.path
import time

import numpy as np
import torch
from agent_ppo import PPOAgent
from Env_UAV import Environment

current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'
timestamp = time.strftime('%Y%m%d%H%M%S')
if not os.path.exists(model):
    os.mkdir(model)

# scenario = 'Pendulum-v1'
# env = gym.make(scenario)
env = Environment()

NUM_EPISODE = 3000
NUM_STEP = 20
# STATE_DIM = env.observation_space.shape[0]
# ACTION_DIM = env.action_space.shape[0]
STATE_DIM = env.state_dim
ACTION_DIM = env.action_dim
BATCH_SIZE = 25
UPDATE_INTERVAL = 5
best_reward = -2000

REWARD_BUFFER = np.empty(shape=NUM_EPISODE)

agent = PPOAgent(STATE_DIM, ACTION_DIM, BATCH_SIZE)  # TODO

for episode_i in range(NUM_EPISODE):
    state, state_normalization = env.reset()
    done = False
    episode_reward = 0

    for step_i in range(NUM_STEP):
        action, value = agent.get_action(state_normalization)
        next_state, next_state_normalization, reward, done = env.step(state, action)
        episode_reward += reward
        done = True if (step_i + 1) == NUM_STEP else False
        agent.replay_buffer.add_memo(state, action, reward, value, done)
        state = next_state
        state_normalization = next_state_normalization

        if (step_i + 1) % UPDATE_INTERVAL == 0 or (step_i + 1) == NUM_STEP-1:
            agent.update()  # TODO

    if episode_reward >= -100 and episode_reward > best_reward:
        best_reward = episode_reward
        agent.save_policy()  # TODO
        torch.save(agent.actor.state_dict(), model + f'ppo_actor_{timestamp}.pth')
        print(f'Best Reward: {best_reward}')

    REWARD_BUFFER[episode_i] = episode_reward
    print(f'Episode: {episode_i}, Reward: {round(episode_reward, 2)}')
