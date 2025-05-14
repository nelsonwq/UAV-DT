import random

import numpy as np
import gym
import torch
import torch.nn.functional as F
from agent_dqn import Agent
from Env_UAV import Environment

# env = gym.make("CartPole-v1", render_mode="human")
# state = env.reset()[0]
env = Environment()

n_episode = 5000
n_time_step = 500
n_state = env.state_dim  # 6
n_action = env.action_dim  # 1

print(f'n_state:{n_state}, n_action:{n_action}')

EPSILON_DECAY = 100000
EPSILON_START = 1.0
EPSILON_END = 0.02
TARGET_UPDATE_FREQUENCY = 10

REWARD_BUFFER = np.empty(shape=n_episode)

agent = Agent(n_input=n_state, n_output=n_action)

state, state_normalization = env.reset()

for episode_i in range(n_episode):
    episode_reward = 0
    for step_i in range(n_time_step):
        epsilon = np.interp(episode_i * n_time_step + step_i, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
        random_sample = random.random()

        if random_sample <= epsilon:
            action = np.random.random()  #[0, 1]
        else:
            action = agent.online_net.act(state_normalization)  # TODO

        next_state, next_state_normalization, reward, done = env.step(state, action)  # TODO
        agent.memo.add_memo(state_normalization, action, reward, next_state_normalization, done)  # TODO
        state = next_state
        state_normalization = next_state_normalization
        episode_reward += reward

        if done:
            state, state_normalization = env.reset()
            REWARD_BUFFER[episode_i] = episode_reward
            break

        batch_state, batch_action, batch_reward, batch_next_state, batch_done = agent.memo.sample()  # TODO

        # Compute targets
        target_q_values = agent.target_net(batch_next_state)  # TODO
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
        targets = batch_reward + agent.GAMMA * (1 - batch_done) * max_target_q_values  # TODO

        # Compute q_values
        q_values = agent.online_net(batch_state)  # TODO
        action_q_values = torch.gather(input=q_values, dim=1, index=batch_action)

        # Compute loss
        loss = F.smooth_l1_loss(targets, action_q_values)

        # Gradient descent
        agent.optimizer.zero_grad()  # TODO
        loss.backward()
        agent.optimizer.step()  # TODO

    if episode_i % TARGET_UPDATE_FREQUENCY == 0:
        agent.target_net.load_state_dict(agent.online_net.state_dict())  # TODO

        # Show the training process
        print('Episode:{}'.format(episode_i))
        print('Avg Reward:{}'.format(np.mean(REWARD_BUFFER[:episode_i])))
