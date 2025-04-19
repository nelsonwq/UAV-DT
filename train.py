import numpy as np
from Env_UAV import Environment
from model import Model
from algorithm import DDPG
from Agent import Agent
from ReplayMemory import ReplayMemory
from UAVCommModel import UAVCommModel

ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
GAMMA = 0.9
TAU = 0.001
MEMORY_SIZE = 200000
MEMORY_WARMUP_SIZE = 2000
BATCH_SIZE = 64
NOISE = 0.05  # 动作噪声方差


def run_episode(agent, env, rpm):
    obs, obs_normalization = env.reset()
    cur_point = 0
    total_loss = 0
    total_reward = 0
    steps = 0
    while True:
        steps += 1
        cur_point += 1
        action = agent.predict(obs.astype('float32'))
        # print(f"action===={action}")
        # action = np.clip(np.random.normal(action, NOISE), 0, 1)
        next_obs, next_obs_normalization, reward, done = env.step(cur_point, obs, action)  # obs是uav位置，action是任务分配比例

        rpm.append((obs_normalization, action, reward, next_obs_normalization, done))

        if len(rpm) > MEMORY_WARMUP_SIZE and (steps % 5) == 0:
            (batch_obs, batch_action, batch_reward, batch_next_obs, batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)
            total_loss += train_loss

        obs = next_obs
        obs_normalization = next_obs_normalization
        total_reward += reward

        if done:
            total_loss = total_loss / steps
            break

    step_sum = steps
    total_reward = total_reward + step_sum
    return total_reward, total_loss, step_sum, obs


def main():
    env = Environment()

    obs_dim = 5  # [x,y,z,r1,r2]
    act_dim = 2  # [任务传输比例, 功率]

    model = UAVCommModel(obs_dim, act_dim)
    algorithm = DDPG(model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = Agent(algorithm, obs_dim, act_dim)
    i = 0
    rpm = ReplayMemory(MEMORY_SIZE)

    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_episode(agent, env, rpm)
        i += 1
        print(f'==========经验池已存入第{i}条数据')

    episode = 0
    reward = []
    loss = []
    max_episode = 5000
    step_num = 50

    while episode < max_episode:
        average_reward = []
        average_loss = []
        total_step = []

        for i in range(step_num):
            total_reward, total_loss, step_sum, obs = run_episode(agent, env, rpm)
            average_reward.append(total_reward)
            average_loss.append(total_loss)
            total_step.append(step_sum)

            if episode == max_episode - 1:
                # 打印
                pass

        average_reward = float(np.mean(average_reward))
        reward.append(average_reward)
        average_loss = float(np.mean(average_loss))
        loss.append(average_loss)

        episode += 1
        print(
            f'episode:{episode}\t\tTrain reward:{average_reward}\t\tTrain loss:{average_loss}\t\ttotal step:{total_step}')


if __name__ == '__main__':
    main()
    # env = Environment()
    # print(env.reset().shape)
