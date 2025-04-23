import math
import numpy as np


class Environment:
    def __init__(self):
        self.dcp_num = 20  # data collection point表示工厂中的采集点
        self.uav_num = 4  # UAV个数
        self.vsp_num = 2  # VSP个数
        # self.beta_0 = 10 ** -5  # 1m参考距离的信道增益
        self.H = 100  # UAV的飞行高度
        # self.W = 10 ** 7  # 分配给无人机的带宽
        self.W = 20  # 分配给无人机的带宽
        self.N_0 = 10 ** -7  # UAV处的噪声功率
        self.C_uav = 10 ** 3  # UAV处理1bit数据需要的CPU计算周期数
        self.f_vsp = 3 * (10 ** 9)  # VSP的CPU可用资源
        self.f_ec = 3 * (10 ** 8)  # VSP的CPU可用资源
        self.varsigma = 3 * (10 ** 3)  # 数据处理的复杂度 cycles/bit
        # self.gamma_c = 10  # UAV 采集数据的速率 包/s
        self.max_power_UAV = 5  # UAV最大传输功率5W
        self.max_power_VSP = 15  # VSP最大传输功率15W
        # self.power_off = 0  # VSP 卸载功率
        self.G_uav = 6  # 无人机处发射端的天线增益(dBi) 线性=10**(dBi/10)
        self.G_vsp = 4  # VSP处接收端的天线增益(dBi)
        self.lambda_0 = 0.125  # 信号波长  频段2.4GHz
        self.E_full = 4.5  # 无人机的总电量 千瓦时
        self.sigma = 10 ** -5  # VSP处的加性高斯白噪音
        self.g_0 = 10 ** -5  # VSP 与 EC 之间的信道增益
        self.I = 10 ** -5  # 共享信道干扰
        self.X = 500  # 500m
        self.Y = 500  # 500m
        self.Z = 200  # 200m
        self.state_dim = 6
        self.action_dim = 1

        """
        1.设置采集点的位置，制定飞行轨迹，飞行轨迹经过这些采集点
        2.假设当前UAV数量为4，采集点数量为20，需合理制定该4个UAV的飞行轨迹
        3.UAV在采集点采集数据时，需做出决策：即将数据传输到哪个VSP，以及传输任务百分比，可以是50%传给vsp0，50%传给vsp1
        4.
        """
        self.Q_c = np.array(self.dcp_num)  # 每个采集点对应的数据量Q_c

    def reset(self):
        vsp_location = list()
        dcp_location = list()
        data_volume = list()
        with open('../VSP_location.txt') as f:
            for i in f.readlines():
                x, y, z = [int(j) for j in i.split()]
                vsp_location.append([x, y, z])

        with open('../transmission_points.txt') as f:
            for i in f.readlines():
                x, y, z = [int(j) for j in i.split()]
                dcp_location.append([x, y, z])

        with open('../data_volume1.txt') as f:
            for i in f.readlines():
                q = [float(j) for j in i.split()]
                data_volume.append(q)

        self.dcp_location = np.array(dcp_location)
        self.vsp_location = np.array(vsp_location)
        self.uav_location = self.vsp_location[0]
        self.data_volume = np.array(data_volume)
        # print(self.dcp_location)
        # print(self.vsp_location)
        # print(self.data_volume)

        # return self.dcp_location, self.vsp_location, self.uav_location, self.data_volume
        """
        观测空间包含UAV的位置(uav_num, 3)、传输功率 (uav_num, 1)
        当前无人机个数为 1
        """
        obs = np.empty(shape=(6, ))
        obs[0] = self.uav_location[0]
        obs[1] = self.uav_location[1]
        obs[2] = self.uav_location[2]
        obs[3] = 5  # 传输速率1
        obs[4] = 5  # 传输速率2
        obs[5] = 0  # current_point
        # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        # obs_normalization = min_max_scaler.fit_transform(obs)
        obs_normalization = obs
        obs_normalization[0] = obs[0] / self.X  # (X-X_min)/(X_max-X_min)
        obs_normalization[1] = obs[1] / self.Y  # 500
        obs_normalization[2] = obs[2] / self.Z  # 200
        obs_normalization[3] = obs[3] / 100
        obs_normalization[4] = obs[4] / 100
        return obs, obs_normalization

        # vsp_pos1 = np.array([100, 100, 0])
        # vsp_pos2 = np.array([400, 400, 0])
        # uav_pos1 = np.array([[106, 165, 47], [212, 212, 78], [113, 318, 42], [29, 266, 61], [35, 127, 45]])
        # uav_pos2 = np.array([[106, 60, 73], [199, 27, 67], [217, 104, 67], [334, 166, 72], [423, 122, 58]])
        # uav_pos3 = np.array([[313, 369, 41], [271, 430, 56], [186, 469, 61], [194, 394, 64], [248, 316, 65]])
        # uav_pos4 = np.array([[395, 371, 60], [488, 354, 66], [428, 300, 60], [449, 232, 58], [324, 259, 60]])

    def step(self, obs, action):
        """
        更新动作
        :param obs uav位置，通信速率
        :param action: 任务分配比和功率
        :return:
        """
        done = False
        next_obs = np.empty(shape=(6,))
        # print(obs)
        next_obs[0] = self.dcp_location[int(obs[5]) - 1][0]
        next_obs[1] = self.dcp_location[int(obs[5]) - 1][1]
        next_obs[2] = self.dcp_location[int(obs[5]) - 1][2]

        # 从动作中提取任务分配比例和传输到 vsp1 的功率
        # task_ratio = action[0]

        task_ratio = action
        # power_to_vsp1 = action[1]

        # power_to_vsp2 = self.max_power_UAV - power_to_vsp1
        # power_to_vsp = action[1]
        power_to_vsp = 2
        next_obs[3] = self.get_comm_rate(next_obs[:3], power_to_vsp, self.vsp_location[0])
        next_obs[4] = self.get_comm_rate(next_obs[:3], power_to_vsp, self.vsp_location[1])
        next_obs[5] = int(obs[5]) + 1

        # next_obs[0][3] = np.random.randint(1, 5)  # 功率
        Q = self.data_volume[int(obs[5]) - 1][0]
        comm_delay_first = self.get_comm_delay(Q, task_ratio, next_obs[:3], power_to_vsp, self.vsp_location[0])
        comm_delay_second = self.get_comm_delay(Q, 1 - task_ratio, next_obs[:3], power_to_vsp, self.vsp_location[1])
        delay = -math.fabs(comm_delay_first-comm_delay_second) * 10
        # print(delay, bonus)
        reward = delay

        print(f"uav_location={next_obs[:3]}, q1={Q * action:.6f}, q2={Q * (1 - action):.6f} \
                comm_rate1={next_obs[3]:.6f}, comm_rate2={next_obs[4]:.6f}")
        print(f'task ratio={action:.6f}, uav_power={power_to_vsp:.6f}, comm_delay_first={comm_delay_first:.6f}, \
            comm_delay_second={comm_delay_second:.6f}, reward={reward:.6f}')

        if obs[5] == self.dcp_num:
            done = True

        next_obs_normalization = next_obs
        next_obs_normalization[0] = next_obs[0] / self.X  # (X-X_min)/(X_max-X_min)
        next_obs_normalization[1] = next_obs[1] / self.Y  # 500
        next_obs_normalization[2] = next_obs[2] / self.Z  # 200
        next_obs_normalization[3] = next_obs[3] / 100
        next_obs_normalization[4] = next_obs[4] / 100
        # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        # next_obs_normalization = min_max_scaler.fit_transform(next_obs)
        # print(f'normal={next_obs_normalization}')

        return next_obs, next_obs_normalization, reward, done

    # def get_sense_delay(self, Q):
    #     """
    #     获取采集点所需感知时延
    #     :param Q: 任务量
    #     :return:
    #     """
    #     return Q / self.gamma_c

    def get_dist_m_v(self, uav_pos, vsp_pos):
        """
        # 获取UAV与VSP之间的距离，第二范式
        :param uav_pos: np.ndarray
        :param vsp_pos: np.ndarray
        :return:
        """
        dist = np.linalg.norm(uav_pos - vsp_pos)
        return dist

    def get_comm_rate(self, uav_pos, uav_power, vsp_pos):
        """
        获取UAV通信速率
        :param uav_pos:
        :param uav_power:
        :param vsp_pos:
        :return:
        """
        channel_gain = (self.G_uav * self.G_vsp * self.lambda_0 ** 2) \
                       / ((4 * math.pi) ** 2 * (self.get_dist_m_v(uav_pos, vsp_pos) ** 2))
        Gamma = uav_power * channel_gain / self.N_0
        R = self.W * math.log(1 + Gamma)
        return R

    def get_comm_delay(self, Q, alpha, uav_pos, uav_power, vsp_pos):
        """
        获取通信时延
        :param Q: 任务量
        :param alpha: 任务分配比例
        :param uav_pos: UAV位置
        :param uav_power: UAV功率
        :param vsp_pos: VSP位置
        :return:
        """
        T_comm = alpha * Q / self.get_comm_rate(uav_pos, uav_power, vsp_pos)
        return T_comm

    def get_local_delay(self, Q):
        """
        计算部分任务本地计算的时延
        :param Q:
        :return:
        """
        T_local = (self.varsigma * Q) / self.f_vsp
        return T_local

    def get_off_comm_delay(self, power_off, Q_off):
        """
        计算部分任务卸载所需的传输时延
        :param power_off:
        :param Q_off:
        :return:
        """
        Gamma_ec = (power_off * self.g_0) / (self.I + self.sigma)
        R = self.W * math.log(1 + Gamma_ec)
        T_comm = Q_off / R
        return T_comm

    def get_off_comp_delay(self, Q_off):
        """
        计算部分任务卸载至EC节点处的计算时延
        :param Q_off:
        :return:
        """
        T_off = self.varsigma * Q_off / self.f_ec
        return T_off


if __name__ == '__main__':
    env = Environment()
    # 规划无人机轨迹
    vsp_pos1 = np.array([100, 100, 0])
    vsp_pos2 = np.array([400, 400, 0])
    uav_pos1 = np.array([[106, 165, 47], [212, 212, 78], [113, 318, 42], [29, 266, 61], [35, 127, 45]])
    uav_pos2 = np.array([[106, 60, 73], [199, 27, 67], [217, 104, 67], [334, 166, 72], [423, 122, 58]])
    uav_pos3 = np.array([[313, 369, 41], [271, 430, 56], [186, 469, 61], [194, 394, 64], [248, 316, 65]])
    uav_pos4 = np.array([[395, 371, 60], [488, 354, 66], [428, 300, 60], [449, 232, 58], [324, 259, 60]])

    for i in uav_pos1:
        print(env.get_dist_m_v(i, vsp_pos1))
