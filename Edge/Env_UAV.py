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
        self.f_ec = 3 * (10 ** 8)  # EC的CPU可用资源s
        self.varsigma = 3 * (10 ** 3)  # 数据处理的复杂度 cycles/bit
        # self.gamma_c = 10  # UAV 采集数据的速率 包/s
        self.max_power_UAV = 5  # UAV最大传输功率5W
        self.max_power_VSP = 15  # VSP最大传输功率15W
        self.power_off = 3  # VSP 卸载功率
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
        self.state_dim = 2
        self.action_dim = 2

        """
        1.设置采集点的位置，制定飞行轨迹，飞行轨迹经过这些采集点
        2.假设当前UAV数量为4，采集点数量为20，需合理制定该4个UAV的飞行轨迹
        3.UAV在采集点采集数据时，需做出决策：即将数据传输到哪个VSP，以及传输任务百分比，可以是50%传给vsp0，50%传给vsp1
        4.
        """
        self.Q_c = np.array(self.dcp_num)  # 每个采集点对应的数据量Q_c

    def reset(self):
        Q = 100
        A = 1
        obs = np.empty(shape=(2,))
        obs[0] = Q
        obs[1] = A

        return obs

    def step(self, state, action):
        """
        :param obs: obs[0]=Q, obs[1]=1
        :param action: [0]=task_ratio, [1]=ec1, [2]=ec2, [3]=ec3
        :return:
        """
        done = False
        local_ratio = action[0]
        Q_local = local_ratio * state[0]
        Q_off = (1 - local_ratio) * state[0]
        power_off = action[1] # 3W action[1]
        t_local = self.get_local_delay(Q_local)
        t_comm = self.get_off_comm_delay(power_off, Q_off)
        t_off = self.get_off_comp_delay(Q_off)
        t_edge = t_comm + t_off

        t_comp = max(t_local, t_edge)
        # print(f'Q_local:{Q_local}, Q_off:{Q_off}')
        # print(f't_local:{t_local}, t_comm:{t_comm}, t_off:{t_off}, t_edge:{t_edge}')
        t_req = 1e-3
        penalty = 0
        if t_comp < t_req:
            penalty = -100

        reward = -t_edge + penalty

        next_state = state

        return next_state, reward, done

    # def get_sense_delay(self, Q):
    #     """
    #     获取采集点所需感知时延
    #     :param Q: 任务量
    #     :return:
    #     """
    #     return Q / self.gamma_c

    # def get_dist_m_v(self, uav_pos, vsp_pos):
    #     """
    #     # 获取UAV与VSP之间的距离，第二范式
    #     :param uav_pos: np.ndarray
    #     :param vsp_pos: np.ndarray
    #     :return:
    #     """
    #     dist = np.linalg.norm(uav_pos - vsp_pos)
    #     return dist

    # def get_comm_rate(self, uav_pos, uav_power, vsp_pos):
    #     """
    #     获取UAV通信速率
    #     :param uav_pos:
    #     :param uav_power:
    #     :param vsp_pos:
    #     :return:
    #     """
    #     channel_gain = (self.G_uav * self.G_vsp * self.lambda_0 ** 2) \
    #                    / ((4 * math.pi) ** 2 * (self.get_dist_m_v(uav_pos, vsp_pos) ** 2))
    #     Gamma = uav_power * channel_gain / self.N_0
    #     R = self.W * math.log(1 + Gamma)
    #     return R

    # def get_comm_delay(self, Q, alpha, uav_pos, uav_power, vsp_pos):
    #     """
    #     获取通信时延
    #     :param Q: 任务量
    #     :param alpha: 任务分配比例
    #     :param uav_pos: UAV位置
    #     :param uav_power: UAV功率
    #     :param vsp_pos: VSP位置
    #     :return:
    #     """
    #     T_comm = alpha * Q / self.get_comm_rate(uav_pos, uav_power, vsp_pos)
    #     return T_comm

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
    obs = env.reset()
    action = [0.7, 3]
    reward = env.step(obs, action)
    print(obs)
    print(reward)
