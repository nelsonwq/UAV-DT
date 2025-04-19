vsp_location = list()
uav_location = list()
with open('./VSP_location.txt') as f:
    x, y, z = [int(i) for i in f.readline().split()]
    vsp_location.append([x, y, z])
# print(vsp_location)

with open('./transmission_points.txt') as f:
    for i in f.readlines():
        x, y, z = [int(j) for j in i.split()]
        uav_location.append([x, y, z])

# print(uav_location)

import numpy as np
from scipy.stats import gamma, weibull_min


def generate_data_volumes():
    """基于设备重要性的分层数据量生成"""
    np.random.seed(20250410)

    # 1. 关键设备（变压器/断路器/充电站）幂律分布
    heavy = np.clip(gamma.rvs(3, scale=120, size=4), 200, 800)

    # 2. 重要设备（储能/光伏/风电）威布尔分布
    medium = weibull_min.rvs(1.8, scale=40, size=10) + 20

    # 3. 边缘设备（监测站/电表）- 均匀分布
    light = np.random.lognormal(2, 0.4, 6)

    # 组合并排序
    all_data = np.concatenate([heavy, medium, light])
    np.random.shuffle(all_data)
    return np.round(all_data, 1)


data_points = generate_data_volumes()
with open(f'./data_volume.txt', 'w+') as f:
    for i in data_points:
        f.write(f"{i}\n")



