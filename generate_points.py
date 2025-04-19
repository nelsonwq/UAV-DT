"""
UAV Transmission Points Generator for Smart Grid Inspection 
Version: 2025.04.01 
Author: DeepSeek Assistant 
Description: Generates 20 transmission points (40-80m height with normal distribution) 
             and 2 VSP points in 500x500m area 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.stats import truncnorm
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常
from matplotlib.colors import Normalize


# ====================== 核心函数 ======================
def generate_poisson_disk_samples(width=500, height=500, min_dist=70, num_samples=20):
    """泊松圆盘采样生成平面坐标"""
    cell_size = min_dist / np.sqrt(2)
    grid_width = int(np.ceil(width / cell_size))
    grid_height = int(np.ceil(height / cell_size))
    grid = np.full((grid_width, grid_height), -1, dtype=int)

    points = []
    active = []

    # 初始点 
    first_point = (np.random.uniform(0, width), np.random.uniform(0, height))
    points.append(first_point)
    active.append(0)
    grid[int(first_point[0] / cell_size), int(first_point[1] / cell_size)] = 0

    while active and len(points) < num_samples:
        idx = np.random.choice(active)
        point = points[idx]
        found = False

        for _ in range(30):
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(min_dist, 2 * min_dist)
            new_x = point[0] + radius * np.cos(angle)
            new_y = point[1] + radius * np.sin(angle)

            if 0 <= new_x < width and 0 <= new_y < height:
                grid_x, grid_y = int(new_x / cell_size), int(new_y / cell_size)
                valid = True

                for i in range(max(0, grid_x - 2), min(grid_width, grid_x + 3)):
                    for j in range(max(0, grid_y - 2), min(grid_height, grid_y + 3)):
                        if grid[i, j] != -1:
                            neighbor = points[grid[i, j]]
                            dist = np.sqrt((new_x - neighbor[0]) ** 2 + (new_y - neighbor[1]) ** 2)
                            if dist < min_dist:
                                valid = False
                                break
                    if not valid:
                        break

                if valid:
                    points.append((new_x, new_y))
                    active.append(len(points) - 1)
                    grid[grid_x, grid_y] = len(points) - 1
                    found = True
                    break

        if not found:
            active.remove(idx)

    return points


def generate_uav_points():
    """生成带高度的传输点（高度40-80m正态分布）"""
    # 平面坐标 
    points_2d = generate_poisson_disk_samples()

    # 高度生成：截断正态分布(μ=60, σ=8, 范围40-80)
    height_dist = truncnorm((40 - 60) / 8, (80 - 60) / 8, loc=60, scale=8)
    heights = height_dist.rvs(size=20).clip(40, 80)

    return [(int(x), int(y), int(h)) for (x, y), h in zip(points_2d, heights)]


def generate_vsp_points():
    """生成VSP坐标"""
    return [(100, 100, 0), (400, 400, 0)]  # 左下和右上角


# ====================== 可视化函数 ======================
def plot_3d_distribution(trans_points, vsp_points):
    """3D空间分布可视化"""
    fig = plt.figure(figsize=(12, 10), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=45)  # 标准工程视角
    ax.set_box_aspect([1, 1, 1])  # 固定xyz比例
    ax.set_ylim(0, 500)  # 强制显示完整y轴范围
    ax.set_zlim(0, 80)  # 突出高度差异
    # 传输点（颜色映射高度）
    tx, ty, tz = zip(*trans_points)
    sc = ax.scatter(tx, ty, tz, c=tz, cmap='viridis', marker='o', s=50, label='传输点')

    # VSP点 
    vx, vy, vz = zip(*vsp_points)
    ax.scatter(vx, vy, vz, c='red', marker='s', s=50, label='VSP')

    # 设置图形属性 
    ax.set_xlabel('X  (m)')
    ax.set_ylabel('Y  (m)')
    ax.set_zlabel('Z  (m)')
    ax.set_title('UAV 传输点空间分布', pad=20)
    ax.legend()
    plt.colorbar(sc, label='高度 (m)')
    plt.tight_layout()

    plt.savefig('transmission point space distribution.png')


# ====================== 可视化函数 ======================
def plot_3d_distribution_1(trans_points, vsp_points):
    """3D空间分布可视化"""
    fig = plt.figure(figsize=(12, 10), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=10, azim=90)  # 标准工程视角
    ax.set_box_aspect([1, 1, 1])  # 固定xyz比例
    ax.set_ylim(0, 500)  # 强制显示完整y轴范围
    ax.set_zlim(0, 80)  # 突出高度差异
    # 传输点（颜色映射高度）
    tx, ty, tz = zip(*trans_points)
    sc = ax.scatter(tx, ty, tz, c=tz, cmap='viridis', marker='o', s=50, label='传输点')

    # VSP点
    vx, vy, vz = zip(*vsp_points)
    ax.scatter(vx, vy, vz, c='red', marker='s', s=50, label='VSP')

    # 设置图形属性
    ax.set_xlabel('X  (m)')
    ax.set_ylabel('Y  (m)')
    ax.set_zlabel('Z  (m)')
    ax.set_title('UAV 传输点空间分布', pad=20)
    ax.legend()
    plt.colorbar(sc, label='高度 (m)')
    plt.tight_layout()

    plt.savefig('transmission point space distribution_1.png')


def plot_3d_distribution_2(trans_points, vsp_points):
    """3D空间分布可视化"""
    fig = plt.figure(figsize=(12, 10), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=10, azim=0)  # 标准工程视角
    ax.set_box_aspect([1, 1, 1])  # 固定xyz比例
    ax.set_ylim(0, 500)  # 强制显示完整y轴范围
    ax.set_zlim(0, 80)  # 突出高度差异
    # 传输点（颜色映射高度）
    tx, ty, tz = zip(*trans_points)
    sc = ax.scatter(tx, ty, tz, c=tz, cmap='viridis', marker='o', s=50, label='传输点')

    # VSP点
    vx, vy, vz = zip(*vsp_points)
    ax.scatter(vx, vy, vz, c='red', marker='s', s=50, label='VSP')

    # 设置图形属性
    ax.set_xlabel('X  (m)')
    ax.set_ylabel('Y  (m)')
    ax.set_zlabel('Z  (m)')
    ax.set_title('UAV 传输点空间分布', pad=20)
    ax.legend()
    plt.colorbar(sc, label='高度 (m)')
    plt.tight_layout()

    plt.savefig('transmission point space distribution_2.png')


def plot_2d_distribution(trans_points, vsp_points):
    """
    2D平面分布可视化（含高度信息）
    更新时间：2025年4月1日
    坐标系：500m×500m平面网格
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 中文字体支持

    # 数据解构
    tx, ty, tz = zip(*trans_points)
    vx, vy, vz = zip(*vsp_points)

    # 绘制传输点（颜色=高度，尺寸=高度比例）
    norm = Normalize(vmin=40, vmax=80)  # 高度归一化
    sc = ax.scatter(tx, ty, c=tz, cmap='coolwarm', s=(np.array(tz) * 2) ** 0.9,
                    edgecolor='k', alpha=0.8, norm=norm, label=f'传输点 (n={len(tx)})')

    # 绘制VSP点
    ax.scatter(vx, vy, c='red', marker='s', s=100, label='垂直起降点', edgecolor='k', zorder=5)

    # 专业图形设置
    ax.set_aspect('equal')  # 1:1比例尺
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_xlabel('X 坐标 (m)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y 坐标 (m)', fontsize=12, labelpad=10)
    ax.set_title(' 智能电网UAV传输点平面分布', fontsize=14, pad=20)

    # 添加图例和色标
    cbar = plt.colorbar(sc, shrink=0.8, pad=0.02)
    cbar.set_label(' 飞行高度 (m)', rotation=270, labelpad=15)
    ax.legend(loc='upper left', framealpha=1)

    # 添加比例尺和方位标识
    ax.plot([400, 450], [30, 30], 'k-', lw=2)  # 50m比例尺
    ax.text(425, 20, '50m', ha='center', fontsize=10)
    # ax.annotate('N', xy=(470, 470), fontsize=12, weight='bold')

    plt.tight_layout()
    plt.savefig('UAV_2D_Distribution.png', bbox_inches='tight', dpi=300)
    plt.show()


def plot_2d_distribution_with_coords(trans_points, vsp_points):
    """带坐标标注的2D平面分布图"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    fig, ax = plt.subplots(figsize=(12, 10), dpi=120)

    # 解构数据
    tx, ty, tz = zip(*trans_points)
    vx, vy, vz = zip(*vsp_points)

    # 绘制散点（缩小点尺寸）
    sc = ax.scatter(tx, ty, c=tz, cmap='coolwarm', s=(np.array(tz) * 2) ** 0.9,
                    edgecolor='k', alpha=0.8, norm=Normalize(40, 80), label=f'传输点 (n={len(tx)})')

    # 添加传输点坐标标签
    for i, (x, y, z) in enumerate(trans_points, 1):
        ax.annotate(
            f"({x:.0f},{y:.0f},{z:.0f})",
            xy=(x, y),
            xytext=(-20, -15),  # 标签偏移量
            textcoords='offset points',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.7),
            arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5)
        )

    # 绘制VSP点（特殊标注）
    ax.scatter(vx, vy, c='red', marker='s', s=150, label='VSP', edgecolor='k')
    for x, y, z in vsp_points:
        ax.annotate(f"VSP\n({x:.0f},{y:.0f},{z:.0f})", (x, y),
                    xytext=(0, -25), ha='center',
                    textcoords='offset points',
                    fontsize=9, weight='bold')

    # 专业图形设置
    ax.set_aspect('equal')  # 1:1比例尺
    ax.set_xlim(-20, 520)
    ax.set_ylim(-20, 520)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_xlabel('X 坐标 (m)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y 坐标 (m)', fontsize=12, labelpad=10)
    ax.set_title(' 智能电网UAV传输点平面分布', fontsize=14, pad=20)

    # 添加图例和色标
    cbar = plt.colorbar(sc, shrink=0.8, pad=0.02)
    cbar.set_label(' 飞行高度 (m)', rotation=270, labelpad=15)
    ax.legend(loc='upper left', framealpha=1)

    # 添加比例尺和方位标识
    # ax.plot([400, 450], [30, 30], 'k-', lw=2)  # 50m比例尺
    # ax.text(425, 20, '50m', ha='center', fontsize=10)
    # ax.annotate('N', xy=(470, 470), fontsize=12, weight='bold')

    plt.tight_layout()
    plt.savefig('UAV_2D_Map_Annotated.png', dpi=300)
    plt.show()


def plot_height_distribution(trans_points):
    """高度分布直方图"""
    heights = [p[2] for p in trans_points]

    plt.figure(figsize=(10, 5), dpi=120)
    plt.hist(heights, bins=12, color='skyblue', edgecolor='k', alpha=0.7)
    plt.axvline(np.mean(heights), color='r', linestyle='dashed',
                linewidth=2, label=f'平均高度: {np.mean(heights):.1f}m')
    plt.xlabel(' 高度 (m)')
    plt.ylabel(' 频数')
    plt.title(' 传输点高度分布 (正态分布 μ=60m, σ=8m)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('Transmission point height distribution.png')


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


# ====================== 主程序 ======================
if __name__ == "__main__":
    # 生成数据
    uav_points = generate_uav_points()


    file = './UAV_points.txt'

    with open(file, 'w+') as f:
        for (x, y, z) in uav_points:
            f.write(f"{x} {y} {z}\n")

    vsp_points = generate_vsp_points()
    with open('./VSP_location.txt', 'w+') as f:
        for (x, y, z) in vsp_points:
            f.write(f"{x} {y} {z}\n")

    # 打印结果
    print("=== UAV传输点坐标 ===")
    for i, (x, y, z) in enumerate(uav_points, 1):
        print(f"点{i:02d}: X={x:3d}m, Y={y:3d}m, 高度={z:3d}m")

    print("\n=== VSP坐标 ===")
    for i, (x, y, z) in enumerate(vsp_points, 1):
        print(f"VSP{i}: X={x:3d}m, Y={y:3d}m, 高度={z:3d}m")

    # 可视化
    plot_3d_distribution(uav_points, vsp_points)
    plot_3d_distribution_1(uav_points, vsp_points)
    plot_3d_distribution_2(uav_points, vsp_points)
    plot_height_distribution(uav_points)
    plot_2d_distribution(uav_points, vsp_points)
    plot_2d_distribution_with_coords(uav_points, vsp_points)
    plt.show()
