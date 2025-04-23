class HyperParams:
    EPISODES = 30  # 局数
    SHOW_EVERY = 1  # 定义每隔多少局展示一次图像
    EPSILON = 1  # 探索的几率
    MIN_EPSILON = 0.001  # 保留一定的随机率，不能有迹可循
    EPS_DECAY = 0.9998  # 衰减
    DISCOUNT = 0.95  # 折扣因子
    LEARNING_RATE = 0.1  # 学习率
    REPLAY_MEMORY_SIZE = 20  # 经验回放池大小
    MINI_REPLAY_MEMORY_SIZE = 4  # 随机采样批次
    UPDATE_TARGET_MODE_EVERY = 5  # 同步模型参数的频率
    STATICS_EVERY = 1  # 统计频率
    MODEL_SAVE_MIN_REWARD = -500  # 大于最小奖励则保存模型
