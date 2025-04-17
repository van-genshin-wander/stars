import numpy as np
import time

np.random.seed(42)
N = 1000

# 生成随机位置和质量
points = np.random.rand(N, 3)
masses = np.random.uniform(1e-3, 1e-2, N)  # 质量范围 [0.001, 0.01]

t1 = time.time()

# 计算位置差（形状 (N, N, 3)）
diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]

# 计算距离的立方（形状 (N, N)）
dist_sq = np.sum(diff**2, axis=2)
dist_cubed = dist_sq ** 1.5
dist_cubed[dist_cubed == 0] = np.inf  # 避免自身引力（除零）

# 计算引力向量（形状 (N, N, 3)）
forces = diff / dist_cubed[:, :, np.newaxis]

# 乘以质量因素（m_i * m_j）
mass_factor = masses[:, np.newaxis] * masses[np.newaxis, :]  # 形状 (N, N)
forces *= mass_factor[:, :, np.newaxis]

# 对每个点求和所有其他点对它的引力（形状 (N, 3)）
total_forces = np.sum(forces, axis=1)

t2 = time.time()
print(total_forces[:5])
print("计算时间:", t2 - t1)