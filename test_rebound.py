import rebound
import numpy as np
from tqdm import tqdm  # 进度条

# 设置模拟参数
N = 10**4  # 粒子数
box_size = 1e3  # 模拟区域大小（单位：AU）
softening = 1e-4  # 软化长度（避免奇点）

# 初始化模拟
sim = rebound.Simulation()
sim.integrator = "leapfrog"  # 高效积分器
sim.dt = 1e-3  # 时间步长（需调整）
sim.softening = softening  # 软化长度
sim.boundary = "open"  # 无周期性边界
sim.gravity = "tree"  # 使用 Barnes-Hut 树算法
sim.tree_update_frequency = 0  # 每步更新树结构（可调）
sim.configure_box(box_size)

# 启用 GPU 加速（若支持）
# if rebound.clibrebound and hasattr(rebound.clibrebound, "gpucode"):
sim.gpu = 1  # 启用 GPU
if hasattr(sim, "gpu_enabled"):
    print("GPU is enabled:", sim.gpu_enabled)
else:
    print("GPU not available (REBOUND not compiled with CUDA support)")

# 随机生成粒子（并行优化）
np.random.seed(42)
masses = np.random.uniform(1e-3, 1e-2, N)  # 随机质量
positions = np.random.uniform(-box_size/2, box_size/2, (N, 3))  # 随机位置

# 批量添加粒子（比循环更快）
for i in tqdm(range(N), desc="Adding particles"):
    sim.add(
        m=masses[i],
        x=positions[i, 0],
        y=positions[i, 1],
        z=positions[i, 2],
        # hash=i  # 将 hash 设为字符串（或改用 int(i)）
    )

# 设置 Barnes-Hut 参数
sim.tree_theta = 0.5  # 近似参数（0=精确，1=最快）
sim.tree_max_level = 20  # 树最大深度

# # 运行模拟（N 步）
# n_steps = 100
# for _ in tqdm(range(n_steps), desc="Simulating"):
#     sim.step()

for t in tqdm(range(10)):
    sim.integrate(10)

# 输出结果（可选）
# print(f"Simulated {N} particles for {n_steps} steps.")
# print(f"Final time: {sim.t}")