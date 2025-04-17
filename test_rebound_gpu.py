import rebound
import numpy as np
import matplotlib.pyplot as plt
import tqdm
# 创建模拟系统
sim = rebound.Simulation()
sim.integrator = "ias15"  # 高精度积分器，适合有非引力力的情况
sim.G = 1.0               # 引力常数设为1，方便计算
sim.dt = 0.01             # 初始时间步长

# 添加中心恒星
sim.add(m=1.0)  # 质量为1的中心恒星

# 添加尘埃粒子
N_particles = 1000
for i in range(N_particles):
    # 在1.0到1.5的半径范围内随机分布
    a = 1.0 + 0.5 * np.random.rand()  # 半长轴
    e = 0.01 * np.random.rand()       # 低偏心率
    inc = 0.01 * np.random.rand()     # 低倾角
    
    # 随机其他轨道参数
    Omega = 2.*np.pi*np.random.rand()
    omega = 2.*np.pi*np.random.rand()
    f = 2.*np.pi*np.random.rand()
    
    sim.add(m=1e-5, a=a, e=e, inc=inc, Omega=Omega, omega=omega, f=f)
N = 20
# 运行模拟
times = np.linspace(0, N, N)
x = np.zeros((len(times), N_particles))
y = np.zeros((len(times), N_particles))

for i, t in enumerate(times):
    print(t)
    sim.integrate(t)
    for j in range(N_particles):
        x[i,j] = sim.particles[j+1].x  # j+1因为第一个是恒星
        y[i,j] = sim.particles[j+1].y
op = rebound.OrbitPlot(sim, lw=0.01)  # 2D轨道图
plt.show()
# op.draw()


# 绘制结果
# print('helo')
# plt.figure(figsize=(10,10))
# plt.scatter(x[-1,:], y[-1,:], s=1)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Dust Ring Simulation")
# plt.axis('equal')
# plt.show()