import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import time
from tqdm import tqdm

def read_trajectories(filename, delta):
    """
    读取轨迹文件，返回时间点和天体数据
    """
    bodies_data = []
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            if not i % delta == 0: continue
            bodies = []
            for body_str in line.strip().split(';'):
                if body_str:
                    try:
                        x, y, z, r = map(float, body_str.split(','))
                        pos = np.array([x, y, z])
                        if np.linalg.norm(pos) > 3 * 1e4: continue
                        bodies.append({'position': pos, 'radius': r})
                    except:
                        print(f"警告: 跳过格式错误的数据: {body_str}")
                        continue
            bodies_data.append(bodies)
    return bodies_data

def generate_gif(bodies_data, output_filename='trajectories.gif', fps=20, dpi=100):
    """
    直接生成GIF，不显示图形窗口
    """
    # 创建3D图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置坐标轴标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('天体运动轨迹可视化')

    # 计算坐标轴范围
    all_positions = np.array([body['position'] for bodies in bodies_data for body in bodies])
    if len(all_positions) > 0:
        min_pos, max_pos = all_positions.min(axis=0), all_positions.max(axis=0)
        center = (min_pos + max_pos) / 2
        max_range = max(max_pos - min_pos) / 2 * 1.1
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)

    # 初始化散点图
    num_bodies = len(bodies_data[0]) if bodies_data else 0
    colors = plt.cm.rainbow(np.linspace(0, 1, num_bodies))
    scatters = [ax.scatter([], [], [], color=colors[i], s=50, alpha=0.8) for i in range(num_bodies)]

    def init():
        for scatter in scatters:
            scatter._offsets3d = ([], [], [])
        return scatters

    def update(frame):
        for scatter in scatters:
            scatter._offsets3d = ([], [], [])
        
        bodies = bodies_data[frame]
        for i, body in enumerate(bodies):
            pos = body['position']
            radius = body['radius']
            scatters[i]._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
            scatters[i].set_sizes([radius * 5])
        
        ax.set_title(f'Frame: {frame+1}/{len(bodies_data)}')
        return scatters

    # 创建进度条包装器
    def update_with_progress(frame):
        update(frame)
        pbar.update(1)
        return scatters

    # 生成动画（不显示窗口）
    with tqdm(total=len(bodies_data), desc="生成GIF进度") as pbar:
        ani = FuncAnimation(
            fig, update_with_progress, frames=len(bodies_data),
            init_func=init, blit=False
        )
        
        # 直接保存GIF
        ani.save(output_filename, writer='pillow', fps=fps, dpi=dpi)
    plt.close()  # 关闭图形释放内存
    print(f"GIF已保存至: {output_filename}")

if __name__ == "__main__":
    t1 = time.time()
    # 读取数据
    filename = "trajectories.txt"
    bodies_data = read_trajectories(filename, delta=1000)[:500]
    
    # 检查数据
    print(f"总帧数: {len(bodies_data)}")
    if bodies_data:
        print(f"每帧天体数: {len(bodies_data[0])}")

    # 采样数据（可选）
    # bodies_data = [bodies_data[100 * k] for k in range(10000)]  # 示例采样
    
    # 直接生成GIF
    generate_gif(bodies_data, output_filename='trajectories.gif', fps=20, dpi=100)
    print(time.time() - t1)