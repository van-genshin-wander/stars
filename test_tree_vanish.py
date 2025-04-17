import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
import multiprocessing as mp
import math

class TreeNode:
    def __init__(self, center, size):
        self.center = center
        self.size = size
        self.children = []
        self.mass = 0
        self.com = np.zeros(3)
        self.particle_index = -1

def build_tree(positions, masses, indices, center, size):
    """串行构建八叉树（保持不变）"""
    node = TreeNode(center, size)
    if len(positions) == 0:
        return None
    if len(positions) == 1:
        node.mass = masses[0]
        node.com = positions[0]
        node.particle_index = indices[0]
    else:
        delta = positions - center
        sign_x = (delta[:, 0] >= 0).astype(int)
        sign_y = (delta[:, 1] >= 0).astype(int)
        sign_z = (delta[:, 2] >= 0).astype(int)
        octant = sign_x * 4 + sign_y * 2 + sign_z
        
        for idx in range(8):
            i = 1 if (idx & 4) else -1
            j = 1 if (idx & 2) else -1
            k = 1 if (idx & 1) else -1
            sc = center + size/4 * np.array([i, j, k])
            mask = (octant == idx)
            if np.any(mask):
                child = build_tree(positions[mask], masses[mask], indices[mask], sc, size/2)
                if child is not None:
                    node.children.append(child)
                    node.mass += child.mass
                    node.com += child.mass * child.com
        if node.mass > 0:
            node.com /= node.mass
        else:
            node.com = np.zeros(3)
    return node

def compute_batch(args):
    """计算一批粒子的引力"""
    node, positions, masses, indices, G, theta_max, eps = args
    batch_size = len(indices)
    forces = np.zeros((batch_size, 3))
    for i in range(batch_size):
        idx = indices[i]
        forces[i] = compute_gravity_tree(node, positions[i], masses[i], idx, G, theta_max, eps)
    return indices, forces

def compute_gravity_tree(node, pos, mass, target_index, G=1.0, theta_max=0.5, eps=0.01):
    """计算单个粒子受到的引力（保持不变）"""
    force = np.zeros(3)
    if node is None:
        return force
    if node.particle_index != -1:
        if node.particle_index == target_index:
            return force
        r = node.com - pos
        r_mag = np.sqrt(np.sum(r**2) + eps)
        force += G * mass * node.mass * r / (r_mag**3)
        return force
    r = node.com - pos
    d = np.linalg.norm(r)
    if (node.mass < (theta_max) * d**2) or not node.children:
        # r_mag = d
        # force += G * mass * node.mass * r / (r_mag**3)
        force += 0
    else:
        for child in node.children:
            force += compute_gravity_tree(child, pos, mass, target_index, G, theta_max, eps)
    return force

def compute_gravity_parallel(root, positions, masses, G=1.0, theta_max=0.5, eps=0.01, n_workers=None, batch_size=100):
    """批量并行计算引力"""
    N = len(positions)
    forces = np.zeros((N, 3))
    
    # 计算批次数量
    num_batches = math.ceil(N / batch_size)
    batches = [(i*batch_size, min((i+1)*batch_size, N)) for i in range(num_batches)]
    
    # 准备共享内存
    pos_shm = shared_memory.SharedMemory(create=True, size=positions.nbytes)
    pos_shared = np.ndarray(positions.shape, dtype=positions.dtype, buffer=pos_shm.buf)
    np.copyto(pos_shared, positions)
    
    mass_shm = shared_memory.SharedMemory(create=True, size=masses.nbytes)
    mass_shared = np.ndarray(masses.shape, dtype=masses.dtype, buffer=mass_shm.buf)
    np.copyto(mass_shared, masses)
    
    # 使用进程池并行计算
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # 准备批次任务
        futures = []
        for start, end in batches:
            batch_indices = np.arange(start, end)
            args = (root, 
                   pos_shared[start:end], 
                   mass_shared[start:end], 
                   batch_indices,
                   G, theta_max, eps)
            futures.append(executor.submit(compute_batch, args))
        
        # 收集结果
        for future in as_completed(futures):
            try:
                indices, batch_forces = future.result()
                forces[indices] = batch_forces
            except Exception as e:
                print(f"批次计算出错: {e}")
    
    # 清理共享内存
    pos_shm.close()
    pos_shm.unlink()
    mass_shm.close()
    mass_shm.unlink()
    
    return forces

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    # 测试参数
    np.random.seed(42)
    N = 10000
    positions = np.random.rand(N, 3)
    masses = np.random.uniform(1e-3, 1e-2, N)
    
    print("开始构建八叉树...")
    t1 = time.time()
    root = build_tree(positions, masses, np.arange(N), center=np.array([0.5, 0.5, 0.5]), size=1.0)
    t_tree = time.time()
    print(f"八叉树构建完成，耗时: {t_tree - t1:.2f}秒")
    
    print("开始计算引力(批量并行)...")
    # 自动调整批次大小，确保每个进程至少处理200个粒子
    n_workers = mp.cpu_count()
    min_batch_size = 200
    batch_size = max(min_batch_size, N // (n_workers))
    print(f"使用{n_workers}个工作进程，每批{batch_size}个粒子")
    
    forces_parallel = compute_gravity_parallel(root, positions, masses, n_workers=n_workers, batch_size=batch_size, theta_max=0.2)
    t_parallel = time.time()
    print(f"批量并行引力计算完成，耗时: {t_parallel - t_tree:.2f}秒")
    print(forces_parallel[:5])
    
    # # 对比串行计算
    # print("开始计算引力(串行)...")
    # forces_serial = np.zeros((N, 3))
    # for i in range(N):
    #     forces_serial[i] = compute_gravity_tree(root, positions[i], masses[i], i)
    # t_serial = time.time()
    # print(f"串行引力计算完成，耗时: {t_serial - t_parallel:.2f}秒")
    
    # # 验证结果一致性
    # print("验证结果一致性...")
    # diff = np.max(np.abs(forces_parallel - forces_serial))
    # print(f"最大差异: {diff:.6f} (应该接近0)")
    
    # print("\n性能总结:")
    # print(f"树构建时间: {t_tree - t1:.2f}秒")
    # print(f"批量并行计算时间: {t_parallel - t_tree:.2f}秒")
    # print(f"串行计算时间: {t_serial - t_parallel:.2f}秒")
    # print(f"加速比: {(t_serial - t_parallel)/(t_parallel - t_tree):.2f}x")