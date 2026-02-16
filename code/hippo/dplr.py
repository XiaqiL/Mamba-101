import torch
import time
import matplotlib.pyplot as plt
import numpy as np

def benchmark_discretization_time():
    # === 1. 实验设置 ===
    # N 的范围：从 64 到 2048 (注意：Naive 方法在 N=4096 时可能会非常慢，建议跑到 2048 即可展示趋势)
    N_values = [64, 128, 256, 512, 1024, 2048] 
    dt = 0.001
    rank = 1
    
    naive_times = []
    dplr_times = []
    
    print(f"{'N':<10} | {'Naive (s)':<15} | {'DPLR (s)':<15} | {'Speedup':<10}")
    print("-" * 60)
    
    # 预热 GPU/CPU
    _ = torch.linalg.inv(torch.randn(64, 64))
    
    for N in N_values:
        # === 数据准备 ===
        # 随机初始化 DPLR 组件 (模拟 HiPPO 矩阵)
        Lambda = torch.randn(N)
        P = torch.randn(N, rank)
        Q = torch.randn(N, rank)
        
        # 构造稠密矩阵 A (用于 Naive 方法)
        A_dense = torch.diag(Lambda) - P @ Q.T
        I = torch.eye(N)
        
        # === 方法 A: Naive Discretization (直接求逆) ===
        # 目标: 计算 (I - dt/2 * A)^-1
        # 复杂度: O(N^3)
        start = time.time()
        
        # 构造要被求逆的矩阵
        denom = I - (dt / 2.0) * A_dense
        # 执行求逆 (这是最耗时的一步)
        _ = torch.linalg.inv(denom)
        
        end = time.time()
        t_naive = end - start
        naive_times.append(t_naive)
        
        # === 方法 B: DPLR Discretization (Woodbury 公式) ===
        # 目标: 利用 Woodbury 公式准备好离散化所需的组件
        # 复杂度: O(N)
        start = time.time()
        
        # 1. 对角部分求逆 (element-wise division) -> O(N)
        step = dt / 2.0
        D_val = 1.0 - step * Lambda
        D_inv = 1.0 / D_val 
        
        # 2. Woodbury 中间小矩阵 (rank x rank) -> O(N) to form, O(1) to invert
        # 我们需要计算: A_small = I + V @ D^-1 @ U
        # U = step * P, V = Q.T
        
        # D_inv @ U (element-wise broadcasting) -> O(N)
        D_inv_U = D_inv.unsqueeze(1) * (step * P) 
        
        # V @ (D_inv_U) (matrix multiplication: 1xN @ Nx1) -> O(N)
        # 结果是一个 1x1 矩阵 (或者 rank x rank)
        A_small = torch.eye(rank) + Q.T @ D_inv_U
        
        # 3. 小矩阵求逆 -> O(1) (因为 rank 很小，通常为 1)
        _ = torch.linalg.inv(A_small)
        
        # 注意：DPLR 方法不需要显式重建 N x N 的 A_bar，
        # 它只需要存下 D_inv 和 A_small_inv 就可以在 O(N) 时间内完成前向传播。
        # 所以这里我们只计时算出这些核心组件的时间。
        
        end = time.time()
        t_dplr = end - start
        dplr_times.append(t_dplr)
        
        speedup = t_naive / t_dplr if t_dplr > 0 else 0
        print(f"{N:<10} | {t_naive:.6f}          | {t_dplr:.6f}          | {speedup:.1f}x")

    # === 3. 绘图 ===
    plt.figure(figsize=(10, 6))
    
    plt.plot(N_values, naive_times, 'o-', color='red', label='Naive Discretization (Inverse) $O(N^3)$')
    plt.plot(N_values, dplr_times, 's-', color='green', label='DPLR Discretization (Woodbury) $O(N)$')
    
    plt.xlabel('State Space Size (N)', fontsize=12)
    plt.ylabel('Computation Time (seconds)', fontsize=12)
    plt.title('Discretization Time: Naive Matrix Inversion vs DPLR Woodbury', fontsize=14)
    plt.yscale('log') # 使用对数坐标轴，因为 O(N^3) 增长太快
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    
    # 标注 Speedup
    plt.annotate(f"{naive_times[-1]/dplr_times[-1]:.0f}x Speedup", 
                 xy=(N_values[-1], naive_times[-1]), 
                 xytext=(N_values[-1]-500, naive_times[-1]),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    benchmark_discretization_time()