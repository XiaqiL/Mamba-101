import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import os

def run_comparison_benchmark():
    os.makedirs("report_assets", exist_ok=True)
    
    # 实验设置
    N_values = [64, 128, 256, 512, 1024, 2048, 4096]
    batch_size = 16
    seq_len = 100
    input_dim = 1
    
    naive_times = []
    fast_times = []
    
    print(f"=== Complexity Benchmark: Naive O(N^2) vs Official DPLR O(N) ===")
    print(f"{'N':<10} | {'Naive (s)':<15} | {'Fast (s)':<15} | {'Speedup':<10}")
    print("-" * 60)
    
    # 1. 预热 (Warm up) - 关键修改！防止第一次运行过慢影响数据
    # 先空跑一次，让 PyTorch 初始化
    print("Warming up...")
    _ = torch.matmul(torch.randn(16, 64), torch.randn(64, 64))
    
    for n in N_values:
        # 准备数据
        inputs = torch.randn(batch_size, seq_len, input_dim)
        c_state = torch.randn(batch_size, n)
        
        # --- 方法 A: Naive (Full Matrix) ---
        A_full = torch.randn(n, n) 
        
        start = time.time()
        with torch.no_grad():
            for t in range(seq_len):
                # O(N^2)
                _ = torch.matmul(c_state, A_full.t())
        end = time.time()
        t_naive = end - start
        naive_times.append(t_naive)
        
        # --- 方法 B: Official Simulation (DPLR) ---
        D = torch.randn(n)
        P = torch.randn(n, 1)
        Q = torch.randn(n, 1)
        
        start = time.time()
        with torch.no_grad():
            for t in range(seq_len):
                # O(N)
                term1 = c_state * D 
                scalar = torch.matmul(c_state, Q) 
                term2 = scalar @ P.t()
                _ = term1 + term2
        end = time.time()
        t_fast = end - start
        fast_times.append(t_fast)
        
        speedup = t_naive / t_fast
        print(f"{n:<10} | {t_naive:.4f}          | {t_fast:.4f}          | {speedup:.1f}x")

    # --- 可视化 (修正版) ---
    plt.figure(figsize=(10, 6))
    
    # 1. 画实际测量数据
    plt.plot(N_values, naive_times, 'o-', color='tab:red', linewidth=2, label='Naive Implementation (Measured)')
    plt.plot(N_values, fast_times, 's-', color='tab:green', linewidth=2, label='Official DPLR (Measured)')
    
    # 2. 画理论曲线 (修正逻辑)
    # 我们用 *最后一个点* (最大 N) 来校准理论曲线
    # 这样能保证理论线和实际线在 N=4096 处重合，从而正确展示之前的趋势
    
    # 校准 O(N^2) 曲线
    scale_quad = naive_times[-1] / (N_values[-1]**2)
    y_quad = [scale_quad * (x**2) for x in N_values]
    plt.plot(N_values, y_quad, '--', color='salmon', alpha=0.5, label='Theoretical $O(N^2)$')
    
    # 校准 O(N) 曲线
    scale_lin = fast_times[-1] / N_values[-1]
    y_lin = [scale_lin * x for x in N_values]
    plt.plot(N_values, y_lin, '--', color='lightgreen', alpha=0.5, label='Theoretical $O(N)$')

    plt.title("Computational Complexity: Full Matrix vs DPLR", fontsize=16)
    plt.xlabel("State Space Size (N)", fontsize=14)
    plt.ylabel("Execution Time (seconds)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # 这里的 ylim 稍微设置一下，防止因为 N=64 的波动导致图不好看
    # 让图聚焦在数据本身
    plt.autoscale(enable=True, axis='y', tight=True)

    save_path = "report_assets/Complexity_Comparison_Fixed.png"
    plt.savefig(save_path, dpi=150)
    print("-" * 60)
    print(f"Comparison graph saved to {save_path}")

if __name__ == "__main__":
    run_comparison_benchmark()