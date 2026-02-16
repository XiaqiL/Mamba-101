import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from hippo_cell import HiPPOCell
import os

def run_benchmark():
    # 确保输出目录存在
    os.makedirs("report_assets", exist_ok=True)
    
    # 实验参数设置
    # N 的范围：从 64 到 2048 (注意：2048 在 CPU 上可能会明显变慢)
    N_values = [64, 128, 256, 512, 1024, 2048]
    
    # 固定输入规模
    batch_size = 16
    seq_len = 1000  # 序列足够长，才能让计算时间显著
    input_dim = 1
    
    print(f"=== HiPPO Efficiency Benchmark (CPU) ===")
    print(f"Batch Size: {batch_size}, Sequence Length: {seq_len}")
    print("-" * 40)
    print(f"{'N (State Size)':<15} | {'Time (seconds)':<15} | {'Relative Increase':<20}")
    print("-" * 40)
    
    times = []
    
    # 1. 预热 (Warm-up)
    # PyTorch 第一次运行通常会有初始化开销，先跑一次不计入时间
    _ = HiPPOCell(N=64, dt=1.0)
    
    base_time = None
    
    for n in N_values:
        # 初始化模型
        cell = HiPPOCell(N=n, dt=0.01)
        inputs = torch.randn(batch_size, seq_len, input_dim)
        
        # 计时开始
        start_time = time.time()
        with torch.no_grad():
            _ = cell(inputs)
        end_time = time.time()
        
        elapsed = end_time - start_time
        times.append(elapsed)
        
        # 计算相对增长倍数
        if base_time is None:
            base_time = elapsed
            rel = 1.0
        else:
            rel = elapsed / base_time
            
        print(f"{n:<15} | {elapsed:.4f} s        | {rel:.2f}x")

    # 2. 数据可视化
    plt.figure(figsize=(8, 6))
    
    # 画实际数据点
    plt.plot(N_values, times, 'o-', linewidth=2, color='purple', label='Measured Time')
    
    # 3. 拟合理论曲线 O(N^2)
    # 我们尝试用二次多项式拟合数据，看看是不是完美的抛物线
    # y = a*x^2 + b*x + c
    coeffs = np.polyfit(N_values, times, 2)
    poly = np.poly1d(coeffs)
    
    x_fit = np.linspace(min(N_values), max(N_values), 100)
    y_fit = poly(x_fit)
    
    plt.plot(x_fit, y_fit, '--', color='gray', alpha=0.7, label='Quadratic Fit ($O(N^2)$)')
    
    plt.title("HiPPO Computational Complexity Analysis", fontsize=14)
    plt.xlabel("State Space Size (N)", fontsize=12)
    plt.ylabel("Inference Time (seconds)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    save_path = "report_assets/Benchmark_Time_vs_N.png"
    plt.savefig(save_path, dpi=150)
    print("-" * 40)
    print(f"Graph saved to {save_path}")

if __name__ == "__main__":
    run_benchmark()