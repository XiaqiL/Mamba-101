import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from hippo_cell_4_measures import HiPPOCell_4 as HiPPOCell
import os

# 确保输出文件夹存在
os.makedirs("report_assets", exist_ok=True)

def generate_signal(length, type='white_noise'):
    """生成测试信号"""
    if type == 'white_noise':
        return torch.randn(1, length, 1)
    return torch.randn(1, length, 1)

def generate_random_walk(length):
    noise = torch.randn(length)
    signal = torch.cumsum(noise, dim=0)
    signal = signal + 5.0  # 增加偏移量，增加难度
    return signal.view(1, length, 1)

def generate_slow_triangle_wave(length, period):
    """
    生成一个周期极长的三角波。
    LegT 的窗口 (100步) 远小于半周期 (600步)。
    在窗口内，信号看起来就是一条直线 (局部不可区分)，
    只有拥有全局记忆(LegS)才能通过记住很久以前的拐点来定位相位。
    """
    t = torch.arange(length).float()
    # 生成三角波: 绝对值锯齿波的变换
    # 周期归一化 -> 0~1 -> 0~2 -> -1~1
    phase = (t % period) / period
    signal = 2 * torch.abs(2 * (phase - 0.5)) - 1
    
    return signal.view(1, length, 1)
def run_reconstruction_task(N, measure, lag, dt=0.01):
    """
    运行单次重建实验
    Args:
        lag: 我们要重建多少步之前的历史 (Time Lag)
    Returns:
        Y_target: 真实历史值
        Y_pred: 模型预测值
        score: R^2 分数
    """
    print(f"Running: N={N}, Measure={measure}, Lag={lag}...")
    
    # 1. 准备数据
    length = 3000
    period = 600
    inputs = generate_slow_triangle_wave(length, period)
    
    # 2. 初始化模型 (CPU 跑得很快)
    cell = HiPPOCell(N=N, dt=dt, measure=measure)
    
    # 3. 前向传播 (Encoding)
    with torch.no_grad():
        # states shape: (1, Length, N)
        states = cell(inputs)
    
    # 转换数据格式供 sklearn 使用
    X_all = states.squeeze(0).numpy() # (Length, N)
    Y_all = inputs.squeeze(0).numpy() # (Length, 1)
    
    # 4. 构建 Dataset (Shifted Target)
    # 任务: 用 X[t] 预测 Y[t - lag]
    # 所以 X 从 index 'lag' 开始，Y 到 'length - lag' 结束
    X_train = X_all[lag:, :]
    Y_target = Y_all[:-lag, :]
    
    # 5. 训练线性解码器 (Linear Regression)
    # 使用前 80% 数据训练，后 20% 测试
    split = int(len(X_train) * 0.8)
    
    reg = Ridge(alpha=0.01).fit(X_train[:split], Y_target[:split])
    
    # 在测试集上预测
    Y_pred = reg.predict(X_train)
    score = reg.score(X_train[split:], Y_target[split:])
    
    print(f"  -> Test R^2 Score: {score:.4f}")
    return Y_target, Y_pred, score

def plot_comparison(results_dict, title_suffix):
    """画图函数: 专门用于生成 Report 素材"""
    plt.figure(figsize=(14, 5))
    
    # 为了看清细节，我们只画测试集中间的一小段 (200个点)
    plot_len = 2000
    # 取数据末尾的一段
    start_idx = -plot_len 
    
    # 1. 画 Ground Truth (取第一个结果的 target 即可，因为输入是固定的)
    first_key = list(results_dict.keys())[0]
    target = results_dict[first_key][0]
    plt.plot(target[start_idx:], color='black', alpha=0.3, linewidth=3, label='Ground Truth (Past Signal)')
    
    # 2. 画各种配置的预测结果
    styles = ['--', '-.', ':']
    for i, (label, (tgt, pred, _)) in enumerate(results_dict.items()):
        style = styles[i % len(styles)]
        plt.plot(pred[start_idx:], linestyle=style, linewidth=2, label=f'Reconstruction ({label})')
        
    plt.title(f"HiPPO Memory Reconstruction: {title_suffix}", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time Steps (Snapshot)", fontsize=12)
    plt.ylabel("Signal Value", fontsize=12)
    
    save_path = f"report_assets/legtlegslagt.png"
    plt.savefig(save_path, dpi=150)
    print(f"Graph saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    # 固定随机种子，保证每次运行结果一致
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("\n=== Experiment 2: LegS vs LagT (The Measure) ===")
    # 目的: 展示不同测度对同一个 Lag 的表现差异
    # 注意: LagT 对于远期记忆(Lag很大)通常需要更大的 N 或者调节 timescale
    results_M = {}
    # 我们把难度加大: 重建 300 步之前的历史
    lag_test = 2000
    for m in ['legs', 'legt', 'lagt']:
        data = run_reconstruction_task(N=256, measure=m, lag=lag_test)
        results_M[f"{m.upper()}"] = data
    plot_comparison(results_M, "Measure_Comparison")
    
    print("\n✅ All experiments finished. Check 'report_assets' folder!")