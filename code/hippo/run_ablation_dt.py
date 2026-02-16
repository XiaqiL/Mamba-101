import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from hippo_cell import HiPPOCell
import os

# 确保输出文件夹存在
os.makedirs("report_assets", exist_ok=True)

# === 1. 保持你的信号生成函数不变 ===
def generate_signal(length=2000, type='mixed_sine'):
    t = torch.linspace(0, 100, length)
    if type == 'white_noise':
        return torch.randn(1, length, 1)
    elif type == 'step':
        signal = torch.zeros(length)
        num_jumps = length // 200
        for i in range(num_jumps):
            val = np.random.choice([-1, 0, 1])
            signal[i*200:] = val
        return signal.view(1, length, 1)
    elif type == 'sinusoid':
        signal = torch.sin(t)
        return signal.view(1, length, 1)
    elif type == 'mixed_sine':
        # 推荐用这个测 dt，因为它有快有慢，能看出不同 dt 的频响特性
        signal = torch.sin(t) + 0.5 * torch.sin(3*t) + 0.3 * torch.sin(7*t)
        return signal.view(1, length, 1)
    else:
        raise ValueError(f"Unknown signal type: {type}")

# === 2. 保持你的绘图函数不变 (非常好用) ===
def plot_comparison(results_dict, title_suffix):
    num_plots = len(results_dict)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5), sharey=True) # sharey=True 方便对比幅度
    if num_plots == 1: axes = [axes]
        
    plot_len = 200
    start_idx = -plot_len 
    
    for i, (label, (tgt, pred, score)) in enumerate(results_dict.items()):
        ax = axes[i]
        ax.plot(tgt[start_idx:], color='gray', alpha=0.5, linewidth=3, label='Ground Truth')
        
        # 根据 R^2 变色
        line_color = 'tab:blue' if score > 0.8 else 'tab:red'
        ax.plot(pred[start_idx:], color=line_color, linestyle='--', linewidth=2, label='Reconstruction')
        
        # 格式化标题
        r2_text = f"{score:.4f}" if score > -10 else f"{score:.1e}"
        ax.set_title(f"{label}\n$R^2 = {r2_text}$", fontsize=14, fontweight='bold')
        
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time Steps")
        if i == 0: ax.set_ylabel("Signal Value")
            
    fig.suptitle(f"HiPPO Ablation Study: {title_suffix}", fontsize=16, y=1.05)
    plt.tight_layout()
    
    save_path = f"report_assets/Exp_{title_suffix}_{fixed_signal}_{fixed_N}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Graph saved to {save_path}")
    plt.close()

# === 3. 主程序：修改为测试 dt ===
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("\n=== Experiment: Effect of Discretization Step Size (dt) ===")
    
    # --- 实验设置 ---
    fixed_N = 256           # 固定 N，控制变量
    fixed_signal = 'white_noise' # 固定信号类型
    fixed_lag = 100         # 固定滞后步数
    
    # 我们要测试的 dt 列表
    # 理论预期：
    # 1.0 -> 太大，失忆，变成 Passthrough (只输出当前值)
    # 0.01 -> 最佳，R^2 最高，重建完美
    # 0.0001 -> 太小，冻结，状态推不动
    dt_list = [1.0, 0.1, 0.01, 0.001]
    
    results_dt = {}
    
    for dt in dt_list:
        print(f"Running dt={dt}...")
        
        # 1. 生成数据
        inputs = generate_signal(length=3000, type=fixed_signal)
        
        # 2. 初始化模型 (传入当前的 dt)
        cell = HiPPOCell(N=fixed_N, dt=dt, measure='legs')
        
        # 3. 前向传播
        with torch.no_grad():
            states = cell(inputs)
        
        # 4. 线性回归解码
        X_all = states.squeeze(0).numpy()
        Y_all = inputs.squeeze(0).numpy()
        
        # 构建 Lag 任务
        X_train = X_all[fixed_lag:, :]
        Y_target = Y_all[:-fixed_lag, :]
        
        split = int(len(X_train) * 0.8)
        reg = Ridge().fit(X_train[:split], Y_target[:split])
        Y_pred = reg.predict(X_train)
        score = reg.score(X_train[split:], Y_target[split:])
        
        # 存结果，Key 是 dt 的值
        results_dt[f"dt={dt}"] = (Y_target, Y_pred, score)
        print(f"  -> R^2: {score:.4f}")

    # 画图
    plot_comparison(results_dt, "Step_Size_Comparison_ridge")