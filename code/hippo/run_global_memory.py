import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from hippo_cell_4_measures import HiPPOCell_4 as HiPPOCell
import os

os.makedirs("report_assets", exist_ok=True)

# === 1. 数据生成器: 随机游走 ===
def generate_random_walk(length=2000):
    noise = torch.randn(length)
    signal = torch.cumsum(noise, dim=0)
    signal = signal + 5.0  # 增加偏移量，增加难度
    return signal.view(1, length, 1)

# === 2. 实验主逻辑 ===
def run_random_walk_challenge():
    print("\n=== Experiment: Global Mean on Random Walk (Infinite Memory Test) ===")
    
    # 参数设置
    N = 256
    dt = 0.01
    L = 2000
    train_steps = 400  # 前 20% 用于训练
    
    # 1. 构造输入
    inputs = generate_random_walk(L)
    inputs_np = inputs.squeeze().numpy()
    
    # 2. 构造目标: 累积平均值 (Global Running Mean)
    steps = np.arange(1, L + 1)
    targets = np.cumsum(inputs_np) / steps
    targets = targets.reshape(-1, 1)
    
    measures = ['legs', 'legt'] 
    results = {}
    scores = {}
    
    # 3. 运行模型
    for m in measures:
        print(f"Testing {m.upper()}...")
        cell = HiPPOCell(N=N, dt=dt, measure=m)
        with torch.no_grad():
            states = cell(inputs).squeeze(0).numpy()
            
        # 4. 训练解码器
        reg = Ridge(alpha=0.01) 
        reg.fit(states[:train_steps], targets[:train_steps])
        
        # 5. 预测与评估
        pred = reg.predict(states)
        
        # 关键: 计算测试集 (泛化段) 的 R^2
        # score() 方法默认返回 R^2
        test_score = reg.score(states[train_steps:], targets[train_steps:])
        
        results[m] = pred
        scores[m] = test_score
        
        print(f"  -> {m.upper()} Test R^2: {test_score:.4f}")

    # 4. 画图
    plt.figure(figsize=(10, 6))
    
    # 画 Ground Truth
    plt.plot(targets, color='black', linewidth=3, alpha=0.3, label='Target: Global Mean')
    
    # 画 LegS (把 R^2 放进图例)
    plt.plot(results['legs'], color='blue', linewidth=2, 
             label=f'LegS (Infinite Memory) $R^2={scores["legs"]:.2f}$')
    
    # 画 LegT (把 R^2 放进图例)
    # LegT 的 R^2 可能会是很大的负数，我们限制一下显示格式以免丑陋
    legt_score_str = f"{scores['legt']:.2f}" if scores['legt'] > -10 else " < -10"
    plt.plot(results['legt'], color='red', linestyle='--', linewidth=2, 
             label=f'LegT (Finite Window) $R^2={legt_score_str}$')
    
    # 画训练集分界线
    plt.axvline(x=train_steps, color='green', linestyle=':', label='Training End')
    
    plt.title("Random Walk Test: Generalization beyond the Window", fontsize=14)
    plt.xlabel("Time Steps (t)")
    plt.ylabel("Cumulative Average")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "report_assets/Exp_Random_Walk_with_Metric.png"
    plt.savefig(save_path, dpi=150)
    print(f"Graph saved to {save_path}")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    run_random_walk_challenge()