import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from hippo_cell import HiPPOCell
import os

# 确保输出目录存在
os.makedirs("report_assets", exist_ok=True)

# --- 1. 定义模型与工具 ---

class LSTMModel(nn.Module):
    """简单的 LSTM 用于对比"""
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        pred = self.linear(out)
        return pred

def train_lstm_quick(train_x, train_y, steps=300, hidden_size=64):
    """快速训练一个 LSTM"""
    model = LSTMModel(hidden_size=hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        output = model(train_x)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()
    model.eval()
    return model

# --- 2. 实验逻辑 ---

def run_robustness_experiment():
    print("=== Robustness Experiment: Denoising Sine Wave ===")
    
    # 参数设置
    seq_len = 1000
    t = torch.linspace(0, 40, seq_len)
    
    # 基础信号：纯净的正弦波
    clean_signal = torch.sin(t).view(1, seq_len, 1)
    
    # 噪声等级
    noise_levels = [0.1, 0.5, 1.0]
    
    # 绘图准备
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    # Lag 设定 (任务：从当前含噪输入中，恢复 lag 步之前的纯净信号)
    # 这里我们做一个较简单的任务：Lag=10 (即去噪并稍作记忆)
    lag = 10 
    
    # 状态空间大小
    N = 128 
    
    for i, noise_std in enumerate(noise_levels):
        print(f"\n--- Testing Noise Level: {noise_std} ---")
        
        # 1. 生成含噪数据
        noise = torch.randn_like(clean_signal) * noise_std
        noisy_input = clean_signal + noise
        
        # 目标：我们希望模型能透过噪声看到本质，恢复 Clean Signal
        # Input: Noisy[t] -> Target: Clean[t-lag]
        
        # 准备数据切片
        X_full = noisy_input
        Y_clean_target = clean_signal
        
        # 训练/测试切分
        # 注意：这里我们简化流程，直接在一段长序列上前80%训练，后20%测试
        # 切片对齐
        # X: [0 ... end]
        # Y: [0 ... end] -> 但实际上我们需要 Y[t-lag]
        # 为了简单，我们让 sklearn 和 LSTM 自己去学这个 mapping
        
        # --- A. HiPPO (Projection) ---
        print("Running HiPPO...")
        cell = HiPPOCell(N=N, dt=0.1, measure='legs') # dt=0.1 对正弦波较好
        with torch.no_grad():
            hippo_states = cell(X_full) # (1, Len, N)
        
        # HiPPO 解码器训练 (Linear Regression)
        # Input: State[t], Target: Clean_Signal[t-lag]
        states_np = hippo_states.squeeze().numpy()
        targets_np = Y_clean_target.squeeze().numpy()
        
        # 对齐: State[lag:] 对应 Target[:-lag]
        X_train_h = states_np[lag:]
        Y_train_h = targets_np[:-lag]
        
        split = int(len(X_train_h) * 0.8)
        reg = LinearRegression().fit(X_train_h[:split], Y_train_h[:split])
        hippo_pred = reg.predict(X_train_h) # 对整个序列预测
        
        hippo_score = reg.score(X_train_h[split:], Y_train_h[split:])
        
        # --- B. LSTM (Gradient Descent) ---
        print("Training LSTM...")
        # 对齐数据用于 LSTM
        # Input: X_full[:, :-lag, :] -> 也就是用 [0..t] 预测 [t+lag]? 不，是用 [t] 预测 [t-lag]
        # 正确做法：LSTM 输入是 noisy_input，目标是 clean_signal (shifted)
        lstm_in = X_full[:, lag:, :]
        lstm_tgt = Y_clean_target[:, :-lag, :]
        
        # 切分训练集
        split_idx = int(lstm_in.shape[1] * 0.8)
        lstm_train_x = lstm_in[:, :split_idx, :]
        lstm_train_y = lstm_tgt[:, :split_idx, :]
        
        lstm_model = train_lstm_quick(lstm_train_x, lstm_train_y, steps=500, hidden_size=N)
        
        with torch.no_grad():
            lstm_pred_tensor = lstm_model(lstm_in)
        lstm_pred = lstm_pred_tensor.squeeze().numpy()
        
        # 计算 LSTM R^2 (在测试集上)
        lstm_test_tgt = lstm_tgt[:, split_idx:, :].squeeze().numpy()
        lstm_test_pred = lstm_pred[split_idx:]
        lstm_res = ((lstm_test_tgt - lstm_test_pred) ** 2).sum()
        lstm_tot = ((lstm_test_tgt - lstm_test_tgt.mean()) ** 2).sum()
        lstm_score = 1 - lstm_res / lstm_tot
        
        print(f"HiPPO R^2: {hippo_score:.4f} | LSTM R^2: {lstm_score:.4f}")
        
        # --- C. 画图 ---
        ax = axes[i]
        # 为了清晰，只画一段测试集
        plot_len = 100
        start = -plot_len
        
        # 1. 画含噪输入 (浅灰色背景)
        noisy_segment = noisy_input.squeeze().numpy()[lag:][start:]
        ax.plot(noisy_segment, color='lightgray', label='Noisy Input', alpha=0.6)
        
        # 2. 画纯净真值 (黑色实线)
        clean_segment = Y_train_h[start:]
        ax.plot(clean_segment, color='black', linewidth=2, label='Clean Truth')
        
        # 3. 画 HiPPO (蓝色)
        hippo_segment = hippo_pred[start:]
        ax.plot(hippo_segment, color='tab:blue', linestyle='--', linewidth=2, label=f'HiPPO ($R^2$={hippo_score:.2f})')
        
        # 4. 画 LSTM (红色)
        lstm_segment = lstm_pred[start:]
        ax.plot(lstm_segment, color='tab:red', linestyle=':', linewidth=2, label=f'LSTM ($R^2$={lstm_score:.2f})')
        
        ax.set_title(f"Noise Level $\sigma={noise_std}$")
        if i == 0:
            ax.legend(loc='upper right', fontsize=9)
            ax.set_ylabel("Signal Value")
        ax.set_xlabel("Time Steps")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Robustness Test: Recovering Clean Signal from Noise", fontsize=16)
    plt.tight_layout()
    save_path = "report_assets/Exp_Robustness_128.png"
    plt.savefig(save_path, dpi=150)
    print(f"\nGraph saved to {save_path}")

if __name__ == "__main__":
    # 固定随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    run_robustness_experiment()