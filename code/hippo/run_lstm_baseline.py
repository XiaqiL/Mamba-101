import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# 复用之前的生成逻辑，保证公平
def generate_signal(length=2000):
    return torch.randn(1, length, 1)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (Batch, Seq, Input)
        out, _ = self.lstm(x)
        # out: (Batch, Seq, Hidden)
        pred = self.linear(out)
        return pred

def train_lstm(lag, hidden_size=64, steps=500):
    print(f"--- Training LSTM Baseline (Lag={lag}, N={hidden_size}) ---")
    
    # 1. 准备数据
    # 为了让 LSTM 学会，我们需要多一点数据，或者在一个长序列上多跑几轮
    T_train = 3000
    T_test = 1000
    
    # 生成训练数据
    inputs = generate_signal(T_train) # (1, T, 1)
    # 目标是重建 lag 步之前的输入
    # Input: u[t], Target: u[t-lag]
    # 我们需要对齐数据：
    # Train Input: inputs[:, lag:, :]
    # Train Target: inputs[:, :-lag, :]
    
    train_x = inputs[:, lag:, :]
    train_y = inputs[:, :-lag, :]
    
    # 2. 初始化模型
    model = LSTMModel(hidden_size=hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # 3. 训练循环 (CPU 也能跑得动)
    losses = []
    model.train()
    for i in range(steps):
        optimizer.zero_grad()
        output = model(train_x)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if (i+1) % 100 == 0:
            print(f"Step {i+1}/{steps}, Loss: {loss.item():.6f}")
            
    # 4. 测试阶段
    model.eval()
    test_inputs = generate_signal(T_test)
    test_x = test_inputs[:, lag:, :]
    test_y = test_inputs[:, :-lag, :]
    
    with torch.no_grad():
        pred = model(test_x)
        
    # 计算 R^2
    target_var = torch.var(test_y)
    mse = criterion(pred, test_y)
    r2 = 1 - mse / target_var
    
    print(f"-> LSTM Test R^2: {r2.item():.4f}")
    
    return test_y.squeeze().numpy(), pred.squeeze().numpy(), r2.item(), losses

def plot_comparison(target, pred, r2, lag):
    plt.figure(figsize=(10, 4))
    start = 0
    end = 200
    
    plt.plot(target[start:end], label='Ground Truth', color='gray', alpha=0.5, linewidth=2)
    plt.plot(pred[start:end], label=f'LSTM (R2={r2:.2f})', color='red', linestyle='--')
    
    plt.title(f"LSTM Reconstruction Task (Lag={lag})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("report_assets", exist_ok=True)
    plt.savefig(f"report_assets/LSTM_Lag{lag}.png")
    print("Plot saved.")

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 实验设置：跟 HiPPO 保持一致
    # 比如之前 N=256, Lag=300 失败了，我们看看 LSTM 行不行
    # 或者试一个简单的 Lag=50, N=64
    
    # 1. 简单任务：Lag=50 (LSTM 应该能学会)
    target, pred, r2, _ = train_lstm(lag=50, hidden_size=64, steps=500)
    plot_comparison(target, pred, r2, 50)
    
    # 2. 困难任务：Lag=300 (HiPPO 曾在此挣扎，LSTM 可能会崩得更惨)
    target, pred, r2, _ = train_lstm(lag=300, hidden_size=256, steps=1000) # 给它更多 N 和 步数
    plot_comparison(target, pred, r2, 300)