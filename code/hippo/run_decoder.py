import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from hippo_cell_4_measures import HiPPOCell_4 as HiPPOCell
import os

os.makedirs("report_assets", exist_ok=True)
def generate_white_noise(length, type='white_noise'):
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
# === 1. 数据生成器 ===
def generate_signal(length=2000):
    # 使用混合正弦波，因为之前证明了它对特征提取要求较高
    t = torch.linspace(0, 100, length)
    signal = torch.sin(t) + 0.5 * torch.sin(3*t) + 0.2 * torch.sin(10*t)
    # 加一点点噪声防止过拟合
    signal += 0.05 * torch.randn(length)
    return signal.view(1, length, 1)

# === 2. 解码器训练包装器 ===
def train_decoder(X_train, Y_train, X_test, Y_test, decoder_type):
    """
    对比三种解码器:
    1. OLS (Linear): 普通线性回归 (无正则)
    2. Ridge (Linear + L2): 带正则的线性回归 (推荐)
    3. MLP (Non-linear): 多层感知机 (强大但昂贵)
    """
    if decoder_type == 'linear':
        model = LinearRegression()
        model.fit(X_train, Y_train)
        score = model.score(X_test, Y_test)
        pred = model.predict(X_test)
        
    elif decoder_type == 'ridge':
        # alpha=1.0 是默认正则化强度
        model = Ridge(alpha=1.0)
        model.fit(X_train, Y_train)
        score = model.score(X_test, Y_test)
        pred = model.predict(X_test)
        
    elif decoder_type == 'mlp':
        # 简单的 MLP: Hidden(100) -> ReLU -> Out
        # solver='adam', max_iter=500
        model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', 
                             solver='adam', max_iter=500, random_state=42)
        # sklearn 的 MLP 输入需要 flatten
        # 注意: MLP 训练比 Linear 慢得多
        model.fit(X_train, Y_train.ravel())
        score = model.score(X_test, Y_test.ravel())
        pred = model.predict(X_test).reshape(-1, 1)
        
    return pred, score

# === 3. 实验一: Decoder 对比 (在 Lag=100 任务上) ===
def run_decoder_comparison():
    print("\n=== Experiment 1: Decoder Comparison (Linear vs Ridge vs MLP) ===")
    N = 256 # 故意用小一点的 N，看谁能榨出更多性能
    measure = 'legs'
    dt = 0.01
    lag = 100
    
    # 生成数据
    inputs = generate_white_noise(3000)
    
    # 提取特征
    cell = HiPPOCell(N=N, dt=dt, measure=measure)
    with torch.no_grad():
        states = cell(inputs).squeeze(0).numpy() # (L, N)
    targets = inputs.squeeze(0).numpy() # (L, 1)
    
    # 构建 Lag 任务
    X_train = states[lag:-500]
    Y_train = targets[:-lag-500]
    X_test = states[-500:]
    Y_test = targets[-500-lag:-lag]
    
    decoders = ['linear', 'ridge', 'mlp']
    results = {}
    
    for dec in decoders:
        print(f"Training {dec.upper()} decoder...")
        pred, score = train_decoder(X_train, Y_train, X_test, Y_test, dec)
        results[dec] = (Y_test, pred, score)
        print(f"  -> {dec} R^2: {score:.4f}")
        
    # 画图
    plt.figure(figsize=(10, 5))
    plt.plot(results['ridge'][0][:200], 'k', alpha=0.3, linewidth=3, label='Ground Truth')
    
    colors = {'linear': 'red', 'ridge': 'blue', 'mlp': 'green'}
    styles = {'linear': ':', 'ridge': '--', 'mlp': '-.'}
    
    for dec, (tgt, pred, score) in results.items():
        plt.plot(pred[:200], color=colors[dec], linestyle=styles[dec], 
                 label=f'{dec.upper()} ($R^2={score:.3f}$)')
        
    plt.title(f"Decoder Performance Comparison (N={N}, Measure={measure})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"report_assets/Exp_Decoder_Comparison_{N}.png")
    print(f"Graph saved to report_assets/Exp_Decoder_Comparison_{N}.png")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    run_decoder_comparison()