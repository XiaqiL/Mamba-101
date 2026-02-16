import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from hippo_matrix_4_measures import get_hippo_legs

# --- 1. 工具函数: R^2 和 GBT ---
def generate_signal(length=2000, type='white_noise'):
    """生成测试信号"""
    if type == 'white_noise':
        return torch.randn(1, length, 1)
    return torch.randn(1, length, 1)
def r2_score(y_true, y_pred):
    """
    计算 R^2 (Coefficient of Determination)
    R^2 = 1 - (SS_res / SS_tot)
    """
    # 展平
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    
    # 防止分母为0
    if ss_tot < 1e-6:
        return torch.tensor(0.0)
        
    return 1 - (ss_res / ss_tot)

def discretize_gbt(A, B, dt, alpha):
    """
    Generalized Bilinear Transform (GBT)
    alpha=0 (Forward), alpha=0.5 (Bilinear), alpha=1 (Backward)
    """
    N = A.shape[0]
    I = torch.eye(N)
    
    # (I - alpha*dt*A) * c_{t+1} = (I + (1-alpha)*dt*A) * c_t + dt*B*u
    term_left = I - alpha * dt * A
    
    try:
        term_left_inv = torch.linalg.inv(term_left)
    except RuntimeError:
        # 如果矩阵奇异（通常不会发生，除非 dt 极大），返回全0
        print(f"Warning: Singular matrix for alpha={alpha}")
        return torch.zeros_like(A), torch.zeros_like(B)
    
    term_right = I + (1 - alpha) * dt * A
    
    A_bar = term_left_inv @ term_right
    B_bar = term_left_inv @ (dt * B)
    
    return A_bar, B_bar

# --- 2. 模型定义 ---

class HiPPOCell_GBT(nn.Module):
    def __init__(self, N, dt=0.01, alpha=0.5):
        super().__init__()
        self.N = N
        A, B = get_hippo_legs(N)
        
        # 离散化
        A_bar, B_bar = discretize_gbt(A, B, dt, alpha)
        
        self.register_buffer('A_bar', A_bar)
        self.register_buffer('B_bar', B_bar)
        
    def forward(self, inputs):
        # inputs: (Batch, Length, 1)
        batch_size, length, _ = inputs.shape
        c = torch.zeros(batch_size, self.N, device=inputs.device)
        history = []
        
        # 为了防止数值爆炸导致整个程序崩溃，加入简单的 NaN 检查
        for t in range(length):
            u_t = inputs[:, t, :]
            c = torch.matmul(c, self.A_bar.t()) + torch.matmul(u_t, self.B_bar.t())
            
            # 如果检测到 NaN 或 Inf，停止计算以保护绘图
            if torch.isnan(c).any() or torch.isinf(c).any():
                c = torch.full_like(c, float('nan'))
                history.extend([c] * (length - t))
                break
                
            history.append(c)
            
        return torch.stack(history, dim=1)

# --- 3. 实验主逻辑 ---

if __name__ == "__main__":
    # 参数
    N = 256
    dt = 0.01
    L = 2000 # 序列长度
    lag = 100

    # 关键：使用带限白噪声 (Band-limited White Noise) 来测试重建能力
    # 为了让 R^2 有意义，我们需要信号有一定的复杂度
    torch.manual_seed(42)
    input_signal = generate_signal(length=L, type='white_noise')
    
    alphas = [0.0, 0.5, 1.0]
    alpha_names = {0.0: 'Forward Euler (Explodes)', 
                   0.5: 'Bilinear (Tustin)', 
                   1.0: 'Backward Euler (Damped)'}
    
    plt.figure(figsize=(15, 6))
    targets = input_signal[:, :-lag, :].squeeze(0)
    
    for i, alpha in enumerate(alphas):
        print(f"Running Lagged Reconstruction (Lag={lag}) with {alpha_names[alpha]}...")
        
        model = HiPPOCell_GBT(N, dt, alpha=alpha)
        
        with torch.no_grad():
            history = model(input_signal) # Shape: (1, L, N)
            
        # 截取对应的 State
        # 我们需要从 index = lag 开始取，这样 state[0] 就是 c[lag]
        states = history[:, lag:, :].squeeze(0) # Shape: (L-lag, N)
        
        # 检查是否数值爆炸
        if torch.isnan(states).any():
            r2 = -float('inf')
            pred = torch.zeros_like(targets)
            status = "EXPLODED"
            col = 'red'
        else:
            # 训练线性解码器: Least Squares
            # W = (X^T X)^-1 X^T Y
            try:
                # 求解 W
                sol = torch.linalg.lstsq(states, targets).solution
                # 预测
                pred = states @ sol
                # 计算 R^2
                r2 = r2_score(targets, pred)
                status = f"R^2 = {r2:.4f}"
                col = 'tab:blue'
            except Exception as e:
                status = "Error"
                pred = torch.zeros_like(targets)
                col = 'red'

        # 绘图
        plt.subplot(1, 3, i+1)
        
        # 画一部分用于展示细节 (只画中间200个点)
        plot_len = 200
        start_idx = 100
        end_idx = start_idx + plot_len
        
        # 绘制 Ground Truth (Delayed Input)
        plt.plot(targets[start_idx:end_idx].numpy(), color='gray', alpha=0.6, linewidth=2, label=f'Input (t-{lag})')
        
        # 绘制 Reconstruction
        if status != "EXPLODED":
            plt.plot(pred[start_idx:end_idx].numpy(), color=col, linestyle='--', linewidth=1.5, label='Reconstruction')
            
        plt.title(f"{alpha_names[alpha]}\nScore: {status}")
        plt.legend(loc='upper right', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.ylim(-3, 3)

    plt.suptitle(f"Memory Reconstruction Task (Lag={lag}, N={N}, White Noise)", fontsize=16)
    plt.tight_layout()
    plt.savefig(f'report_assets/gbt_lag_{N}.png')
    print(f"Graph saved to report_assets/gbt_lag_{N}.png")