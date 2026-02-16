import torch
import torch.nn as nn
import numpy as np
from hippo_matrix import get_hippo_legs, get_hippo_lagt

def discretize_bilinear(A, B, dt):
    """
    使用双线性变换 (Bilinear/Tustin Transform) 将连续 ODE 离散化
    
    原理 (Report 核心公式):
    c_{t+1} = A_bar @ c_t + B_bar @ u_t
    
    其中:
    A_bar = (I - dt/2 * A)^{-1} @ (I + dt/2 * A)
    B_bar = (I - dt/2 * A)^{-1} @ (dt * B)
    """
    N = A.shape[0]
    I = torch.eye(N)
    
    # 预计算中间项: (I - dt/2 * A)
    # 这是一个线性系统求解问题，比直接求逆更稳，但在这里 N 较小直接求逆也可以
    # term1 = (I - dt/2 * A)
    term1 = I - (dt / 2.0) * A
    term2 = I + (dt / 2.0) * A
    
    # 求逆: term1_inv
    term1_inv = torch.linalg.inv(term1)
    
    # 计算离散矩阵
    A_bar = term1_inv @ term2
    B_bar = term1_inv @ (dt * B)
    
    return A_bar, B_bar

class HiPPOCell(nn.Module):
    def __init__(self, N, dt=1.0, measure='legs'):
        """
        HiPPO 记忆单元
        
        Args:
            N (int): 多项式阶数 (记忆容量)
            dt (float): 采样步长
            measure (str): 'legs' (Legendre) or 'lagt' (Laguerre)
        """
        super().__init__()
        self.N = N
        self.dt = dt
        self.measure = measure
        
        # 1. 生成连续矩阵 A, B
        if measure == 'legs':
            A, B = get_hippo_legs(N)
        elif measure == 'lagt':
            A, B = get_hippo_lagt(N)
        else:
            raise ValueError(f"Unknown measure: {measure}")
            
        # 2. 离散化 (Discretization)
        # 这是将 ODE 变成 RNN 的关键一步
        A_bar, B_bar = discretize_bilinear(A, B, dt)
        
        # 3. 注册为 Buffer (非训练参数)
        # 我们不训练 A_bar 和 B_bar，它们是数学推导出来的常量
        self.register_buffer('A_bar', A_bar)
        self.register_buffer('B_bar', B_bar)
        
    def forward(self, inputs):
        """
        Args:
            inputs: (Batch, Length, 1) - 输入的时间序列
            
        Returns:
            history: (Batch, Length, N) - 每个时刻的记忆状态 c(t)
        """
        batch_size, length, _ = inputs.shape
        
        # 初始化状态 c_t = 0
        c = torch.zeros(batch_size, self.N, device=inputs.device)
        
        # 记录历史状态
        history = []
        
        # 循环时间步 (Simple Recurrence)
        # 组会 Note: 这里可以用并行 Scan 加速 (S4)，但为了展示原理，我们写成循环
        for t in range(length):
            u_t = inputs[:, t, :] # (Batch, 1)
            
            # 核心更新公式: c_{t+1} = A_bar * c_t + B_bar * u_t
            # 矩阵乘法注意维度匹配: (Batch, N) = (Batch, N) @ (N, N)^T + ...
            # PyTorch 的 linear/matmul 处理 batch 很智能
            
            # c = A c + B u
            # 维度变换: (N, N) @ (N, B)^T -> (B, N)
            # 更直观的写法:
            c = torch.matmul(c, self.A_bar.t()) + torch.matmul(u_t, self.B_bar.t())
            
            history.append(c)
            
        # 堆叠成 Tensor
        history = torch.stack(history, dim=1) # (Batch, Length, N)
        return history

# --- 验证代码 ---
if __name__ == "__main__":
    # 参数设置
    N = 64
    dt = 1.0
    seq_len = 100
    batch_size = 16
    
    print(f"Initializing HiPPOCell (N={N}, dt={dt})...")
    cell = HiPPOCell(N=N, dt=dt, measure='legs')
    
    # 检查离散矩阵属性
    # 理论上 A_bar 应该是稳定的（特征值绝对值 <= 1 或者接近 1）
    eigenvalues = torch.linalg.eigvals(cell.A_bar)
    max_ev = eigenvalues.abs().max()
    print(f"Max eigenvalue magnitude of A_bar: {max_ev.item():.6f}")
    if max_ev <= 1.0001:
        print("✅ System is stable (eigenvalues <= 1)")
    else:
        print("⚠️ Warning: System might be unstable!")
        
    # 造一个伪数据跑一下
    dummy_input = torch.randn(batch_size, seq_len, 1)
    print(f"\nRunning forward pass with input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = cell(dummy_input)
        
    print(f"Output shape: {output.shape}")
    
    # 验证维度是否正确
    expected_shape = (batch_size, seq_len, N)
    if output.shape == expected_shape:
        print("✅ Output shape matches expected (Batch, Time, N).")
        print("Day 2 Mission Complete! Ready for experiments.")
    else:
        print(f"❌ Shape mismatch! Expected {expected_shape}, got {output.shape}")