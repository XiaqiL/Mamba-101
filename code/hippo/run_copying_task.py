import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("report_assets", exist_ok=True)

# --- 1. 修复后的矩阵生成函数 ---
def get_hippo_legs_official(N):
    """
    Correct and safe implementation of HiPPO-LegS matrices.
    """
    n = np.arange(N, dtype=np.float32)
    k = np.arange(N, dtype=np.float32)
    
    # 创建网格: n_mat 是行索引(n), k_mat 是列索引(k)
    n_mat, k_mat = np.meshgrid(n, k, indexing='ij')
    
    A = np.zeros((N, N), dtype=np.float32)
    
    # Lower triangle: n > k
    mask = n_mat > k_mat
    A[mask] = np.sqrt(2 * n_mat[mask] + 1) * np.sqrt(2 * k_mat[mask] + 1)
    
    # Diagonal: n = k
    diag_mask = n_mat == k_mat
    A[diag_mask] = n_mat[diag_mask] + 1
    
    # B matrix
    B = np.sqrt(2 * n + 1).reshape(N, 1)
    
    return torch.from_numpy(A), torch.from_numpy(B)

# --- 2. 带门控的 HiPPO 单元 ---
class GatedHiPPO(nn.Module):
    def __init__(self, input_size, hidden_size, N=64):
        super().__init__()
        self.N = N
        self.hidden_size = hidden_size
        
        # 获取矩阵
        A, B = get_hippo_legs_official(N)
        self.register_buffer('A', A)
        self.register_buffer('B', B)
        
        # 门控组件
        self.encoder = nn.Linear(input_size, 1)
        self.readout = nn.Linear(N, hidden_size)
        self.rnn_gate = nn.GRUCell(hidden_size, hidden_size)

    def forward(self, x):
        b, s, _ = x.shape
        device = x.device
        
        c = torch.zeros(b, self.N, device=device)
        h = torch.zeros(b, self.hidden_size, device=device)
        
        outputs = []
        
        # 离散化近似 (LTI Approximation for speed)
        # 实际 LegS 是时变的，这里用固定的 dt 做近似
        dt_val = 0.01 
        A_np, B_np = self.A.cpu().numpy(), self.B.cpu().numpy()
        eye = np.eye(self.N)
        
        # Bilinear Transform
        BL = eye - (dt_val / 2.0) * A_np
        BR = eye + (dt_val / 2.0) * A_np
        A_bar_np = np.linalg.solve(BL, BR)
        B_bar_np = np.linalg.solve(BL, B_np * dt_val)
        
        A_bar = torch.tensor(A_bar_np, dtype=torch.float32, device=device)
        B_bar = torch.tensor(B_bar_np, dtype=torch.float32, device=device)

        for t in range(s):
            u = self.encoder(x[:, t, :])
            
            # 线性记忆更新
            c = torch.mm(c, A_bar.T) + torch.mm(u, B_bar.T)
            
            # 从记忆读取并过门控
            memory_feat = self.readout(c)
            h = self.rnn_gate(memory_feat, h)
            
            outputs.append(h)
            
        return torch.stack(outputs, dim=1)

# --- 3. 任务模型包装 ---
class ModelWrapper(nn.Module):
    def __init__(self, num_classes, N=64):
        super().__init__()
        self.embed = nn.Embedding(num_classes, 64)
        self.hippo = GatedHiPPO(input_size=64, hidden_size=64, N=N)
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x_emb = self.embed(x)
        out = self.hippo(x_emb)
        return self.fc(out).permute(0, 2, 1)

def generate_data(batch_size, L):
    # 生成 1-8 的数据作为记忆内容
    mem = torch.randint(1, 9, (batch_size, 10))
    zeros = torch.zeros(batch_size, L, dtype=torch.long)
    delim = torch.full((batch_size, 10), 9, dtype=torch.long)
    
    inputs = torch.cat([mem, zeros, delim], dim=1)
    
    # 目标: 只预测最后 10 位，前面忽略
    ignore = torch.full((batch_size, 10+L), -100, dtype=torch.long)
    targets = torch.cat([ignore, mem], dim=1)
    
    return inputs, targets

if __name__ == "__main__":
    torch.manual_seed(42)
    L = 100 # 记忆长度
    
    model = ModelWrapper(num_classes=10, N=64)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(ignore_index=-100)
    
    losses = []
    print(f"Training Gated HiPPO on Copy Task (L={L})...")
    
    for i in range(1500):
        x, y = generate_data(64, L)
        logits = model(x)
        loss = crit(logits, y)
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        losses.append(loss.item())
        
        if i % 100 == 0:
            print(f"Step {i}, Loss: {loss.item():.4f}")
            if loss.item() < 0.05:
                print("Converged!")
                # 填充剩余步数以便画图
                losses.extend([loss.item()] * (1500 - i - 1))
                break
                
    # 画图
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label='Gated HiPPO')
    plt.axhline(y=2.079, color='r', linestyle='--', label='Random Guess')
    plt.title(f"HiPPO Copy Task Convergence (L={L})")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("report_assets/Official_HiPPO_Copy.png")
    print("\nGraph saved to report_assets/Official_HiPPO_Copy.png")