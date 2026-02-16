import torch
import numpy as np
import matplotlib.pyplot as plt

def get_hippo_legs(N):
    """
    生成 HiPPO-LegS (Scaled Legendre) 矩阵
    
    数学定义 (NeurIPS 2020 Sec 3):
    A_{nk} = -(2n+1)^{1/2} (2k+1)^{1/2}  if n > k
    A_{nn} = -(n+1)
    B_n    = (2n+1)^{1/2}
    """
    # 1. 初始化空矩阵
    A = torch.zeros(N, N)
    B = torch.zeros(N, 1)
    
    # 辅助向量: (2n+1)^{1/2}
    # n 从 0 到 N-1
    n = torch.arange(N).float()
    sqrt_2n_plus_1 = (2 * n + 1).sqrt()

    # 2. 填充 B 矩阵
    # B_n = (2n+1)^{1/2}
    B[:, 0] = sqrt_2n_plus_1

    # 3. 填充 A 矩阵
    # 方法：利用广播机制避免慢速循环
    # A_nk (n > k) 部分
    col_vec = sqrt_2n_plus_1.unsqueeze(1) # (N, 1)
    row_vec = sqrt_2n_plus_1.unsqueeze(0) # (1, N)
    
    # 外积：(2n+1)^{1/2} * (2k+1)^{1/2}
    outer_prod = col_vec @ row_vec 
    
    # 生成下三角掩码 (Lower Triangular Mask, k < n)
    # torch.tril 返回下三角，k=-1 表示不包含对角线
    mask = torch.tril(torch.ones(N, N), diagonal=-1)
    
    # 应用公式：A = - outer_prod * mask
    A = - outer_prod * mask
    
    # 修正对角线: A_{nn} = -(n+1)
    # torch.diagonal 返回的是视图，可以直接修改
    A.diagonal().copy_(-(n + 1))
    
    return A, B

def get_hippo_lagt(N):
    """
    生成 HiPPO-LagT (Translated Laguerre) 矩阵
    
    数学定义 (Appendix D.2):
    A_{nk} = -1  if n >= k (下三角全是 -1, 包括对角线)
    B_n    = 1
    """
    # 1. 初始化
    A = torch.zeros(N, N)
    B = torch.ones(N, 1) # B 全是 1
    
    # 2. 填充 A (下三角全为 -1)
    # torch.tril(ones) 生成下三角全 1 矩阵
    A = -1.0 * torch.tril(torch.ones(N, N))
    
    return A, B

def plot_hippo_matrix(A, title="HiPPO Matrix"):
    """
    可视化矩阵 A (Report 素材生成器)
    """
    plt.figure(figsize=(6, 5))
    # 使用 imshow 绘制热力图
    im = plt.imshow(A.numpy(), cmap='coolwarm')
    plt.colorbar(im)
    plt.title(title)
    plt.xlabel("k (Input history)")
    plt.ylabel("n (Polynomial order)")
    
    # 保存图片到 report_assets
    save_path = f"report_assets/{title.replace(' ', '_')}.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close() # 关闭画布防止在某些IDE中卡住

def get_hippo_legt(N):
    """
    生成 HiPPO-LegT (Translated Legendre) 矩阵
    
    数学定义 (NeurIPS 2020 Appendix D.1 & S4 Paper Eq.5):
    对应滑动窗口 measure (Sliding Window)，权重在 [t-theta, t] 上均匀分布。
    
    A_{nk} = -(2n+1)^{1/2} (2k+1)^{1/2} * M_{nk}
    其中 M_{nk} = 1          if k <= n
                 (-1)^{n-k}  if k > n
                 
    B_n    = (2n+1)^{1/2}
    """
    # 1. 基础向量 sqrt(2n+1)
    n = torch.arange(N).float()
    sqrt_2n_plus_1 = (2 * n + 1).sqrt()
    
    # 2. 填充 B 矩阵
    # B_n = (2n+1)^{1/2}
    B = torch.zeros(N, 1)
    B[:, 0] = sqrt_2n_plus_1
    
    # 3. 填充 A 矩阵
    # A 的构造分为两步：
    # Step 3.1: 基础缩放因子 -(2n+1)^{1/2} (2k+1)^{1/2}
    # 这部分和 LegS 是一样的
    col_vec = sqrt_2n_plus_1.unsqueeze(1) # (N, 1)
    row_vec = sqrt_2n_plus_1.unsqueeze(0) # (1, N)
    scaling = -1 * (col_vec @ row_vec)    # (N, N)
    
    # Step 3.2: 特殊结构矩阵 M
    # k <= n (下三角 + 对角线): 1
    # k > n  (上三角): (-1)^(n-k)
    
    # 生成 n 和 k 的网格
    n_mat, k_mat = torch.meshgrid(n, n, indexing='ij')
    
    # 初始化 M 为全 1 (处理 k <= n 部分)
    M = torch.ones(N, N)
    
    # 处理 k > n 部分 (上三角)
    upper_mask = k_mat > n_mat
    diff = n_mat - k_mat # n-k (负数)
    # (-1)^(n-k) 等价于 (-1)^(k-n) 因为 (-1)^x = (-1)^(-x)
    # 或者直接计算 power
    M[upper_mask] = (-1.0) ** (n_mat[upper_mask] - k_mat[upper_mask])
    
    # 组合 A
    A = scaling * M
    
    return A, B


def get_hippo_fout(N):
    """
    生成 HiPPO-FouT (Translated Fourier) 矩阵
    
    数学定义 (来源于 S4 Paper, Gu et al. 2022):
    FouT 使用傅里叶基 (sin/cos) 在滑动窗口上逼近信号。
    
    矩阵 A 的结构包含:
    1. 频率旋转项 (Rotation): 2πk, 对应 sin/cos 的导数关系
    2. 边界交互项 (Interaction): 处理滑动窗口进入/移出的数据
    
    定义:
    A_nk 包含多个条件分支 (详见代码注释)
    B_n = 2 (n=0), 2√2 (n odd), 0 (otherwise)
    """
    A = torch.zeros(N, N)
    B = torch.zeros(N, 1)
    
    n = torch.arange(N)
    k = torch.arange(N)
    
    # --- 生成 B 矩阵 ---
    # B_0 = 2
    B[0, 0] = 2.0
    # B_n = 2√2 if n is odd
    # n[1::2] 取所有奇数索引
    B[1::2, 0] = 2.0 * np.sqrt(2.0)
    
    # --- 生成 A 矩阵 ---
    # 使用循环虽然慢一点，但为了准确复现复杂的条件分支，这是最安全的可读方式
    # 对于 N=256 这种规模，循环完全没问题
    
    for n_idx in range(N):
        for k_idx in range(N):
            # Condition 1: n=k=0
            if n_idx == 0 and k_idx == 0:
                A[n_idx, k_idx] = -2.0
            
            # Condition 2: n=0, k odd
            elif n_idx == 0 and k_idx % 2 == 1:
                A[n_idx, k_idx] = -2.0 * np.sqrt(2.0)

            # Condition 3: k=0, n odd
            elif k_idx == 0 and n_idx % 2 == 1:
                A[n_idx, k_idx] = -2.0 * np.sqrt(2.0)
                
            # Condition 4: Both n, k are odd
            # 这是稠密部分 (Dense Interaction)
            elif n_idx % 2 == 1 and k_idx % 2 == 1:
                A[n_idx, k_idx] = -4.0
            
            # Condition 5: n-k=1, k odd (即 n是偶数, k是n-1)
            # 对应频率项: sin' -> cos
            elif (n_idx - k_idx == 1) and (k_idx % 2 == 1):
                A[n_idx, k_idx] = 2.0 * np.pi * k_idx
            
            # Condition 6: k-n=1, n odd (即 k是偶数, n是k-1)
            # 对应频率项: cos' -> -sin
            elif (k_idx - n_idx == 1) and (n_idx % 2 == 1):
                A[n_idx, k_idx] = -2.0 * np.pi * n_idx
            
            # Otherwise 0 (已经初始化为0)

    return A, B
def plot_all_measures(N):
    """
    对比画出 LegS, LagT, LegT 三种矩阵的热力图
    """
    # 获取三种矩阵
    # 假设你之前的 get_hippo_legs 和 get_hippo_lagt 已经定义在同个文件或import进来
    # 这里为了演示完整性，直接调用 (请确保上下文中有这两个函数)
    A_legs, _ = get_hippo_legs(N)
    A_lagt, _ = get_hippo_lagt(N)
    A_legt, _ = get_hippo_legt(N)
    A_fout, _ = get_hippo_fout(N)
    
    
    matrices = [
        ("HiPPO-LegS (Scaled)", A_legs, "History: All [0, t]"),
        ("HiPPO-LagT (Translated)", A_lagt, "History: Exp Decay"),
        ("HiPPO-LegT (Window)", A_legt, "History: Window [t-w, t]"),
        ("HiPPO-FouT (Fourier)", A_fout, "History: Window [t-w, t] + Frequencies"),
    ]
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    for i, (name, matrix, desc) in enumerate(matrices):
        ax = axes[i]
        im = ax.imshow(matrix.numpy(), cmap='coolwarm', vmin=-2*N**0.5, vmax=2*N**0.5) # 统一色阶方便对比
        ax.set_title(f"{name}\n{desc}")
        ax.set_xlabel("k (Input)")
        ax.set_ylabel("n (Order)")
        if i == 0:
            ax.set_ylabel("n (Polynomial Order)")
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = "report_assets/HiPPO_Measures_Comparison.png"
    plt.savefig(save_path)
    print(f"Comparison plot saved to {save_path}")

# --- 补充：关于 Chebyshev ---
# HiPPO 论文中的 Chebyshev 通常指“离线最优逼近”作为基准线。
# 也就是不跑 RNN，而是直接在每一时刻 t，把 f[0...t] 拿出来，
# 用 Chebyshev 多项式公式算出系数 c_n。
# 如果你需要在代码中实现它作为对比 Baseline，通常不需要生成 A 矩阵，
# 而是直接写一个函数 calculate_chebyshev_coeffs(signal)。

if __name__ == "__main__":
    # 运行对比
    plot_all_measures(N=64)