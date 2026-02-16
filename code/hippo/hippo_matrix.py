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

# --- 验证代码 (直接运行此文件时执行) ---
if __name__ == "__main__":
    # 1. 验证 LegS
    N = 8 # 用个小一点的 N 方便肉眼看数值
    A_legs, B_legs = get_hippo_legs(N)
    
    print(f"=== HiPPO-LegS (N={N}) ===")
    print("Matrix A (First 4x4):\n", A_legs[:4, :4])
    print("Matrix B (First 4):\n", B_legs[:4])
    
    plot_hippo_matrix(A_legs, "HiPPO-LegS Matrix A")
    
    # 2. 验证 LagT
    A_lagt, B_lagt = get_hippo_lagt(N)
    print(f"\n=== HiPPO-LagT (N={N}) ===")
    print("Matrix A (First 4x4):\n", A_lagt[:4, :4])
    
    plot_hippo_matrix(A_lagt, "HiPPO-LagT Matrix A")
    
    print("\n✅ Matrix Generation Complete. Check 'report_assets' for images.")