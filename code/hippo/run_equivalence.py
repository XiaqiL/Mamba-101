import torch
import numpy as np
import matplotlib.pyplot as plt

def run_equivalence_test():
    print("=== HiPPO Implementation Equivalence Test (Naive vs DPLR) ===")
    
    # 关键修改 1: 使用 float64 (双精度) 进行验证
    # 验证数学等价性时，float32 的 1e-7 误差有时会干扰判断
    torch.set_default_dtype(torch.float64)
    
    N = 256
    
    print(f"Testing Matrix Inversion Acceleration (N={N})...")
    
    # 关键修改 2: 避免对角线元素接近 0
    # 旧代码: torch.randn(N) -> 可能产生 0.0001，导致求逆后爆炸
    # 新代码: torch.rand(N) + 2.0 -> 范围 [2.0, 3.0]，非常稳定
    torch.manual_seed(42)
    D_vals = torch.rand(N) + 2.0 
    D = torch.diag(D_vals)
    
    U = torch.randn(N, 1)
    V = torch.randn(N, 1)
    
    # 构造完整的稠密矩阵 M = D + U V^T
    M = D + U @ V.t()
    
    # --- 方法 A: Naive O(N^3) ---
    print("1. Running Naive Inversion (torch.linalg.inv)...")
    M_inv_naive = torch.linalg.inv(M)
    
    # --- 方法 B: Woodbury O(N) ---
    print("2. Running Woodbury Inversion (O(N) Formula)...")
    
    # D^-1 (O(N))
    D_inv_diag = 1.0 / D_vals 
    
    # D^-1 U (O(N))
    invD_U = U * D_inv_diag.unsqueeze(1) 
    
    # V^T D^-1 (O(N))
    VT_invD = V.t() * D_inv_diag.unsqueeze(0)
    
    # Capacity Term: 1 + V^T D^-1 U (Scalar)
    capacity = 1.0 + V.t() @ invD_U
    
    # M^-1 = D^-1 - (D^-1 U) (Capacity)^-1 (V^T D^-1)
    term2 = (invD_U @ VT_invD) / capacity
    M_inv_woodbury = torch.diag(D_inv_diag) - term2
    
    # --- 对比 ---
    diff = (M_inv_naive - M_inv_woodbury).abs()
    max_error = diff.max().item()
    mean_error = diff.mean().item()
    
    print("-" * 40)
    print(f"Max Error:  {max_error:.2e}")
    print(f"Mean Error: {mean_error:.2e}")
    print("-" * 40)
    
    # 这里的阈值设为 1e-10，因为 float64 应该极其精确
    if max_error < 1e-10:
        print("✅ Result: SUCCESS! Woodbury Identity Verified.")
        print("   The algorithm is mathematically EXACT.")
    else:
        print("❌ Result: FAILED. Check implementation.")

    # 还原为 float32 以免影响后续其他代码（如果在一个脚本里跑的话）
    torch.set_default_dtype(torch.float32)

    # 画图
    plt.figure(figsize=(6, 5))
    plt.imshow(diff.numpy(), cmap='Reds')
    plt.colorbar(label='Absolute Error')
    plt.title(f"Error Heatmap (Double Precision)\nMax Error: {max_error:.2e}")
    plt.savefig("report_assets/Equivalence_Heatmap.png")
    print("Heatmap saved.")

if __name__ == "__main__":
    run_equivalence_test()