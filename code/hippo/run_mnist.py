import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from torchvision import datasets, transforms
from hippo_cell import HiPPOCell

def run_mnist_reconstruction():
    print("Loading MNIST...")
    # 1. 加载 MNIST 数据 (自动下载)
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # 取出一张图，比如第 0 张 (是个 5)
    image, label = dataset[0] 
    # image shape: (1, 28, 28) -> flatten -> (784, 1)
    seq = image.view(-1, 1)
    length = seq.shape[0]
    
    # 2. HiPPO 设置 (N要大一点，因为 784 比较长)
    N = 256 
    cell = HiPPOCell(N=N, dt=1.0, measure='legs') # dt=1.0 在这里通常可以，或者试 0.1
    
    print(f"Encoding Image (Length={length}) with HiPPO (N={N})...")
    
    # 3. 前向传播
    # 这里我们只取 "最后一个时刻" 的状态来重建 "整个历史"
    # 或者像之前一样，用所有时刻的状态去训练解码器
    # 为了简单，我们还是用 "Online Reconstruction" 的方式训练
    
    inputs = seq.view(1, length, 1) # (Batch, Time, Dim)
    with torch.no_grad():
        states = cell(inputs) # (1, 784, N)
        
    X = states.squeeze().numpy() # (784, N)
    Y = seq.squeeze().numpy()    # (784,)
    
    # 4. 训练解码器 (Ridge Regression 防止过拟合)
    # 任务：利用 c_t 重建 f_t (当前时刻重建) 
    # 或者：利用 c_T (最后时刻) 重建 f_{0...T} (全历史回顾) <--- 这个更酷但更难
    # 我们先做最稳的：Function Reconstruction (Online)
    # 即测试 HiPPO 是否在每一步都记住了之前的样子
    
    # 比如我们在 t=784 时，尝试重建 t=0 到 784 的所有像素
    # 这需要一个特定的 Projection Matrix。
    # 为了简化，我们直接展示 HiPPO 的 "当前时刻重建能力" (Reconstruction at lag=0)
    # 或者稍微难一点：Lag=200 (能否在读脚的时候记住头)
    
    lag = 0 # 先试简单的，看能否实时复现
    clf = Ridge(alpha=0.1)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    
    # 5. 画图
    img_recon = Y_pred.reshape(28, 28)
    img_orig = Y.reshape(28, 28)
    
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_orig, cmap='gray')
    plt.title("Original MNIST")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_recon, cmap='gray')
    plt.title(f"HiPPO Recon (N={N})")
    plt.axis('off')
    
    plt.savefig("report_assets/MNIST_Recon.png")
    print("Saved MNIST_Recon.png")

if __name__ == "__main__":
    run_mnist_reconstruction()