# HiPPO_Experiment

Checklist

- [x]  LegS (handwritten): Legendre Polynomials → Change of Variables from [-1,1] to [0,t] → Derivatives of Polynomials → Dynamics of c(t) (Leibniz Integration) → Induction of Matrices A and B → Discretisation
- [x]  Different methods of Discretisation
- [x]  Fourier Basis
- [x]  From O(N^2) complexity to O(N)
- [x]  Different decoders

---

涉及到的变量/参数：

tested

1. input_signal(f) = [step, mixed_sine, white_noise, random_walk, slow_triangle_wave]
2. input_signal_period = [600, 1200]
3. input_signal_length = [2000, 3000]
4. lag = [100, 300, 2000]
5. N = [64, 128, 256, 512, 1024, 2048]
6. measure_families = [legt, lagt, legs, fout]
7. decoder = [LinearRegression, Ridge, mlp]
- Ridge_alpha = [1.0]
- mlp_param = (hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
1. discretisation = [forward_euler, Bilinear, backward_euler]
2. noise_level(robustness_test) = [0, 0.5, 1.0]
3. dt(step_size) = [1.0, 0.1, 0.01, 0.001]
4. legt_sliding_window(\theta) = [1.0]

to be tested

decoder_param:

- ridge_alpha: [1e-3, 1]
- mlp_config: {lr, hidden_dim, layers}

seed

lagt_scaling: `[fixed, adaptive]` 

---

一些比较重要的code dump (其中参数代码如下，实验代码会在讲到这个实验的时候贴上)

input_signal(f) = [step, mixed_sine, white_noise, random_walk, slow_triangle_wave]

```python
def generate_signal(length, type):
		# -----------------------------
    # Input
    #   length: 序列长度 L（时间步数）
    #   type:   信号类型字符串
    #
    # Output
    #   Return shape (B=1, L=length, C=1)
    #
    # Shape convention
    #   (B, L, C) = (batch, time, channel)
    #   Always returning (1, L, 1).
    # -----------------------------
    
    # Generate time axis t (1D tensor, shape=(L,))
    # Key point:
    #   t 的范围与 length 一起决定 dt = (100-0)/(L-1)（隐含采样间隔）
    #   dt 会影响离散频率表现，从而影响“多频信号”的难度与模型频响讨论
    t = torch.linspace(0, 100, length)
    if type == 'white_noise':
        # White noise (uniform [0,1); Gaussian white noise typically uses randn)
        # 输出 shape: torch.rand(1, L, 1) => (B=1, L, C=1)
        return torch.randn(1, length, 1)
    elif type == 'step':
        # Piecewise-constant step signal: random jump every fixed interval, then hold.
        # signal 初始 shape=(L,)
        signal = torch.zeros(length)
        # Key parameter:
        # Here step_interval=200:
        #   Smaller interval => more jumps (higher frequency) => harder.
        num_jumps = length // 200
        for i in range(num_jumps):
            # Value chosen from {-1,0,1} to create clear discontinuities.
            val = np.random.choice([-1, 0, 1])
            # This assignment sets the tail; later jumps overwrite the later tail,
            # resulting in stepwise-constant segments.
            signal[i*200:] = val
        # reshape: (L,) -> (1, L, 1)
        # Shape change: 1D vector to (B,L,C)
        return signal.view(1, length, 1)
    elif type == 'sinusoid':
        # Single sinusoid: smooth and single-frequency.
        signal = torch.sin(t)
        return signal.view(1, length, 1)
    elif type == 'mixed_sine':
        # Mixed sine: combines low + high frequency components (multi-scale).
        
        # Difficulty:
        #   高频成分越强（频率越高/幅值越大），越需要更大的状态维度 N 才能保留细节；
        signal = torch.sin(t) + 0.5 * torch.sin(3*t) + 0.3 * torch.sin(7*t)
        # reshape: (L,) -> (1, L, 1)
        return signal.view(1, length, 1)
    else:
        # 输入校验 / Input validation
        raise ValueError(f"Unknown signal type: {type}")
```

```python
def generate_random_walk(length=2000):
    # Random walk: strongly tests long-range dependence and drift.
    #
    # Output:
    #   (1, L, 1）
    noise = torch.randn(length) # shape=(L,)  高斯噪声 / Gaussian noise
    signal = torch.cumsum(noise, dim=0) # shape=(L,)  累积和形成随机游走 / random walk
    # Add offset (+5.0). Insight:
    #   - 这不会改变频率结构，但会改变均值/动态范围；
    signal = signal + 5.0  
    # reshape: (L,) -> (1, L, 1)
    return signal.view(1, length, 1)
```

```python
def generate_slow_triangle_wave(length, period):
    """
    生成一个周期极长的三角波。
    LegT 的窗口 (100步) 远小于半周期 (600步)。
    在窗口内，信号看起来就是一条直线 (局部不可区分)，
    只有拥有全局记忆(LegS)才能通过记住很久以前的拐点来定位相位。
    """
    # Extra insight:
    #   slow_triangle_wave 的价值在于制造“相位歧义”：
    #   当 period 很大时，任意短窗口里都近似线性，无法判断自己在上升段还是下降段，
    #   需要更长记忆才能利用更久远的拐点信息消除歧义。
    # Input:
    #   length: L
    #   period: 周期（步数）。period 越大 => 越慢 => 局部越不可区分 => 更难

    t = torch.arange(length).float() # shape=(L,) 离散时间索引 / discrete time index
    # phase: normalized phase in [0,1)
    # shape=(L,)
    phase = (t % period) / period
    # 三角波公式 / Triangle wave formula:
    # Range in [-1, 1]
    # shape=(L,)
    signal = 2 * torch.abs(2 * (phase - 0.5)) - 1
    # reshape: (L,) -> (1, L, 1）
    return signal.view(1, length, 1)
```

measure_families = [legt, lagt, legs, fout]

```python
def get_hippo_legs(N):
    """
    Generate HiPPO-LegS (Scaled Legendre) matrices (A, B).
    ----------------------------
    What is LegS? 
    ----------------------------
    LegS corresponds to a scaled Legendre measure (HiPPO 2020 Sec.3),
    encoding history via global projection onto a Legendre basis (multi-scale/global memory).
    ----------------------------
    Mathematical definition
    ----------------------------
    A_{nk} = -(2n+1)^{1/2} (2k+1)^{1/2}  if n > k
    A_{nn} = -(n+1)
    B_n    = (2n+1)^{1/2}
    ----------------------------
    Inputs/Outputs 
    ----------------------------
    Input:
        N: larger N => higher memory capacity but more compute/memory.
    Output:
        A: shape (N, N)
        B: shape (N, 1)
    ----------------------------
    Complexity / 复杂度
    ----------------------------
    broadcasting + outer product (O(N^2))
    """
    # 1. 初始化空矩阵
    A = torch.zeros(N, N)      # A: (N, N)
    B = torch.zeros(N, 1)      # B: (N, 1)

    # 辅助向量: (2n+1)^{1/2}
    n = torch.arange(N).float()                 # n: (N,)
    sqrt_2n_plus_1 = (2 * n + 1).sqrt()         # sqrt_2n_plus_1: (N,)

    # 2. 填充 B 矩阵
    # Shape: (N,) -> writes into (N,1)
    B[:, 0] = sqrt_2n_plus_1

    # 3. 填充 A 矩阵
    # A_{nk} for n>k: -sqrt(2n+1)*sqrt(2k+1)
    col_vec = sqrt_2n_plus_1.unsqueeze(1)  # (N, 1) column vector
    row_vec = sqrt_2n_plus_1.unsqueeze(0)  # (1, N) row vector

    # 外积：(2n+1)^{1/2} * (2k+1)^{1/2}
    outer_prod = col_vec @ row_vec         # (N, N)

    # 下三角掩码 (strictly lower triangular: k < n)
    mask = torch.tril(torch.ones(N, N), diagonal=-1)  # (N, N), entries 1 where k<n

    # 应用公式：A = - outer_prod * mask
    A = - outer_prod * mask

    # 修正对角线: A_{nn} = -(n+1)
    A.diagonal().copy_(-(n + 1))

    return A, B

def get_hippo_lagt(N):
    """
    Generate HiPPO-LagT (Translated Laguerre) matrices (A, B).
    ----------------------------
    What is LagT? 
    ----------------------------
    LagT uses a translated Laguerre structure (Appendix D.2):
    A is all -1 on and below diagonal; B is all ones.
    ----------------------------
    Mathematical definition 
    ----------------------------
    A_{nk} = -1  if n >= k (lower-triangular including diagonal)
    B_n    = 1
    ----------------------------
    Inputs/Outputs 
    ----------------------------
    Input:
        N: 状态维度
    Output:
        A: (N, N)
        B: (N, 1)
    ----------------------------
    Complexity / 复杂度
    ----------------------------
    A is O(N^2) 
    """
    # 1. 初始化
    A = torch.zeros(N, N)          # A: (N, N)
    B = torch.ones(N, 1)           # B: (N, 1)

    # 2. 填充 A (下三角全为 -1)
    A = -1.0 * torch.tril(torch.ones(N, N))    # A: (N, N)

    return A, B

def get_hippo_legt(N):
    """
    Generate HiPPO-LegT (Translated Legendre) matrices (A, B).
    ----------------------------
    What is LegT? 
    ----------------------------
    LegT corresponds to a sliding-window measure using a Legendre basis.
    Intuitively more window/local than LegS, while retaining Legendre structure.
    ----------------------------
    Mathematical definition
    ----------------------------
    A_{nk} = -(2n+1)^{1/2} (2k+1)^{1/2} * M_{nk}
    where
        M_{nk} = 1          if k <= n
                 (-1)^{n-k} if k > n

    B_n = (2n+1)^{1/2}
    ----------------------------
    Inputs/Outputs
    ----------------------------
    Input:
        N: 状态维度
    Output:
        A: (N, N)
        B: (N, 1)
    ----------------------------
    Shape notes / 维度说明
    ----------------------------
    - sqrt_2n_plus_1: (N,)
    - scaling: outer product -> (N, N)
    - M: (N, N) with piecewise sign pattern
    - A = scaling * M -> (N, N)
    """
    # 1. 基础向量 sqrt(2n+1)
    n = torch.arange(N).float()                 # n: (N,)
    sqrt_2n_plus_1 = (2 * n + 1).sqrt()         # (N,)

    # 2. 填充 B 矩阵
    B = torch.zeros(N, 1)                       # B: (N,1)
    B[:, 0] = sqrt_2n_plus_1

    # 3. 填充 A 矩阵
    # Step 3.1: scaling = - sqrt(2n+1) sqrt(2k+1) (same outer-product form as LegS)
    col_vec = sqrt_2n_plus_1.unsqueeze(1)       # (N, 1)
    row_vec = sqrt_2n_plus_1.unsqueeze(0)       # (1, N)
    scaling = -1 * (col_vec @ row_vec)          # (N, N)

    # Step 3.2: build M with piecewise pattern:
    #   if k <= n: M=1 (lower triangle + diagonal)
    #   if k > n : M = (-1)^(n-k) (alternating signs in upper triangle)
    n_mat, k_mat = torch.meshgrid(n, n, indexing='ij')   # both (N, N)

    # 初始化 M 为全 1（先覆盖 k<=n 的部分）
    M = torch.ones(N, N)                                 # (N, N)

    # 上三角区域 k>n：设置为 (-1)^(n-k)
    upper_mask = k_mat > n_mat                            # boolean mask (N, N)
    diff = n_mat - k_mat                                  # (N, N), negative values in upper region

    # (-1)^(n-k): sign alternation.
    M[upper_mask] = (-1.0) ** (n_mat[upper_mask] - k_mat[upper_mask])

    # 组合 A
    A = scaling * M

    return A, B
```

```python
def get_hippo_fout(N):
    """
    Generate HiPPO-FouT (Translated Fourier) matrices (A, B).
    ----------------------------
    What is FouT? 
    ----------------------------
    FouT uses a Fourier (sin/cos) basis on a sliding window (translated family),
    often discussed in S4/SSM context (e.g., Gu et al. 2022).
    ----------------------------
    High-level structure / 结构直觉
    ----------------------------
    A mixes:
      (1) rotation terms ~ ±2πk (local coupling between sine/cosine modes)
      (2) boundary interaction terms causing dense-ish interactions for window translation.
    ----------------------------
    Inputs/Outputs 
    ----------------------------
    Input:
        N: 状态维度（Fourier 模式数 / number of modes）
    Output:
        A: (N, N)
        B: (N, 1)
    ----------------------------
    Complexity / 复杂度
    ----------------------------
    双重 for 循环构造 A，复杂度 O(N^2)。
    ----------------------------
    Performance & modeling insight / 性能与建模讨论
    ----------------------------
    - FouT is naturally suited to periodic/frequency-rich signals.
    - 但 window translation 的边界交互会带来稠密项，数值稳定性与离散化方式会影响训练（尤其大 N）。
    """
    # Initialize A, B.
    A = torch.zeros(N, N)          # A: (N, N)
    B = torch.zeros(N, 1)          # B: (N, 1)

    # n, k index vectors (currently not used later except conceptual; loop uses n_idx/k_idx)
    n = torch.arange(N)
    k = torch.arange(N)

    # --- 生成 B 矩阵 ---
    # B definition matches the translated Fourier basis construction used in S4-style formulations.
    # B[0] = 2, B[odd] = 2*sqrt(2), others 0
    #
    # Intuition:
    #   B defines how input feeds into Fourier modes.
    B[0, 0] = 2.0
    B[1::2, 0] = 2.0 * np.sqrt(2.0)   # odd indices -> 2√2

    # --- 生成 A 矩阵 ---
    for n_idx in range(N):
        for k_idx in range(N):

            # Condition 1: n=k=0
            # DC mode self-dynamics 
            if n_idx == 0 and k_idx == 0:
                A[n_idx, k_idx] = -2.0

            # Condition 2: n=0, k odd
            # DC mode interacts with odd modes 
            elif n_idx == 0 and k_idx % 2 == 1:
                A[n_idx, k_idx] = -2.0 * np.sqrt(2.0)

            # Condition 3: k=0, n odd
            # Symmetric interaction: odd modes with DC 
            elif k_idx == 0 and n_idx % 2 == 1:
                A[n_idx, k_idx] = -2.0 * np.sqrt(2.0)

            # Condition 4: Both n, k are odd
            # Dense interaction among odd modes
            elif n_idx % 2 == 1 and k_idx % 2 == 1:
                A[n_idx, k_idx] = -4.0

            # Condition 5: n-k=1, k odd  (n even, k=n-1)
            # Rotation term: sin' -> cos; adjacent coupling with +2πk
         
            elif (n_idx - k_idx == 1) and (k_idx % 2 == 1):
                A[n_idx, k_idx] = 2.0 * np.pi * k_idx

            # Condition 6: k-n=1, n odd (k even, n=k-1)
            # Rotation term: cos' -> -sin; adjacent coupling with -2πn
		        elif (k_idx - n_idx == 1) and (n_idx % 2 == 1):
                A[n_idx, k_idx] = -2.0 * np.pi * n_idx

            # Otherwise remains 0.

    return A, B

```

```python

def get_general_frequencies(N):
    """
    生成对称的频率列表: [0, 1, -1, 2, -2, ...]
    """
    k_list = [0.0] 
    for i in range(1, (N + 1) // 2):
        k_list.append(float(i))
        k_list.append(float(-i))
    
    # 如果 N 是偶数，上面的循环会生成 N-1 个数，需要补最后一个
    return torch.tensor(k_list[:N])
    
def get_hippo_fout_complex(N):
    """
    FouT的复数域实现
    对应公式: dc/dt = (D - 1*1^T)c + 1*f(t)
    """
    # 1. 定义频率 k
    k = get_general_frequencies(N)
    
    # 2. 构造旋转项 D (Diagonal)
    # A_rot = diag(i * 2 * pi * k)
    # 1j 是 Python 中的虚数单位 i
    D = torch.diag(1j * 2 * np.pi * k)
    
    # 3. 构造边界反馈项 (Dense Interaction)
    # A_fb = 1 * 1^T 
    ones = torch.ones(N, 1)
    A_fb = ones @ ones.t()
    
    # 4. 组合得到 A
    # A = D - A_fb
    A = D - A_fb.to(torch.cfloat)
    
    # 5. 构造 B
    # B = 1
    B = ones.to(torch.cfloat)
    
    return A, B
```

decoder = [LinearRegression, Ridge, mlp]

```python
def train_decoder(X_train, Y_train, X_test, Y_test, decoder_type):
    """
    Compare three decoder choices
    ------------------------------------------------------------
    典型用法：先用 HiPPO/SSM/RNN 得到状态特征 X，再用 decoder 从 X 预测目标 Y。
    ------------------------------------------------------------
    Inputs
    ------------------------------------------------------------
    X_train: array-like, shape (n_train, d)
        - n_train: 训练样本数（通常是时间步数或时间步展开后的样本）
        - d: 特征维度（例如 HiPPO 的状态维 N，或 N 的某种拼接/统计）
        - sklearn expects 2D input: (num_samples, num_features)

    Y_train: array-like, shape (n_train, y_dim) or (n_train,)
 
    X_test:  shape (n_test, d)
    Y_test:  shape (n_test, y_dim) or (n_test,)

    decoder_type: str in {'linear', 'ridge', 'mlp'}
        - 'linear': OLS 普通最小二乘线性回归（无正则）
        - 'ridge' : L2 正则线性回归（数值更稳，通常更推荐）
        - 'mlp'   : 非线性 MLP 回归器（更强但更慢，更易过拟合/更依赖数据量）

    ------------------------------------------------------------
    Outputs
    ------------------------------------------------------------
    pred: array-like
        - 对 X_test 的预测结果
        - 维度通常为 (n_test, 1) 或 (n_test,)（取决于 sklearn 模型输出）

    score: float
        - sklearn 的 score 对回归任务默认是 R^2 (coefficient of determination)
    ------------------------------------------------------------
    三种 decoder 的差异
    ------------------------------------------------------------
    1) OLS LinearRegression（无正则）
       - 优点：训练快、可解释性强，是最基础的线性 probe
       - 缺点：当特征维度 d 大、样本数相对少或特征共线性强时，容易不稳定/泛化差

    2) Ridge（线性 + L2 正则）
       - 优点：在高维/共线特征下更稳，能显著改善泛化与数值稳定性
       - 缺点：仍是线性模型，无法捕捉强非线性关系

    3) MLPRegressor（非线性）
       - 优点：能拟合非线性映射，理论上更强（如果特征里存在可被非线性提取的结构）
       - 缺点：训练成本明显更高；对超参、初始化、数据规模敏感；更容易过拟合；结果方差更大
    """
    if decoder_type == 'linear':
        # 1) OLS: 无正则线性回归
        model = LinearRegression()
        model.fit(X_train, Y_train)                 # fits mapping from d-dim features to target
        score = model.score(X_test, Y_test)         # default regression metric: R^2
        pred = model.predict(X_test)                # prediction shape depends on Y_train shape
        
    elif decoder_type == 'ridge':
        # 2) Ridge: 线性 + L2 正则
        # alpha=1.0 
        # alpha controls strength of L2 penalty; higher alpha => more shrinkage, often more stable.
        model = Ridge(alpha=1.0)
        model.fit(X_train, Y_train)
        score = model.score(X_test, Y_test)         # R^2
        pred = model.predict(X_test)
        
    elif decoder_type == 'mlp':
        model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', 
                             solver='adam', max_iter=500, random_state=42)

        model.fit(X_train, Y_train.ravel())

        # score 仍是 R^2
        score = model.score(X_test, Y_test.ravel())

        # predict 输出通常是 (n_test,)；reshape 到 (n_test,1) 方便统一处理
        pred = model.predict(X_test).reshape(-1, 1)
       
    return pred, score

```

discretisation = [forward_euler, Bilinear, backward_euler]

```python
def discretize_gbt(A, B, dt, alpha):
    """
    Generalized Bilinear Transform (GBT)
    ------------------------------------------------------------
    Background
    ------------------------------------------------------------
    This discretizes a continuous-time linear system:
        c'(t) = A c(t) + B u(t)
    into a discrete-time recursion:
        c_{t+1} = A_bar c_t + B_bar u_t
    ------------------------------------------------------------
    Unified GBT form
    ------------------------------------------------------------
    alpha=0   -> Forward Euler (explicit)
    alpha=0.5 -> Bilinear/Tustin (semi-implicit, good frequency behavior)
    alpha=1   -> Backward Euler (implicit, more stable, more damping)
    ------------------------------------------------------------
    输入输出 / Inputs & Outputs (Shapes)
    ------------------------------------------------------------
    Inputs:
        A: (N, N)  continuous-time state matrix
        B: (N, 1)  input matrix (single-input case)
        dt: float  time step size
        alpha: float in [0,1]  

    Outputs:
        A_bar: (N, N) discrete transition matrix
        B_bar: (N, 1) discrete input matrix
    ------------------------------------------------------------
    Key stability intuition
    ------------------------------------------------------------
    Stability is mainly governed by the spectral radius of A_bar:
        rho(A_bar) < 1  stable
        rho(A_bar) > 1  explosive

    在 HiPPO 里，A 往往具有较强的负对角/下三角结构，但离散化不当 + dt 过大仍可能导致发散。
    ------------------------------------------------------------
    Key parameter
    ------------------------------------------------------------
    dt:
      - dt 越大：离散误差更大、也更容易出现数值不稳定（尤其 forward euler）
      - dt 越小：更接近连续系统，但计算步数可能更多（同长度下不变，只是稳定性更好）

    alpha:
      - 0: explicit, cheapest, smallest stability region, most likely to explode
      - 0.5: good stability + frequency response (common in SSM)
      - 1: most stable but more damping, can smear high-frequency details
    """
    # N: state dimension
    N = A.shape[0]

    # Identity matrix I (N,N)
    I = torch.eye(N)

    # 离散化公式GBT：
    # (I - alpha*dt*A) * c_{t+1} = (I + (1-alpha)*dt*A) * c_t + dt*B*u
    #
    # term_left  : (N,N) 左侧矩阵，若可逆才能求出显式的 A_bar, B_bar
    # term_right : (N,N) 右侧矩阵
    term_left = I - alpha * dt * A  # (N,N)

    try:
        # Inversion is O(N^3), but done once at init (offline cost).
        term_left_inv = torch.linalg.inv(term_left)
    except RuntimeError:
        # 如果 term_left 奇异：通常是 dt 太大或 A 的谱导致 I - alpha dt A 不可逆
        print(f"Warning: Singular matrix for alpha={alpha}")
        return torch.zeros_like(A), torch.zeros_like(B)

    term_right = I + (1 - alpha) * dt * A  # (N,N)

    # 得到离散系统矩阵：
    # A_bar = (I - alpha dt A)^{-1} (I + (1-alpha) dt A)
    # B_bar = (I - alpha dt A)^{-1} (dt B)
    # Shapes:
    #   A_bar: (N,N)
    #   B_bar: (N,1)
    A_bar = term_left_inv @ term_right
    B_bar = term_left_inv @ (dt * B)

    return A_bar, B_bar

# 三种离散化选择：alpha 对应 GBT family
alphas = [0.0, 0.5, 1.0]
alpha_names = {0.0: 'Forward Euler (Explodes)', 0.5: 'Bilinear (Tustin)', 1.0: 'Backward Euler (Damped)'}

```

---

第一部分不使用官方代码，只通过论文中手写初始化不同measure family的矩阵+离散化（这里写了lagt和legs）

Comparison between 4 measure families (legs, lagt, legt, fourier)[N=64]

![HiPPO_Measures_Comparison.png](HiPPO_Experiment/HiPPO_Measures_Comparison.png)

### **(1) HiPPO-LegS (Scaled Legendre)**

- 视觉特征：严格的下三角矩阵 (Lower Triangular)。颜色呈渐变状，$n, k$ 越大数值负得越多。
- 数学含义：
    - 下三角性质：意味着 $c_n$ 的更新只依赖于 $c_0, \dots, c_n$。高阶系数是由低阶系数演化而来的，但未来的高阶项不会反过来影响低阶项。
    - 物理意义：对应 "History: All $[0, t]$"。是一个变长窗口。随着时间 $t$ 增加，窗口不断拉长，为了保持多项式的正交性，系数必须进行特定的缩放（Scaled）。

### **(3) HiPPO-LegT (Translated Legendre / Window)**

- 视觉特征：
    - 非下三角：上三角部分也有很深的颜色（红/蓝）。
    - 棋盘格纹理 (Checkerboard)：红蓝交替非常明显。
- 数学含义：
    - Dense Coupling：上三角非零意味着低阶系数 $c_{low}$ 的更新竟然依赖于高阶系数 $c_{high}$。
    - 物理意义：对应 "History: Sliding Window $[t-w, t]$"。是一个硬窗口。
    - Insight：当数据从窗口的左边缘“掉出去”时，它会对窗口内所有频率的系数产生冲击。为了精确地“切断”最老的那部分信息，各个阶数之间必须进行复杂的相互抵消（这就是为什么矩阵是满的）。棋盘格纹理来自推导中的 $(-1)^{n-k}$ 项。

### **(2) HiPPO-LagT (Translated Laguerre)**

- 视觉特征：颜色非常单调（几乎全灰/浅蓝），且也是下三角。
- 数学含义：
    - 恒定值：下三角部分全是 $-1$。这是 Laguerre 多项式导数性质的直接体现 ($\frac{d}{dx} L_n(x) = -\sum_{i=0}^{n-1} L_i(x)$)。
    - 物理意义：对应 "History: Exp Decay"。这是一种软窗口。旧信息以指数速率衰减。它的结构最简单，计算最快，但捕捉长距离依赖的能力通常不如 LegS（因为它主动遗忘）。
    
    ### **(4) HiPPO-FouT (Translated Fourier)**
    
    - 视觉特征：稀疏且有规律。
        - 对角主导主要集中在对角线附近。
        - 点状分布：呈现出离散的“红点”和“蓝点”。
    - 数学含义：
        - 频率旋转：对角线附近的项对应傅里叶变换中的 $e^{i\omega t}$ 旋转操作。
        - 边界交互：那些离散的点（奇数行列的非零项）是用来处理非周期信号在进入周期性傅里叶基时的边界效应的。
        - S4 的前身：这种结构非常接近后续 S4 模型使用的 Normal Plus Low-Rank (NPLR) 形式，适合处理震荡信号。

<aside>
🖍️

- LegS (Scaled): 呈现严格的下三角结构，反映了其层级化的特征提取过程。这种结构支持了对全历史 $[0, t]$ 的尺度不变性记忆。
- LagT (Translated): 表现为均匀的下三角矩阵 ($A_{nk}=-1$)，对应简单的指数衰减机制，适合关注近期历史的任务。
- LegT (Window): 展现出独特的全矩阵（稠密）特征与棋盘格纹理。这揭示了滑动窗口机制的数学代价——为了将旧信息移出窗口，所有阶数的多项式必须进行复杂的耦合交互。
- FouT (Fourier): 呈现出高度稀疏的对角结构。与多项式基不同，FouT 通过旋转机制（Rotation）来保持记忆，这使其在处理周期性或震荡信号时具有独特的优势，并为后续 S4 模型奠定了基础。"
</aside>

---

当用如下的setting测试4个measure的lag reconstruction能力时（input_signal = white noise; length = 3000; dt = 0.001, N=256, lag=300）：legs和lagt都崩了；

```python
def run_reconstruction_task(N, measure, lag, dt=0.001):
    """
    运行单次重建实验
    Args:
        lag: 我们要重建多少步之前的历史 (Time Lag)
    Returns:
        Y_target: 真实历史值
        Y_pred: 模型预测值
        score: R^2 分数
    """
    print(f"Running: N={N}, Measure={measure}, Lag={lag}...")
    
    # 1. 准备数据
    length = 3000
    inputs = generate_signal(length)
    
    # 2. 初始化模型
    cell = HiPPOCell(N=N, dt=dt, measure=measure)
    
    # 3. 前向传播 (Encoding)
    with torch.no_grad():
        # states shape: (1, Length, N)
        states = cell(inputs)
    
    # 转换数据格式供 sklearn 使用
    X_all = states.squeeze(0).numpy() # (Length, N)
    Y_all = inputs.squeeze(0).numpy() # (Length, 1)
    
    # 4. 构建 Dataset (Shifted Target)
    # 任务: 用 X[t] 预测 Y[t - lag]
    # 所以 X 从 index 'lag' 开始，Y 到 'length - lag' 结束
    X_train = X_all[lag:, :]
    Y_target = Y_all[:-lag, :]
    
    # 5. Decoder
    # 使用前 80% 数据训练，后 20% 测试
    split = int(len(X_train) * 0.8)
    
    reg = Ridge(alpha=1.0).fit(X_train[:split], Y_target[:split])
    
    # 在测试集上预测
    Y_pred = reg.predict(X_train)
    score = reg.score(X_train[split:], Y_target[split:])
    
    print(f"  -> Test R^2 Score: {score:.4f}")
    return Y_target, Y_pred, score
```

分析原因可能为Linear Regression的Decoder在step size过于小的情况下（dt=0.001）出现严重的**多重共线性 (Multicollinearity)**，使得无正则化的线性解码器 (OLS) 数值爆炸。

有两种解决方法：1. 适当增加step size; 2. 把Decoder加正则化，即把Linear换成Ridge；

![Exp_Measure_Comparison_4_measures.png](HiPPO_Experiment/Exp_Measure_Comparison_4_measures.png)

![image.png](HiPPO_Experiment/image.png)

### 1. FouT (Fourier)

- 原因：Nyquist Theorem。
- 分析：
    - FouT 的核心是旋转。当 $N=256$ 时，高阶系数旋转得非常快（频率很高）。
    - `dt=0.001` (1000Hz 采样)，采样率足够高了，能捕捉到高频旋转了，所以 FouT 正常工作了 ($R^2=0.34$，起码是正的)。

### 3. 为什么 LegT (Legendre Translated) 依然稳定？

- 原因一：Task Alignment
    - 任务是："重建 300 步之前的信号" (Lag Reconstruction)。
    - LegT 的物理含义：滑动窗口 (Sliding Window)。把过去的信息原封不动地保留在窗口里 $[t-W, t]$。
    - LegS 的物理含义：它把 $[0, t]$ 的所有历史压缩在一起。时间越久，分辨率越低（$1/t$ 衰减）。
    - 结论：对于“找回具体的某一段过去”这个任务，滑动窗口 (LegT) 天然比 压缩机 (LegS) 强。
- 原因二：正交性更强 (Robustness)
    - LegT 即使在 $dt$ 很小的时候，由于其矩阵结构是稠密的（Dense Interaction）且模拟的是移位操作，其产生的状态向量在不同阶数 $n$ 之间的差异性 (Diversity) 通常比 LegS 更好，不那么容易导致线性回归过拟合。

### 2. LegS 和 LagT ？ (R^2 = -9e8)

大概率是解码器 (Linear Regression) 坏了。典型的多重共线性 (Multicollinearity) 问题。

- 现象：LegS 和 LagT 的 $R^2$ 变成了巨大的负数。
- 原因：`dt=0.001` 太小了。
- 物理过程：
    - LegS 和 LagT 是平滑的积分器。
    - 把步长设为极小的 `0.001` 时，模型的状态 $c_t$ 和 $c_{t+1}$ 几乎一模一样（变化极其微小）。
    - 这导致特征矩阵 $X$（即 `states`）的列与列之间、行与行之间高度相关 (Highly Correlated)。
    - 后果：普通的 `LinearRegression` (最小二乘法 OLS) 在面对这种高度相关的特征时，需要求逆矩阵 $(X^T X)^{-1}$。因为相关性太高，行列式接近 0，求逆后数值爆炸。
    - 于是，解码器学出了一些巨大的权重（比如 $10^{9}$），在测试集上一算，误差会很大。
    
    验证方法：把 `sklearn.linear_model.LinearRegression` 换成 `sklearn.linear_model.Ridge` (岭回归)。Ridge 加了正则化，专门解决这种共线性爆炸的问题。
    

---

遵照建议，我把解码器换成了Ridge,

在相同的setting下，可以看到4个measure都恢复了正常

dt=0.001

![image.png](HiPPO_Experiment/image%201.png)

![Exp_Effect_of_N_4_measures_ridge.png](HiPPO_Experiment/Exp_Effect_of_N_4_measures_ridge.png)

dt=0.01

![image.png](HiPPO_Experiment/image%202.png)

![Exp_Effect_of_N_4_measures_ridge_0.01.png](HiPPO_Experiment/Exp_Effect_of_N_4_measures_ridge_0.01.png)

Comparison between 4 measures: input_signal = white noise; length = 3000; lag=[100, 300]; decoder=ridge(\alpha=1.0), dt = [0.01, 0.001]

```python
print("\n=== Experiment: LegS vs LagT vs LegT vs Fout (The Measure) ===")
results_M = {}
lag_test = 100
for m in ['legs', 'lagt', 'legt', 'fout']:
		data = run_reconstruction_task(N=256, measure=m, lag=lag_test)
		results_M[f"{m.upper()}"] = data
plot_comparison(results_M, "Measure_Comparison")
```

ep01: lag=100, dt=0.01

![Exp_Measure_Comparison_4_measures_ridge_0.01_lag100.png](HiPPO_Experiment/Exp_Measure_Comparison_4_measures_ridge_0.01_lag100.png)

![image.png](HiPPO_Experiment/2fff0bd1-965a-4ee8-91c8-124b38494a6d.png)

ep02: lag=300, dt=0.01

![Exp_Measure_Comparison_4_measures_ridge_0.01_lag300.png](HiPPO_Experiment/Exp_Measure_Comparison_4_measures_ridge_0.01_lag300.png)

![image.png](HiPPO_Experiment/c06775e0-b809-4ed0-b88d-68836d8fd665.png)

ep03: lag=100, dt=0.001

![Exp_Measure_Comparison_4_measures_ridge_0.001_lag100.png](HiPPO_Experiment/Exp_Measure_Comparison_4_measures_ridge_0.001_lag100.png)

![image.png](HiPPO_Experiment/image%203.png)

---

但这里又有一个问题，我们发现这一系列的实验中，legt一直outperform legs；这在之前的实验中是正常的，因为我们的lag test设置成了300；

inspired by this, 我们接下来做3个实验：

1. 对比多种decoder的性能；
2. 既然我们现在所有的任务都是滑动窗口，有没有任务可以让legs和lagt这种全局机制outperform的任务；
3. 对比多种不同的离散化方法（以legs为基础）；

第一个任务：

measure=legs, lag=100, length=2000, input signal=white noise, N=64, 128, 256的情况下三个decoder(linear, ridge, mlp)分别的表现

关于进一步decoder的参数（如ridge下的\alpha和mlp中的param并没有做网格调参）

```python
def run_decoder_comparison():
    print("\n=== Experiment 1: Decoder Comparison (Linear vs Ridge vs MLP) ===")
    N = 256 
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

```

下面的ep说明了低维时mlp极容易产生幻觉，而用linear/ridge更适合（关于linear和ridge的区别上面已经有extreme case讨论）；

![Exp_Decoder_Comparison_64.png](HiPPO_Experiment/Exp_Decoder_Comparison_64.png)

![Exp_Decoder_Comparison_128.png](HiPPO_Experiment/Exp_Decoder_Comparison_128.png)

![Exp_Decoder_Comparison_256.png](HiPPO_Experiment/Exp_Decoder_Comparison_256.png)

---

第二个任务：什么任务更适合用legs?(legs vs legt)

input_signal=slow_triangle_wave, length=3000, lag=2000, measure = [legt, legs, lagt], N = 256;

这个实验验证了legt在滑动窗口下的局限性；

```python
def generate_slow_triangle_wave(length, period):
    """
    生成一个周期极长的三角波。
    length = 3000, period = 600;
    LegT 的窗口 (100步) 远小于半周期 (300步)。
    在窗口内，信号看起来就是一条直线 (局部不可区分)，
    只有拥有全局记忆(LegS)才能通过记住很久以前的拐点来定位相位。
    """
    t = torch.arange(length).float()
    # 生成三角波: 绝对值锯齿波的变换
    # 周期归一化 -> 0~1 -> 0~2 -> -1~1
    phase = (t % period) / period
    signal = 2 * torch.abs(2 * (phase - 0.5)) - 1
    
    return signal.view(1, length, 1)
```

```python
results_M = {}
lag_test = 2000
for m in ['legs', 'legt', 'lagt']:
	data = run_reconstruction_task(N=256, measure=m, lag=lag_test)
  results_M[f"{m.upper()}"] = data
plot_comparison(results_M, "Measure_Comparison")
```

![legtlegs.png](HiPPO_Experiment/legtlegs.png)

![image.png](HiPPO_Experiment/image%204.png)

![legtlegslagt.png](HiPPO_Experiment/legtlegslagt.png)

![image.png](HiPPO_Experiment/image%205.png)

第三个任务：对比不同的离散化方法(Forward Euler, Bilinear, Backward Euler)在measure=legs, lag=100, input_signal=white_noise, length=2000, N=[64,128,256]下的重建效果；Bilinear效果最好，我们选择Bilinear among 3是对的；

![gbt_lag_64.png](HiPPO_Experiment/gbt_lag_64.png)

![gbt_lag_128.png](HiPPO_Experiment/gbt_lag_128.png)

![gbt_lag_256.png](HiPPO_Experiment/gbt_lag_256.png)

---

现在我们选择linear, bilinear decoder, legs, 改变input function和N；

run_exp_input.py: Different input function f (white noise, mixed_sine, step function)

N = 256, 128, 64; lag=300

```python
results_M = {}
lag_test = 100
for m in ['legs', 'lagt', 'legt', 'fout']:
	data = run_reconstruction_task(N=256, measure=m, lag=lag_test)
	results_M[f"{m.upper()}"] = data
plot_comparison(results_M, "Measure_Comparison")
```

![Exp_Signal_Type_Comparison_subplots_256.png](HiPPO_Experiment/Exp_Signal_Type_Comparison_subplots_256.png)

![Exp_Signal_Type_Comparison_subplots_128.png](HiPPO_Experiment/Exp_Signal_Type_Comparison_subplots_128.png)

![Exp_Signal_Type_Comparison_subplots.png](HiPPO_Experiment/Exp_Signal_Type_Comparison_subplots.png)

run_exp.py: Effect of N (Memory Capacity)

```python
results_N = {}
for n in [16, 64, 256]:
	data = run_reconstruction_task(N=n, measure='legs', lag=100)
	results_N[f"N={n}"] = data
	plot_comparison(results_N, "Effect_of_N")
```

![Exp_Effect_of_N.png](HiPPO_Experiment/Exp_Effect_of_N.png)

![image.png](HiPPO_Experiment/image%206.png)

---

step_size comparison(White noise; N=[64,128,256]; dt=[1, 0.1, 0.01, 0.001])

选择white noise的原因：mixed_sine和step都太简单了，重建能力差别不大；

选择N=256的原因：在N=[128,64]时完全没有重建white noise的能力；在N=256时，dt=0.01时可以完美重建（R^2=1）；

```python
fixed_N = 64           
fixed_signal = 'white_noise' 
fixed_lag = 100         
dt_list = [1.0, 0.1, 0.01, 0.001]
results_dt = {}
for dt in dt_list:
        print(f"Running dt={dt}...")
        
        # 1. 生成数据
        inputs = generate_signal(length=3000, type=fixed_signal)
        
        # 2. 初始化模型 (传入当前的 dt)
        cell = HiPPOCell(N=fixed_N, dt=dt, measure='legs')
        
        # 3. 前向传播
        with torch.no_grad():
            states = cell(inputs)
        
        # 4. 线性回归解码
        X_all = states.squeeze(0).numpy()
        Y_all = inputs.squeeze(0).numpy()
        
        # 构建 Lag 任务
        X_train = X_all[fixed_lag:, :]
        Y_target = Y_all[:-fixed_lag, :]
        
        split = int(len(X_train) * 0.8)
        reg = LinearRegression().fit(X_train[:split], Y_target[:split])
        Y_pred = reg.predict(X_train)
        score = reg.score(X_train[split:], Y_target[split:])
        
        # 存结果，Key 是 dt 的值
        results_dt[f"dt={dt}"] = (Y_target, Y_pred, score)
        print(f"  -> R^2: {score:.4f}")

    # 画图
    plot_comparison(results_dt, "Step_Size_Comparison")   
    
```

![Exp_Step_Size_Comparison_white_noise_256.png](HiPPO_Experiment/Exp_Step_Size_Comparison_white_noise_256.png)

N = 256

![image.png](HiPPO_Experiment/image%207.png)

![Exp_Step_Size_Comparison_white_noise_128.png](HiPPO_Experiment/Exp_Step_Size_Comparison_white_noise_128.png)

N = 128

![image.png](HiPPO_Experiment/image%208.png)

![Exp_Step_Size_Comparison_white_noise_64.png](HiPPO_Experiment/Exp_Step_Size_Comparison_white_noise_64.png)

N = 64

![image.png](HiPPO_Experiment/800975b6-97df-4d99-bcfd-01f8f4609fb8.png)

---

![image.png](HiPPO_Experiment/53aaf5be-3e65-4e66-bdb6-f415a2afde42.png)

![Exp_Step_Size_Comparison_ridge_white_noise_256.png](HiPPO_Experiment/Exp_Step_Size_Comparison_ridge_white_noise_256.png)

接下来对比LSTM和legs

run_copying_task.py: copying task between lstm and legs

![Exp_CopyingTask_Final.png](HiPPO_Experiment/Exp_CopyingTask_Final.png)

run_lstm_baseline.py: lstm baseline (compared to hippo)-在lag为300的时候LSTM失效了

![LSTM_Lag300.png](HiPPO_Experiment/LSTM_Lag300.png)

![image.png](HiPPO_Experiment/56ac10bd-18a6-4ac6-8f63-6c055fcfb475.png)

![LSTM_Lag50.png](HiPPO_Experiment/LSTM_Lag50.png)

![image.png](HiPPO_Experiment/9bb6f842-720a-4355-9699-a57a9e66c997.png)

---

run_mnist.py: mnist reconstruction-把像素拉成序列，尝试用hippo重建（L=784），证明c(t)包含了足以重建原始信号的完整信息（和论文中不同在于论文中是打乱的序列，是需要学习的，这里仅仅是按顺序把像素打平测试c(t)的记忆力）；

![MNIST_Recon.png](HiPPO_Experiment/MNIST_Recon.png)

---

run_robustness.py: different noise level (N=64 vs N=128)【noise level不同】，和LSTM对比

```python
def run_robustness_experiment():
    print("=== Robustness Experiment: Denoising Sine Wave ==="
    seq_len = 1000
    t = torch.linspace(0, 40, seq_len)
    
    # 基础信号：纯净的正弦波
    clean_signal = torch.sin(t).view(1, seq_len, 1)
    
    # 噪声等级
    noise_levels = [0.1, 0.5, 1.0]
    
    # 绘图准备
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    # Lag 设定 (任务：从当前含噪输入中，恢复 lag 步之前的纯净信号)
    lag = 10 
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
        # Input: X_full[:, :-lag, :] -> 用 [t] 预测 [t-lag]
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
```

![Exp_Robustness.png](HiPPO_Experiment/Exp_Robustness.png)

![image.png](HiPPO_Experiment/304a1d42-00ba-462c-bcab-1c6d8a4d3d44.png)

![Exp_Robustness_128.png](HiPPO_Experiment/Exp_Robustness_128.png)

![image.png](HiPPO_Experiment/image%209.png)

---

接下来看计算复杂度

run_benchmark_comparison.py：和手写初始化的矩阵相比，官方论文通过DPLR把A变成了对角矩阵+Low Rank Matrix的和，使复杂度从O(N^2)降低到了O(N)，这个plot展示了效率对比

![Complexity_Comparison_Fixed.png](HiPPO_Experiment/Complexity_Comparison_Fixed.png)

![image.png](HiPPO_Experiment/5813f00e-b35c-49e7-835f-7fcc88ecfc3e.png)

run_benchmark.py：不同维度的N的计算速度O(N^2)-验证了当N大到一定程度计算效率会变低

![Benchmark_Time_vs_N.png](HiPPO_Experiment/Benchmark_Time_vs_N.png)

![image.png](HiPPO_Experiment/image%2010.png)

run_equivalence.py: robustness test between O(N) and O(N^2) - 降低计算复杂度是无损的压缩

![Equivalence_Heatmap.png](HiPPO_Experiment/Equivalence_Heatmap.png)

![image.png](HiPPO_Experiment/image%2011.png)

---

官方代码复现：基于

[https://github.com/HazyResearch/hippo-code](https://github.com/HazyResearch/hippo-code)

Experiment: copying task

python [train.py](http://train.py/) runner=pl dataset=copying model.cell=legs train.epochs=50

![image.png](HiPPO_Experiment/image%2012.png)

Experiment: **Char Trajectories - Timescale Robustness**

python [tr.py](http://tr.py/) runner=pl runner.ntrials=1 dataset=ct dataset.timestamp=False dataset.train_ts=1 dataset.eval_ts=1 dataset.train_hz=0.5 dataset.eval_hz=1 dataset.train_uniform=True dataset.eval_uniform=True model.cell=legs model.cell_args.hidden_size=64 train.epochs=30 train.batch_size=100 train.lr=0.001

![char_traj_viz.png](HiPPO_Experiment/char_traj_viz.png)

Experiment: Online Function Approximation

![hippo_math_demo_fixed.png](HiPPO_Experiment/hippo_math_demo_fixed.png)

Experiment Invariant: Using low freq+high freq+step input signal to demonstrate the effect of Ns. 

![hippo_math_demo_comparison.png](HiPPO_Experiment/hippo_math_demo_comparison.png)

有几个没有跑的实验：一个是pMNIST一个长序列建模，都是CPU跑不动的问题；pMNIST我用legS重建（sMNIST）长序列作为替代；