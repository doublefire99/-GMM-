import numpy as np
import torch

mu = 398600.4418  # km^3/s^2
REarth = 6371.0  
GM = 398600.4418  
DU = REarth  
TU = np.sqrt(REarth**3 / GM)  
VU = DU / TU  
GM_std = GM * TU**2 / (REarth**3)  # 无量纲的引力常数
"""
输入权重tensor.size[1,n,1]
输入均值tensor.size[1,n,6]
输入方差tensor.size[1,n,6,6]

对于每一个成分应用UT变换传播
输出的格式和输入的格式一致
"""
# 两体问题的加速度模型
def two_body_acceleration(r):
    r_norm = np.linalg.norm(r)
    return -GM_std * r / r_norm**3

# 四阶龙格-库塔法（RK4）单步积分
def rk4_step(r, v, dt):
    dt = dt / TU 
    k1_v = two_body_acceleration(r)
    k1_r = v

    k2_v = two_body_acceleration(r + 0.5 * dt * k1_r)
    k2_r = v + 0.5 * dt * k1_v

    k3_v = two_body_acceleration(r + 0.5 * dt * k2_r)
    k3_r = v + 0.5 * dt * k2_v

    k4_v = two_body_acceleration(r + dt * k3_r)
    k4_r = v + dt * k3_v

    r_next = r + (dt / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
    v_next = v + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    return r_next, v_next

# 从高斯分布生成样本（使用 torch.distributions.MultivariateNormal）
def Sample_of_Gussian(mean, cov_diag):
    if cov_diag.ndimension() == 1:  # 如果是1D数组，说明它是对角线元素
        cov_diag = torch.diag(cov_diag)  # 转换为对角矩阵
    epsilon = 1e-6
    cov_diag = cov_diag + torch.eye(cov_diag.size(0)) * epsilon
    # 使用 torch.distributions.MultivariateNormal 来生成样本
    m_dist = torch.distributions.MultivariateNormal(mean, cov_diag)
    samples = m_dist.sample()  # 生成样本
    return samples

def unscented_transform(mean, cov_diag, dt, alpha=1e-3, beta=2, kappa=0):
    n = len(mean)  # 计算状态维度
    lambda_ = alpha**2 * (n + kappa) - n
    
    # 使用 clone().detach() 来避免警告
    mean = mean.clone().detach().to(torch.float64)
    cov_diag = cov_diag.clone().detach().to(torch.float64)
    
    x_true = torch.zeros([2 * n + 1, n], dtype=torch.float64)
    
    for i in range(2 * n + 1):
        x_true[i] = Sample_of_Gussian(mean, cov_diag)
   
    x = torch.mean(x_true, dim=0)  # 使用 torch.mean 计算均值
    P = torch.cov(x_true.T)  # 使用 torch.cov 计算协方差矩阵

    sigma_points = torch.zeros((2 * n + 1, n), dtype=torch.float64)
    sigma_points[0] = x
    sqrt_P = torch.linalg.cholesky((n + lambda_) * P)  # Cholesky分解

    for i in range(n):
        sigma_points[i + 1] = x + sqrt_P[:, i]  
        sigma_points[i + n + 1] = x - sqrt_P[:, i]  

    # 计算均值权重和协方差权重
    W_m = torch.full((2 * n + 1,), 1 / (2 * (n + lambda_)), dtype=torch.float64)
    W_m[0] = lambda_ / (n + lambda_)  
    W_c = torch.full((2 * n + 1,), 1 / (2 * (n + lambda_)), dtype=torch.float64)
    W_c[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)  

    transformed_points = torch.zeros((2 * n + 1, n), dtype=torch.float64)
    for i in range(2 * n + 1):
        sample_point = sigma_points[i]
        r = sample_point[:n//2]
        v = sample_point[n//2:]

        r_new, v_new = rk4_step(r, v, dt)  # 用RK4进行变换
        transformed_points[i] = torch.cat([r_new, v_new], dim=0)

    # 计算变换后的均值和协方差
    y_mean = torch.sum(W_m[:, None] * transformed_points, dim=0)  
    y_cov = torch.zeros((n, n), dtype=torch.float64)  
    for i in range(2 * n + 1):
        diff = transformed_points[i] - y_mean
        y_cov += W_c[i] * torch.outer(diff, diff)

    return y_mean, y_cov


# GMM函数 - 输入：权重、均值和协方差，输出：变换后的均值和协方差

def apply_gmm_ut(weights, means, covariances, dt=60):
    """
    GMM 无迹变换

    参数
    ----
    weights : torch.Tensor
        形状大致类似 (1, K, 1) 或 (K,) 或 (1,)
        在本函数中其实不用到，只是为了接口统一保留。
    means : torch.Tensor
        GMM 各分量的均值，形状大致类似：
            (1, K, n) 或 (K, n) 或 (n,)
    covariances : torch.Tensor
        GMM 各分量的协方差，形状大致类似：
            (1, K, n, n) 或 (K, n, n) 或 (n, n)
    返回
    ----
    transformed_means_tensor : (1, K, n)
    transformed_covs_tensor  : (1, K, n, n)
    """

    # -------- 只整理 means / covariances 的形状，不再用 weights 判分量数 --------
    means = means.squeeze()
    covariances = covariances.squeeze()

    # 可能的情况：
    # 1) 单个分量：means: (n,), cov: (n, n)
    # 2) 多个分量：means: (K, n), cov: (K, n, n)

    if means.dim() == 1:
        # 单一分量
        n = means.shape[0]
        means = means.reshape(1, n)             # (1, n)
        if covariances.dim() == 2:
            covariances = covariances.unsqueeze(0)  # (1, n, n)
        elif covariances.dim() == 3:
            # 已经是 (1, n, n)，保持
            pass
        else:
            raise ValueError(f"covariances 维度不合法: {covariances.shape}")
    elif means.dim() == 2:
        # 多个分量 (K, n)
        K, n = means.shape
        if covariances.dim() == 3:
            if covariances.shape[0] != K:
                raise ValueError(
                    f"means 和 covariances 分量数不一致: {K} vs {covariances.shape[0]}"
                )
        elif covariances.dim() == 2:
            # 只有一个协方差，给所有分量共用（一般不会这样用，但这里做个兜底）
            covariances = covariances.unsqueeze(0).repeat(K, 1, 1)
        else:
            raise ValueError(f"covariances 维度不合法: {covariances.shape}")
    else:
        raise ValueError(f"means 维度不合法: {means.shape}")

    n_components = means.shape[0]   # 用 means 的第一维作为 GMM 成分数

    # -------- 正常做 UT --------
    transformed_means = []
    transformed_covs = []

    for i in range(n_components):
        mean_i = means[i]             # (n,)
        cov_i = covariances[i]        # (n, n)

        transformed_mean, transformed_cov = unscented_transform(mean_i, cov_i, dt)
        transformed_means.append(transformed_mean)
        transformed_covs.append(transformed_cov)

    transformed_means_tensor = torch.stack(transformed_means, dim=0).unsqueeze(0)  # (1, K, n)
    transformed_covs_tensor = torch.stack(transformed_covs, dim=0).unsqueeze(0)    # (1, K, n, n)

    return transformed_means_tensor, transformed_covs_tensor



# # # 示例：GMM权重、均值和协方差
# weights = torch.tensor([[[0.2444],
#          [0.5112],
#          [0.2444]]], dtype=torch.float64)

# means =torch.tensor([[[ 0.1936, -0.0301,  0.1547,  0.1204,  0.1636, -0.1254],
#          [ 0.4224, -0.0630,  0.3235,  0.2518,  0.3422, -0.2622],
#          [ 0.2103, -0.0301,  0.1547,  0.1204,  0.1636, -0.1254]]],
#        dtype=torch.float64)
# covariances = torch.tensor([[[[8.1473e-05, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#            0.0000e+00],
#           [0.0000e+00, 2.4442e-05, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#            0.0000e+00],
#           [0.0000e+00, 0.0000e+00, 2.4442e-05, 0.0000e+00, 0.0000e+00,
#            0.0000e+00],
#           [0.0000e+00, 0.0000e+00, 0.0000e+00, 2.4442e-10, 0.0000e+00,
#            0.0000e+00],
#           [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.4442e-10,
#            0.0000e+00],
#           [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#            2.4442e-10]],

#          [[1.7039e-04, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#            0.0000e+00],
#           [0.0000e+00, 5.1116e-05, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#            0.0000e+00],
#           [0.0000e+00, 0.0000e+00, 5.1116e-05, 0.0000e+00, 0.0000e+00,
#            0.0000e+00],
#           [0.0000e+00, 0.0000e+00, 0.0000e+00, 5.1116e-10, 0.0000e+00,
#            0.0000e+00],
#           [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 5.1116e-10,
#            0.0000e+00],
#           [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#            5.1116e-10]],

#          [[8.1473e-05, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#            0.0000e+00],
#           [0.0000e+00, 2.4442e-05, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#            0.0000e+00],
#           [0.0000e+00, 0.0000e+00, 2.4442e-05, 0.0000e+00, 0.0000e+00,
#            0.0000e+00],
#           [0.0000e+00, 0.0000e+00, 0.0000e+00, 2.4442e-10, 0.0000e+00,
#            0.0000e+00],
#           [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.4442e-10,
#            0.0000e+00],
#           [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#            2.4442e-10]]]], dtype=torch.float64)

# # 时间步长
# dt = 60  # 单位：秒

# # 应用 GMM 无迹变换
# transformed_means, transformed_covs = apply_gmm_ut(weights, means, covariances)
# print(transformed_means.shape)
# print(transformed_covs.shape)
# # 打印变换后的均值和协方差
# for i in range(transformed_means.size(1)):
#     print(f"第 {i+1} 个成分的变换后的均值: {transformed_means[0,i:].shape}")
#     print(f"第 {i+1} 个成分的变换后的协方差: {transformed_covs[0,i:].shape}")
