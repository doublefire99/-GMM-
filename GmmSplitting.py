import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import torch
from unscentde_transform import *
class SplitGussion:
    """
    输入:均值和协方差矩阵

    输出:分裂后的均值,协方差矩阵和对应的权重
    """
    
    def __init__(self, u, P_x):
        
        self.u = u
        self.P_x = P_x
        
        self.mu = 398600.4418  # km^3/s^2
        self.REarth = 6371.0  
        self.GM = 398600.4418  
        self.DU = self.REarth  
        self.TU = np.sqrt(self.REarth**3 / self.GM)  
        self.VU = self.DU / self.TU  
        self.GM_std = self.GM * self.TU**2 / (self.REarth**3)  # 无量纲的引力常数
        
        self.u_std = self.convert_to_standard_units(self.u)

    def convert_to_standard_units(self, u):
        x1, x2, x3, x4, x5, x6 = u
        r = np.array([x1, x2, x3])
        v = np.array([x4, x5, x6])
        r_standard = r / self.DU 
        v_standard = v / self.VU  
        u_std = np.concatenate([r_standard, v_standard])
        return u_std

    def convert_to_original_units(self, u_std):
        x1, x2, x3, x4, x5, x6 = u_std
        r = np.array([x1, x2, x3])
        v = np.array([x4, x5, x6])
        r_standard = r * self.DU  
        v_standard = v * self.VU  
        u = np.concatenate([r_standard, v_standard])
        return u
    
    def convert_to_original_units_tensor(self, u_std_tensor):
        """
        支持两种输入形状：
        - (K, 6)
        - (1, K, 6)
        """
        # 如果是三维（1, K, 6），先挤掉 batch 维
        if u_std_tensor.dim() == 3:
            u_std_tensor = u_std_tensor.squeeze(0)  # -> (K, 6)

        # 现在假定形状是 (K, 6)
        r_std = u_std_tensor[:, :3]   # (K, 3)
        v_std = u_std_tensor[:, 3:]   # (K, 3)

        r_original = r_std * self.DU
        v_original = v_std * self.VU

        u_original = torch.cat((r_original, v_original), dim=-1)  # (K, 6)
        return u_original


    def compute_square_root_and_max_eigenvector(self, P_x):
        P_x = P_x.squeeze() 
        eigvals, eigvecs = np.linalg.eig(P_x)
        eigvals_sqrt = np.sqrt(np.abs(eigvals))  
        eigvals_sqrt_diag = np.diag(eigvals_sqrt)
        S = eigvecs @ eigvals_sqrt_diag @ eigvecs.T
        return S, eigvecs

    def compute_unit_vector(self, S, a):
        S_inv = np.linalg.inv(S)
        a_hat = S_inv @ a / np.linalg.norm(S_inv @ a)
        return a_hat

    def read_split_data(self, N):
        df = pd.read_csv("split.csv", header=None)
        N = int(N)
        row_index = int((N - 1) // 2) - 1
        row = df.iloc[row_index]
        N_value = row[0]  
        std_dev = row[1]  
        means = row[2:N+2].values  
        weights = row[N+2:2*N+2].values
        return std_dev, means, weights

    def compute_means_and_covariances(self, u, µi, S, a_hat, std_dev, weights):
        mu_i_updated_list = []
        P_i_list = []
        weighted_mu = np.zeros_like(u, dtype=np.float64)

        for i in range(len(µi)):
            # 1) 子成分自身的均值：不乘权重
            mu_i_updated = u + µi[i] * (S @ a_hat)

            # 2) 子成分自身的协方差：不乘权重
            I = np.eye(S.shape[0])
            P_i = S @ (I + (std_dev**2 - 1) * np.outer(a_hat, a_hat)) @ S.T

            mu_i_updated_list.append(mu_i_updated)
            P_i_list.append(P_i)

            # 3) 整体加权均值，可以用 weighted_mu
            weighted_mu += weights[i].item() * mu_i_updated

        return mu_i_updated_list, P_i_list, weighted_mu


    def compute_nonlinearity_measure(self, x_bar, a_hat, S, h_tilde):
        S_inv = np.linalg.inv(S)
        norm_term = np.linalg.norm(S_inv @ a_hat) ** -1
        term1 = self.f(x_bar + h_tilde * norm_term * a_hat)
        term2 = self.f(x_bar - h_tilde * norm_term * a_hat)
        term3 = 2 * self.f(x_bar)
        phi = (term1 + term2 - term3)/(2*h_tilde**2)
        phi = np.linalg.norm(phi)
        return phi

    def f(self, x):
        x1, x2, x3, x4, x5, x6 = x 
        r = np.array([x1, x2, x3])  
        v = np.array([x4, x5, x6])  
        r_next, v_next = self.rk4_step_std(r, v)  
        x_next = np.concatenate([r_next, v_next])
        return x_next

    def rk4_step_std(self, r, v, dt=60):
        dt_std = dt / self.TU  
        k1_v = self.two_body_acceleration_std(r)
        k1_r = v
        k2_v = self.two_body_acceleration_std(r + 0.5 * dt_std * k1_r)
        k2_r = v + 0.5 * dt_std * k1_v
        k3_v = self.two_body_acceleration_std(r + 0.5 * dt_std * k2_r)
        k3_r = v + 0.5 * dt_std * k2_v
        k4_v = self.two_body_acceleration_std(r + dt_std * k3_r)
        k4_r = v + dt_std * k3_v
        r_next = r + (dt_std / 6.0) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
        v_next = v + (dt_std / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        return r_next, v_next

    def two_body_acceleration_std(self, r):
        r_norm = np.linalg.norm(r)
        return -self.GM_std * r / r_norm**3
    def merge_component(self, pi, mu, var, alpha=0.01, Thd=0.1, eps=1e-9, max_components=120):
        """
        多轮高斯成分融合：

        输入:
            pi  : (1, K, 1)   每个高斯成分的权重
            mu  : (1, K, n)   每个成分的均值（标准化量纲）
            var : (1, K, n, n) 每个成分的协方差

        规则:
            - 对所有 (i, j) 成分对计算一次“融合高斯”和一个 RKL 指标（不因权重小跳过）
            - 若某对 (i, j) 的 RKL < Thd，则把这一对视为“适合合并”的成分
            - 一轮融合完成后，如果成分数仍 > max_components，就用本轮结果作为新的 GMM，继续下一轮
            - 如果某一轮没有任何成分被合并，就停止（避免死循环）

        输出:
            pi_merged : (K', 1)
            mu_merged : (K', n)
            var_merged: (K', n, n)
        """

        import numpy as np
        import torch

        device = pi.device
        dtype = pi.dtype

        # 去掉 batch 维度，内部都用 2D / 3D
        pi_curr = pi[0]      # (K, 1)
        mu_curr = mu[0]      # (K, n)
        var_curr = var[0]    # (K, n, n)

        max_iter = 5  # 防止极端情况无限循环
        iter_count = 0

        while True:
            K = pi_curr.shape[0]
            n_dim = mu_curr.shape[1]

            # 条件 1：已经小于等于最大成分数
            if K <= max_components:
                break
            # 条件 2：保险停止条件
            if iter_count >= max_iter:
                break

            iter_count += 1

            eye = torch.eye(n_dim, dtype=dtype, device=device)

            # ---- 协方差对称化 + jitter ----
            def safe_sym_pd(S):
                return 0.5 * (S + S.transpose(-1, -2)) + eps * eye

            # ---- 安全 logdet（slogdet，自动处理负行列式）----
            def safe_logdet(S):
                S2 = safe_sym_pd(S)
                sign, logabsdet = torch.slogdet(S2)
                if sign <= 0 or not torch.isfinite(logabsdet):
                    return torch.tensor(float("nan"), dtype=dtype, device=device)
                return logabsdet

            # 先把当前协方差全部修正一遍
            var_fixed = safe_sym_pd(var_curr)  # (K, n, n)

            # 预分配 (i,j) 的融合候选
            pi_m = torch.zeros(K * K, 1, dtype=dtype, device=device)
            mu_m = torch.zeros(K * K, n_dim, dtype=dtype, device=device)
            var_m = torch.zeros(K * K, n_dim, n_dim, dtype=dtype, device=device)

            RKL = np.full((K, K), np.inf, dtype=np.float64)

            # =========================
            # 1. 对所有 (i,j) 计算融合分量和 RKL
            # =========================
            for i in range(K):
                for j in range(K):
                    w_i = pi_curr[i, 0]
                    w_j = pi_curr[j, 0]
                    denom = w_i + w_j

                    # 不跳过任何一对：若 denom 很小，直接当成 0.5 / 0.5
                    if torch.abs(denom) < eps:
                        alpha_ij = torch.tensor(0.5, dtype=dtype, device=device)
                        beta_ij = torch.tensor(0.5, dtype=dtype, device=device)
                    else:
                        alpha_ij = w_i / denom
                        beta_ij = w_j / denom

                    idx = i * K + j

                    # 融合后的总权重（还没归一化）
                    pi_m[idx, 0] = denom

                    mu_i = mu_curr[i, :]
                    mu_j = mu_curr[j, :]

                    # 均值融合
                    mu_mix = alpha_ij * mu_i + beta_ij * mu_j
                    mu_m[idx, :] = mu_mix

                    # 协方差融合: Σ_mix = α Σ_i + β Σ_j + αβ (μ_i - μ_j)(μ_i - μ_j)^T
                    Sigma_i = var_fixed[i, :, :]
                    Sigma_j = var_fixed[j, :, :]

                    diff = (mu_i - mu_j).unsqueeze(-1)         # (n, 1)
                    between = diff @ diff.transpose(-1, -2)    # (n, n)

                    Sigma_mix = (
                        alpha_ij * Sigma_i +
                        beta_ij * Sigma_j +
                        alpha_ij * beta_ij * between
                    )
                    Sigma_mix = safe_sym_pd(Sigma_mix)
                    var_m[idx, :, :] = Sigma_mix

                    # 计算 logdet，如遇数值问题则把 RKL 记成 +inf
                    ld_mix = safe_logdet(Sigma_mix)
                    ld_i = safe_logdet(Sigma_i)
                    ld_j = safe_logdet(Sigma_j)

                    if not (torch.isfinite(ld_mix) and
                            torch.isfinite(ld_i) and
                            torch.isfinite(ld_j)):
                        RKL[i, j] = float("inf")
                    else:
                        ai = float(alpha_ij.item())
                        bj = float(beta_ij.item())
                        # 一个简单稳定的 RKL 度量：
                        # RKL = 0.5 * [ log|Σ_mix| - (α log|Σ_i| + β log|Σ_j|) ]
                        RKL[i, j] = 0.5 * (
                            float(ld_mix.item()) -
                            (ai * float(ld_i.item()) + bj * float(ld_j.item()))
                        )

            # =========================
            # 2. 根据 RKL 和阈值 Thd 选择要合并的成分对
            # =========================
            index_row = []
            index_col = []
            indices_rc = []
            list_pi = []
            list_mu = []
            list_var = []

            for i in range(K):
                for j in range(K):
                    if i == j:
                        continue
                    RKL_ij = RKL[i, j]
                    if np.isfinite(RKL_ij) and RKL_ij < Thd:
                        index_row.append(i)
                        index_col.append(j)
                        indices_rc.append((i, j))

            # 2.1 没有被选中参与合并的成分，原样保留
            for i in range(K):
                if i not in index_row and i not in index_col:
                    list_pi.append(pi_curr[i])
                    list_mu.append(mu_curr[i])
                    list_var.append(var_fixed[i])

            # 2.2 被标记“合并”的成分对，用融合结果替代（只取 i < j 避免重复）
            for (i, j) in indices_rc:
                if i < j:
                    idx = i * K + j
                    list_pi.append(pi_m[idx])
                    list_mu.append(mu_m[idx])
                    list_var.append(var_m[idx])

            # 如果这一轮根本没合并任何成分，就没必要再继续下一轮，防止死循环
            if len(list_pi) == 0:
                break

            # =========================
            # 3. 汇总 + 归一化权重
            # =========================
            pi_new = torch.stack(list_pi, dim=0)      # (K', 1)
            mu_new = torch.stack(list_mu, dim=0)      # (K', n)
            var_new = torch.stack(list_var, dim=0)    # (K', n, n)

            pi_sum = torch.sum(pi_new)
            if torch.abs(pi_sum) < eps:
                # 极限情况：总权重几乎为 0，直接设成均匀分布
                pi_new = torch.full_like(pi_new, 1.0 / pi_new.shape[0])
            else:
                pi_new = pi_new / pi_sum

            # 用本轮结果作为下一轮的输入
            pi_curr, mu_curr, var_curr = pi_new, mu_new, var_new

            # while 循环下一轮会用新的 K = pi_curr.shape[0] 继续判断

        # while 结束时，返回的是二维 / 三维张量：
        # pi_curr : (K_final, 1)
        # mu_curr : (K_final, n)
        # var_curr: (K_final, n, n)
        return pi_curr, mu_curr, var_curr



    
    def run(self):
        """
        执行：高斯分裂 + GMM 无迹变换 + GMM 成分融合

        返回:
            pi_merged:  (1, K, 1)  最终权重
            mu_merged_original: (1, K, 6)  最终均值（已经还原到原始单位）
            var_merged: (1, K, 6, 6)  最终协方差
        """
        import numpy as np
        import torch

        # ---- 1. 使用无量纲状态 u_std 和协方差 P_x ----
        u = self.u_std                       # 6 维无量纲均值
        P_x = self.P_x                       # 6x6 或 1x6x6 协方差
        S, eigvecs = self.compute_square_root_and_max_eigenvector(P_x)

        a_hat_list = []
        N_list = []

        # ---- 2. 对每个特征向量计算非线性度量 φ 和分裂数 N ----
        for i in range(eigvecs.shape[1]):
            a_hat = self.compute_unit_vector(S, eigvecs[:, i])  # 单位方向
            h_tilde = np.sqrt(3.0)
            phi = self.compute_nonlinearity_measure(u, a_hat, S, h_tilde)

            # 保护一下数值问题
            if phi <= 0 or not np.isfinite(phi):
                N_val = 1
            else:
                N_raw = 2 * np.ceil(0.404 * np.log(phi) / 2 - 1) + 7.0
                if not np.isfinite(N_raw):
                    N_val = 1
                else:
                    N_val = int(max(1, min(N_raw, 7)))

            a_hat_list.append(a_hat)
            N_list.append(N_val)

        # 按 N 从大到小排序方向（先沿非线性最强方向分裂）
        order = np.argsort(N_list)[::-1]
        a_hat_list = [a_hat_list[i] for i in order]
        N_list = [N_list[i] for i in order]

        max_components = 120  # 允许的最大成分数（用于限制分裂规模）

        # ---- 3. 如果最大 N 也只有 1：不做分裂，直接单高斯 ----
        if len(N_list) == 0 or N_list[0] <= 1:
            all_means = [u]                     # 只有一个均值
            all_covariances = [np.squeeze(P_x)]
            all_weights = [1.0]
        else:
            # ---------- 3.1 第一个方向的分裂 ----------
            N0 = N_list[0]
            a_hat0 = a_hat_list[0]

            std_dev0, means0, weights0 = self.read_split_data(N0)
            # 这里 compute_means_and_covariances 使用的是「带权」形式
            mu_list0, P_list0, _ = self.compute_means_and_covariances(
                u, means0, S, a_hat0, std_dev0, weights0
            )

            all_means = []
            all_covariances = []
            all_weights = []

            # 把带权的 μ、P 还原为真正的 μ、P，再赋予子权重
            for i, w_rel in enumerate(weights0):
                w_rel = float(w_rel)
                if w_rel <= 0:
                    continue

                # compute_means_and_covariances 返回的是 w_rel * μ_i, w_rel * P_i
                mu_unweighted = mu_list0[i] / w_rel
                P_unweighted = P_list0[i] / w_rel

                all_means.append(mu_unweighted)
                all_covariances.append(P_unweighted)
                all_weights.append(w_rel)  # 初始父权重是 1.0

            # 归一化一次权重
            w_sum = sum(all_weights)
            if w_sum > 0:
                all_weights = [w / w_sum for w in all_weights]

            # ---------- 3.2 后续方向的分裂 ----------
            for k_dir in range(1, len(a_hat_list)):
                N_k = N_list[k_dir]
                a_hat_k = a_hat_list[k_dir]

                if N_k <= 1:
                    # 对这个方向不再分裂
                    continue
                if len(all_means) >= max_components:
                    # 成分数太多就停止进一步分裂
                    break

                new_all_means = []
                new_all_covariances = []
                new_all_weights = []

                # 针对当前所有成分，逐个再沿新的方向分裂
                for i_comp in range(len(all_means)):
                    parent_mean = all_means[i_comp]
                    parent_cov = all_covariances[i_comp]
                    parent_w = all_weights[i_comp]

                    std_dev_k, means_k, weights_k = self.read_split_data(N_k)
                    mu_children, P_children, _ = self.compute_means_and_covariances(
                        parent_mean, means_k, parent_cov, a_hat_k, std_dev_k, weights_k
                    )

                    for j, w_rel_child in enumerate(weights_k):
                        w_rel_child = float(w_rel_child)
                        if w_rel_child <= 0:
                            continue

                        # 同样先把带权的 μ、P 还原
                        mu_child_unweighted = mu_children[j] / w_rel_child
                        P_child_unweighted = P_children[j] / w_rel_child

                        # 全局权重 = 父权重 * 子相对权重
                        w_child_global = parent_w * w_rel_child

                        new_all_means.append(mu_child_unweighted)
                        new_all_covariances.append(P_child_unweighted)
                        new_all_weights.append(w_child_global)

                if len(new_all_means) == 0:
                    # 分裂失败/无有效子成分，则停止
                    break

                all_means = new_all_means
                all_covariances = new_all_covariances
                all_weights = new_all_weights

                # 每一轮分裂后都归一化一次权重
                w_sum = sum(all_weights)
                if w_sum > 0:
                    all_weights = [w / w_sum for w in all_weights]

        # ---- 4. 列表 -> torch 张量 ----
        all_means_np = np.array(all_means, dtype=np.float64)          # (K, 6)
        mu_tensor = torch.tensor(all_means_np, dtype=torch.float64).reshape(1, -1, 6)

        all_weights_np = np.array(all_weights, dtype=np.float64)      # (K,)
        w_sum = all_weights_np.sum()
        if w_sum > 0:
            all_weights_np = all_weights_np / w_sum
        weights_tensor = torch.tensor(all_weights_np, dtype=torch.float64).reshape(1, -1, 1)

        all_cov_np = np.array(all_covariances, dtype=np.float64)      # (K, 6, 6)
        covariances_tensor = torch.tensor(all_cov_np, dtype=torch.float64).reshape(1, -1, 6, 6)

        # ---- 5. GMM 无迹变换 ----
        # 这里假定你已经在文件开头有：
        # from unscentde_transform import apply_gmm_ut
        mu_ut, cov_ut = apply_gmm_ut(weights_tensor, mu_tensor, covariances_tensor)

        # ---- 6. GMM 成分融合（merge_component 内部再做多轮融合 & 限制成分数）----
        pi_merged, mu_merged, var_merged = self.merge_component(weights_tensor, mu_ut, cov_ut)

        # ---- 7. 把均值从无量纲还原到原始单位，并补 batch 维度 ----
        mu_merged_original = self.convert_to_original_units_tensor(mu_merged)  # (K, 6)

        pi_merged = pi_merged.unsqueeze(0)               # (1, K, 1)
        mu_merged_original = mu_merged_original.unsqueeze(0)  # (1, K, 6)
        var_merged = var_merged.unsqueeze(0)             # (1, K, 6, 6)

        return pi_merged, mu_merged_original, var_merged




def main():
    u = np.array([5264.123, -784.807, 4031.808, 3.897, 5.295, -4.057])   
    P_x = np.array([np.diag([1e-3, 1e-4, 1e-4, 1e-9, 1e-9, 1e-9])])

    orbital_mechanics = SplitGussion(u, P_x)
    weights,mean_matrix,covariances= orbital_mechanics.run()

    print(mean_matrix)
    print(mean_matrix.shape)
    print(covariances.shape)
    print(weights.shape)


def run_gmm_propagation(u, P_x, num_steps=2):
    orbital_mechanics = SplitGussion(u, P_x)
    weights, mean_matrix, covariances = orbital_mechanics.run()

    for step in range(num_steps):
        new_weights = []
        new_means = []
        new_covariances = []

        for i in range(weights.shape[1]):
            # 当前第 i 个分量
            mean = mean_matrix[0, i, :]              # (6,)
            covariance = covariances[0, i, :, :].unsqueeze(0)  # (1, 6, 6)
            current_weight = weights[0, i, 0]        # 标量

            orbital_mechanics = SplitGussion(mean.numpy(), covariance.numpy())
            new_weight, new_mean, new_covariance = orbital_mechanics.run()
            # new_weight: (1, K_i, 1)
            # new_mean:  (1, K_i, 6)
            # new_cov:   (1, K_i, 6, 6)

            updated_weight = current_weight * new_weight  # 广播 -> (1, K_i, 1)

            new_weights.append(updated_weight)
            new_means.append(new_mean)
            new_covariances.append(new_covariance)

        # ★ 在成分维度上拼接（dim=1）
        weights = torch.cat(new_weights, dim=1)      # (1, sum K_i, 1)
        mean_matrix = torch.cat(new_means, dim=1)    # (1, sum K_i, 6)
        covariances = torch.cat(new_covariances, dim=1)  # (1, sum K_i, 6, 6)

        # 如果希望每一步后把权重归一化，可以加一句（建议）
        # weights = weights / weights.sum()

    return weights, mean_matrix, covariances


u = np.array([5264.123, -784.807, 4031.808, 3.897, 5.295, -4.057])  # 初始均值
P_x = np.diag([1e-3, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5])  # 初始协方差

# 调用主函数
weights_final, mean_matrix_final, covariances_final = run_gmm_propagation(u, P_x)
print(weights_final)
print(mean_matrix_final)