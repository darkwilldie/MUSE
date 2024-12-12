"""
Library for instantiating and running the optimizer for Tangram. The optimizer comes in two flavors,
which correspond to two different classes:
- Mapper: optimizer without filtering (i.e., all single cells are mapped onto space). At the end, the learned mapping
matrix M is returned.
- MapperConstrained: optimizer with filtering (i.e., only a subset of single cells are mapped onto space).
At the end, the learned mapping matrix M and the learned filter F are returned.
"""

import numpy as np
import logging
import torch
from torch.nn.functional import softmax, cosine_similarity
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
import torch.linalg as LA
from tqdm import tqdm


def torch_corrcoef(X, Y):
    X_demean = X - torch.mean(X, dim=0, keepdim=True)
    Y_demean = Y - torch.mean(Y, dim=0, keepdim=True)
    cov_matrix = torch.mm(X_demean.T, Y_demean) / (X.shape[0] - 1)
    X_std = torch.std(X_demean, dim=0)
    Y_std = torch.std(Y_demean, dim=0)
    corr_matrix = cov_matrix / torch.outer(X_std, Y_std)
    return corr_matrix


def torch_cca(X, Y, n_components=2, reg_param=1e-5):
    device = X.device  # 确保计算在同一设备上（GPU或CPU）

    X = X.float()
    Y = Y.float()

    # 标准化和中心化
    X = X - X.mean(dim=0)
    Y = Y - Y.mean(dim=0)

    n = X.size(0) - 1
    cov_xx = torch.mm(X.T, X) / n + reg_param * torch.eye(X.size(1), device=device)
    cov_yy = torch.mm(Y.T, Y) / n + reg_param * torch.eye(Y.size(1), device=device)
    cov_xy = torch.mm(X.T, Y) / n

    # Cholesky分解
    chol_xx = LA.cholesky(cov_xx)
    chol_yy = LA.cholesky(cov_yy)

    # 白化
    whitened_x = LA.solve_triangular(chol_xx, X.T, upper=False).T
    whitened_y = LA.solve_triangular(chol_yy, Y.T, upper=False).T

    # 奇异值分解
    u, _, v = torch.svd(torch.mm(whitened_x.T, whitened_y))

    X_c = torch.mm(whitened_x, u[:, :n_components])
    Y_c = torch.mm(whitened_y, v[:, :n_components])

    # 使用PyTorch计算的相关系数矩阵
    correlations_matrix = torch_corrcoef(X_c, Y_c)

    return X_c, Y_c, correlations_matrix


def calculate_correlations(sc_torch, st_torch):
    # n_features_sc = sc_torch.shape[0]
    # n_features_st = st_torch.shape[0]
    sum_cos_sim1 = 0

    for j in tqdm(range(st_torch.shape[1])):
        # print(f"Processing feature {j}")

        Y = st_torch[:, j, :]  # 现在正确地选择 j 且只处理那一维度

        for i in range(sc_torch.shape[0]):
            X = sc_torch[i, :].reshape(-1, 1)  # 将sc的列转换为列矩阵

            # 假设 torch_cca 函数已经定义，用于执行 CCA
            X_c_torch, Y_c_torch, torch_correlations = torch_cca(X, Y, n_components=1)
            # print("torch_correlations:", torch_correlations.shape)
            sum_cos_sim1 = sum_cos_sim1 + torch_correlations[0, 0]  # 取典型相关系数
        # print(sum_cos_sim1)
    return sum_cos_sim1


class Mapper:
    """
    Allows instantiating and running the optimizer for Tangram, without filtering.
    Once instantiated, the optimizer is run with the 'train' method, which also returns the mapping result.
    """

    def __init__(
        self,
        S,
        G,
        image,
        lambda_g1=1.0,
        lambda_d=0,
        lambda_g2=0,
        lambda_r=0,
        device="cpu",
        random_state=0,
    ):
        """
        Instantiate the Tangram optimizer (without filtering).
        """
        self.S = torch.tensor(S, device=device, dtype=torch.float32)
        self.G = torch.tensor(G, device=device, dtype=torch.float32)
        self.image = torch.tensor(image, device=device, dtype=torch.float32)

        # 使用 torch.cat 在第一个维度上堆叠它们

        self.spot_feature = torch.cat(
            (self.G.unsqueeze(0), self.image.unsqueeze(0)), dim=0
        ).permute(2, 1, 0)

        self.lambda_g1 = lambda_g1
        self.lambda_d = lambda_d
        self.lambda_g2 = lambda_g2
        self.lambda_r = lambda_r
        self.random_state = random_state
        np.random.seed(seed=self.random_state)

        self.M = np.random.normal(0, 1, (S.shape[0], G.shape[0]))
        self.M = torch.tensor(
            self.M, device=device, requires_grad=True, dtype=torch.float32
        )
        # self.M = softmax(self.M, dim=1)

    def _loss_fn(self, S_batch, G, M_block):
        # M_probs = softmax(self.M, dim=1)
        # M_batch = M_probs[start_idx:end_idx]
        # M_batch = softmax(self.M[start_idx:end_idx, :], dim=1)
        # print(M_batch.t().shape,S_batch.shape,G.shape)
        # quit()
        G_pred = torch.matmul(M_block.t(), S_batch)
        gv_term = self.lambda_g1 * calculate_correlations(G_pred, G)
        print(gv_term)
        return -gv_term

    def train(self, num_epochs, learning_rate=0.1, print_each=100, batch_size=8):
        torch.manual_seed(self.random_state)
        optimizer = torch.optim.Adam([self.M], lr=learning_rate)
        print(self.M.shape)
        num_batches = self.S.shape[0] // batch_size
        # s cell 100*50  g spot 40*50
        for epoch in range(num_epochs):
            epoch_loss = 0
            for i in range(num_batches):

                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                S_batch = self.S[start_idx:end_idx]
                # M_block = self.M[start_idx:end_idx]
                M_block = softmax(self.M[start_idx:end_idx], dim=1)

                loss = self._loss_fn(S_batch, self.spot_feature, M_block)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if epoch % print_each == 0:
                print(f"Epoch {epoch}: Loss {epoch_loss / num_batches}")

        with torch.no_grad():
            output = softmax(self.M, dim=1)
            return output.cpu().detach().numpy()
