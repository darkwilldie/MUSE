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
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from os.path import join


def cos_sim(sc_feature, other_feature):
    sc_feature = F.normalize(sc_feature, dim=1)
    other_feature = F.normalize(other_feature, dim=1)
    cos_sim_matrix = torch.matmul(sc_feature, other_feature.T)
    # softmax_output = F.softmax(cos_sim_matrix, dim=1)
    # print(softmax_output)
    return cos_sim_matrix


# 计算pearson相关系数
def torch_corrcoef(X, Y):
    X_demean = X - torch.mean(X, dim=1, keepdim=True)
    Y_demean = Y - torch.mean(Y, dim=1, keepdim=True)
    cov_matrix = torch.sum(X_demean * Y_demean, dim=1) / (X.shape[1] - 1)

    X_std = torch.std(X_demean, dim=1, keepdim=True)
    Y_std = torch.std(Y_demean, dim=1, keepdim=True)

    out = torch.sum(X_std * Y_std, dim=1)
    corr_matrix = cov_matrix / out

    return corr_matrix


def torch_cca(X, Y, n_components=1, reg_param=1e-5):
    # 确保 X 和 Y 是三维矩阵
    if X.ndim != 3:
        raise ValueError(
            "X must be a 3D tensor with shape (batch_size, num_samples, num_features)"
        )
    if Y.ndim != 3:
        raise ValueError(
            "Y must be a 3D tensor with shape (batch_size, num_samples, num_features)"
        )

    device = X.device  # 确保计算在同一设备上（GPU或CPU）

    X = X.float()
    Y = Y.float()

    # 标准化和中心化
    #! 二维计算改为三维，此处的dim应从0->1
    X = X - X.mean(dim=1, keepdim=True)
    Y = Y - Y.mean(dim=1, keepdim=True)

    n = X.size(1) - 1

    cov_xx = torch.matmul(X.transpose(1, 2), X) / n + reg_param * torch.eye(
        X.size(-1), device=device
    )
    cov_yy = torch.matmul(Y.transpose(1, 2), Y) / n + reg_param * torch.eye(
        Y.size(-1), device=device
    )
    cov_xy = torch.matmul(X.transpose(1, 2), Y) / n

    # Cholesky分解
    # assert torch.all(cov_xx >= 0), print(cov_xx)
    chol_xx = LA.cholesky(cov_xx)
    chol_yy = LA.cholesky(cov_yy)

    # 白化
    whitened_x = torch.empty_like(X)
    whitened_y = torch.empty_like(Y)

    whitened_x = LA.solve_triangular(chol_xx, X.transpose(1, 2), upper=False).transpose(
        1, 2
    )
    whitened_y = LA.solve_triangular(chol_yy, Y.transpose(1, 2), upper=False).transpose(
        1, 2
    )

    # 奇异值分解
    u, _, v = torch.svd(torch.matmul(whitened_x.transpose(1, 2), whitened_y))

    X_c = torch.matmul(whitened_x, u[:, :, :n_components])
    Y_c = torch.matmul(whitened_y, v[:, :, :n_components])

    correlations_matrix = torch_corrcoef(X_c, Y_c)

    return X_c, Y_c, correlations_matrix


def calculate_correlations(G_pred, G):
    # cos sim
    # info (st sample, st sample, n)
    # cos_sim_list = []
    # for i in range(G.shape[-1]):
    #     for j in range(G_pred.shape[-1]):
    #         cos_sim_list.append(cos_sim_dig(G_pred[:, :, j], G[:, :, i]).unsqueeze(-1))
    # cos_sim1 = torch.mean(torch.stack(cos_sim_list, dim=-1), dim=-1)
    # cos_sim1 = cos_sim_dig(G_pred[:, :, 0], G[:, :, 0]).unsqueeze(-1)
    cos_sim1 = cosine_similarity(G_pred[:, :, 0], G[:, :, 0]).unsqueeze(-1)
    # CCA计算典型变量的相关性
    # 只计算一个典型变量
    X_c_torch, Y_c_torch, torch_correlations = torch_cca(G_pred, G, n_components=1)
    # print("torch_correlations:\n", torch_correlations)
    assert (
        torch_correlations.shape == cos_sim1.shape
    ), f"torch_correlations: {torch_correlations.shape}, cos_sim1: {cos_sim1.shape}"

    # 将cos sim的符号赋给CCA计算出的相关性
    torch_correlations = torch.sign(cos_sim1) * torch.abs(torch_correlations)

    # torch_correlations = cos_sim1
    # torch_correlations = torch_correlations

    return torch_correlations.mean()


class Mapper:
    """
    Allows instantiating and running the optimizer for Tangram, without filtering.
    Once instantiated, the optimizer is run with the 'train' method, which also returns the mapping result.
    """

    def __init__(
        self,
        S,
        G,
        lambda_g1=1.0,
        lambda_d=0,
        lambda_g2=0,
        lambda_r=0,
        device="cpu",
        random_state=0,
        lr=0.1,
    ):
        """
        Instantiate the Tangram optimizer (without filtering).
        """
        self.lambda_g1 = lambda_g1
        self.lambda_d = lambda_d
        self.lambda_g2 = lambda_g2
        self.lambda_r = lambda_r
        self.random_state = random_state
        # 设置随机种子
        np.random.seed(seed=self.random_state)
        torch.manual_seed(self.random_state)

        self.S = S.detach()
        self.G = G.detach()
        self.spot_feature = self.G

        print("*" * 50)
        print("S:", self.S.shape)
        print("G:", self.G.shape)
        self.M = np.random.normal(0, 1, (self.S.shape[0], self.G.shape[0]))
        self.M = torch.tensor(
            self.M, device=device, requires_grad=True, dtype=torch.float32
        )
        self.optimizer = torch.optim.Adam([self.M], lr=lr)
        # self.M = softmax(self.M, dim=1)

    def _loss_fn(self, S_batch, G_with_image, M_block):
        # G_pred = torch.matmul(M_block.t(), S_batch).unsqueeze(-1)
        G_pred = torch.einsum("ij,ikl->jkl", M_block, S_batch)
        # print("G_pred:", G_pred.shape)
        # G_pred (spot num, gene num, 1)
        # G_with_image (spot num, gene num, 2)
        # 计算预测值和真实值之间的相关性
        #! G_with_image[:,:,:1]代表不用image feature
        # print("G_pred:\n", G_pred)
        # print("G_with_image:\n", G_with_image)
        gv_term_spotwise = self.lambda_g1 * calculate_correlations(G_pred, G_with_image)
        # print("gv_term_spotwise:\n", gv_term_spotwise.item())
        gv_term_genewise = self.lambda_g2 * calculate_correlations(
            G_pred.transpose(0, 1), G_with_image.transpose(0, 1)
        )

        return -gv_term_genewise - gv_term_spotwise

    # def train_one_epoch(self, learning_rate=0.1):

    def train(self, num_epochs, learning_rate=0.1, print_interval=100):

        loss_plot = []

        torch.manual_seed(self.random_state)
        # tqdm
        for epoch in tqdm(range(num_epochs)):
            M_block = softmax(self.M, dim=1)
            loss = self._loss_fn(self.S, self.spot_feature, M_block)

            loss_plot.append(loss.detach().cpu().numpy())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % print_interval == 0:
                print(f"Epoch {epoch + 1}: Loss {loss.item()}")

        # 画出loss_plot的折线图
        plt.plot(loss_plot)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.savefig(join("loss", "tangram_loss_plot.png"))

        with torch.no_grad():
            # output = softmax(self.M, dim=1)
            output = self.M
            return output.cpu().detach().numpy(), loss
