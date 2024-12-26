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

        # self.spot_feature  = torch.cat((self.G.unsqueeze(0), self.image.unsqueeze(0)), dim=0).permute(2, 1, 0)

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

    def _loss_fn(self):
        """
        Evaluates the loss function.

        Args:
            verbose (bool): Optional. Whether to print the loss results. If True, the loss for each individual term is printed as:
                density_term, gene-voxel similarity term, voxel-gene similarity term. Default is True.

        Returns:
            Tuple of 5 Floats: Total loss, gv_loss, vg_loss, kl_reg, entropy_reg
        """
        M_probs = softmax(self.M, dim=1)

        """
            lambda_d=0,
            lambda_g1=1,
            lambda_g2=0,
            lambda_r=0,
        """
        G_pred = torch.matmul(M_probs.t(), self.S)
        gv_term = (
            self.lambda_g1 * cosine_similarity(G_pred, self.G, dim=0).mean()
            + self.lambda_g1 * cosine_similarity(G_pred, self.image, dim=0).mean()
        )

        total_loss = -gv_term

        return total_loss

    def train(self, num_epochs, learning_rate=0.1, print_each=100):
        """
        Run the optimizer and returns the mapping outcome.

        Args:
            num_epochs (int): Number of epochs.
            learning_rate (float): Optional. Learning rate for the optimizer. Default is 0.1.
            print_each (int): Optional. Prints the loss each print_each epochs. If None, the loss is never printed. Default is 100.

        Returns:
            output (ndarray): The optimized mapping matrix M (ndarray), with shape (number_cells, number_spots).
            training_history (dict): loss for each epoch
        """

        torch.manual_seed(seed=self.random_state)
        optimizer = torch.optim.Adam([self.M], lr=learning_rate)

        for t in range(num_epochs):
            loss = self._loss_fn()
            if t % 100 == 0:

                print(f"epoch {t} loss: {loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # take final softmax w/o computing gradients
        with torch.no_grad():
            output = softmax(self.M, dim=1)
            return output
