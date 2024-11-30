import torch
import torch.nn as nn
import torch.nn.functional as F
from .triplet_loss import batch_hard_triplet_loss


class MUSE(nn.Module):
    def __init__(
        self,
        n_input_x,
        n_input_y,
        dim_z,
        n_hidden,
        weight_penalty,
    ):
        super(MUSE, self).__init__()
        self.encoder_x = Encoder(n_input_x, n_hidden)
        self.encoder_y = Encoder(n_input_y, n_hidden)
        self.decoder_x = Decoder(dim_z, n_hidden, n_input_x)
        self.decoder_y = Decoder(dim_z, n_hidden, n_input_y)
        self.w_selection_x = nn.Parameter(torch.randn(dim_z, dim_z))
        self.w_selection_y = nn.Parameter(torch.randn(dim_z, dim_z))
        self.weight_penalty = weight_penalty

        self.fc_latent = nn.Linear(2 * n_hidden, dim_z)

    def forward(self, x, y, label_x, label_y, triplet_margin=0, triplet_lambda=0):
        z, encode_x, encode_y = self.encode(x, y)
        x_hat = self.decoder_x(torch.matmul(z, self.w_selection_x))
        y_hat = self.decoder_y(torch.matmul(z, self.w_selection_y))

        sparse_penalty = torch.sqrt(
            torch.sum(torch.square(self.w_selection_x))
            + torch.sum(torch.square(self.w_selection_y))
        )

        x_mask = (x != 0).float()
        reconstruct_x = torch.sum(torch.norm(x_mask * (x_hat - x), dim=1)) / torch.sum(
            x_mask
        )
        reconstruct_y = torch.mean(torch.norm(y_hat - y, dim=1))
        reconstruct_loss = reconstruct_x + reconstruct_y

        if triplet_lambda > 0:
            trip_loss_x = batch_hard_triplet_loss(label_x, z, triplet_margin)
            trip_loss_y = batch_hard_triplet_loss(label_y, z, triplet_margin)
        else:
            trip_loss_x = 0
            trip_loss_y = 0

        loss = (
            reconstruct_loss
            + self.weight_penalty * sparse_penalty
            + triplet_lambda * trip_loss_x
            + triplet_lambda * trip_loss_y
        )

        return (
            z,
            x_hat,
            y_hat,
            encode_x,
            encode_y,
            loss,
            reconstruct_loss,
            sparse_penalty,
            trip_loss_x,
            trip_loss_y,
        )

    def encode(self, x, y):
        h_x = self.encoder_x(x)
        h_y = self.encoder_y(y)
        h = torch.cat([h_x, h_y], dim=1)
        z = self.fc_latent(h)
        return z, h_x, h_y


class Encoder(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)

    def forward(self, x):
        h0 = F.elu(self.fc1(x))
        h1 = F.tanh(self.fc2(h0))
        return h1


class Decoder(nn.Module):
    def __init__(self, dim_z, n_hidden, n_output):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(dim_z, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_output)

    def forward(self, z):
        h0 = F.elu(self.fc1(z))
        h1 = F.tanh(self.fc2(h0))
        y = self.fc3(h1)
        return y
