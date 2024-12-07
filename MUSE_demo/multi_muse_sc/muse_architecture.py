import torch
import torch.nn as nn
import torch.nn.functional as F
from .triplet_loss import batch_hard_triplet_loss


class MUSE(nn.Module):
    def __init__(
        self,
        dims,
        dim_z,
        n_hidden,
        weight_penalty,
    ):
        super(MUSE, self).__init__()
        self.num_modalities = len(dims)
        self.encoders = nn.ModuleList([Encoder(dim, n_hidden) for dim in dims])
        self.decoders = nn.ModuleList([Decoder(dim_z, n_hidden, dim) for dim in dims])
        self.w_selections = nn.ParameterList(
            [
                nn.Parameter(torch.randn(dim_z, dim_z))
                for _ in range(self.num_modalities)
            ]
        )
        self.weight_penalty = weight_penalty

        self.fc_latent = nn.Linear(self.num_modalities * n_hidden, dim_z)

    def forward(self, inputs, labels=None, triplet_margin=0, triplet_lambda=0):
        z, encoded_inputs = self.encode(inputs)
        inputs_hat = torch.stack(
            [
                self.decoders[i](torch.matmul(z, self.w_selections[i]))
                for i in range(self.num_modalities)
            ]
        )

        sparse_penalty = torch.sqrt(
            sum(
                torch.sum(torch.square(self.w_selections[i]))
                for i in range(self.num_modalities)
            )
        )

        # !! reconstruct loss of x and y are differently calculated
        # where x_mask is used for x for weight selection and 1 is used for y
        x_mask = (inputs != 0).float()
        reconstruct_loss = torch.sum(
            torch.norm(x_mask * (inputs_hat - inputs), dim=2)
        ) / torch.sum(x_mask)

        if triplet_lambda > 0:
            if labels is not None:
                trip_losses = [
                    batch_hard_triplet_loss(labels[i], z, triplet_margin)
                    for i in range(self.num_modalities)
                ]
            else:
                raise ValueError("Labels are required for triplet loss")
        else:
            trip_losses = [
                torch.tensor(0.0, dtype=torch.float16, device=z.device)
                for _ in range(self.num_modalities)
            ]
        trip_loss = sum(trip_losses)

        loss = (
            reconstruct_loss
            + self.weight_penalty * sparse_penalty
            + triplet_lambda * trip_loss
        )

        return (
            z,
            inputs_hat,
            encoded_inputs,
            loss,
            reconstruct_loss,
            sparse_penalty,
            trip_loss,
        )

    def encode(self, inputs):
        hs = [self.encoders[i](inputs[i]) for i in range(self.num_modalities)]
        cat_hs = torch.cat(hs, dim=1)
        encoded_inputs = torch.stack(hs, dim=0)
        z = self.fc_latent(cat_hs)
        return z, encoded_inputs


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
