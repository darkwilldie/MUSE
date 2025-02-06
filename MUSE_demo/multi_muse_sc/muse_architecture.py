import torch
import torch.nn as nn
import torch.nn.functional as F
from .triplet_loss import batch_hard_triplet_loss
import torch.linalg as LA


class MUSE(nn.Module):
    def __init__(
        self,
        dims,
        dim_z,
        n_hidden,
        weight_penalty,
        lambda_info_nce=1,
        temperature=0.07,
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
        self.temperature = temperature
        self.lambda_info_nce = lambda_info_nce
        self.fc_latent = nn.Linear(self.num_modalities * n_hidden, dim_z)

    def forward(
        self,
        inputs,
        labels=None,
        triplet_margin=0,
        triplet_lambda=0,
    ):
        z, encoded_inputs = self.encode(inputs)
        inputs_hat = [
            self.decoders[i](torch.matmul(z, self.w_selections[i]))
            for i in range(self.num_modalities)
        ]

        sparse_penalty = self.weight_penalty * torch.sqrt(
            sum(
                torch.sum(torch.square(self.w_selections[i]))
                for i in range(self.num_modalities)
            )
        )

        # 计算InfoNCE loss
        #! 需要增加对多模态的处理
        info_nce = (
            self.lambda_info_nce
            * info_nce_loss(encoded_inputs[0], encoded_inputs[1], self.temperature)
            if len(encoded_inputs) == 2
            else torch.tensor(0.0, device=z.device)
        )
        # !! reconstruct loss of x and y are differently calculated
        # where x_mask is used for x for weight selection and 1 is used for y

        # x_mask = torch.sign(inputs[0])
        # reconstruct_x = torch.sum(
        #     LA.vector_norm(x_mask * (inputs_hat[0] - inputs[0]), dim=1)
        # ) / torch.sum(x_mask)
        recons = [
            torch.mean(LA.vector_norm(inputs_hat[i] - inputs[i], dim=1))
            for i in range(self.num_modalities)
        ]
        reconstruct_loss = sum(recons)

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
                torch.tensor(0.0, device=z.device) for _ in range(self.num_modalities)
            ]
        trip_loss = triplet_lambda * sum(trip_losses)

        loss = reconstruct_loss + sparse_penalty + trip_loss + info_nce

        return (
            z,
            inputs_hat,
            encoded_inputs,
            loss,
            reconstruct_loss,
            sparse_penalty,
            trip_loss,
            info_nce,
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


class DualMUSE(nn.Module):
    def __init__(
        self,
        dims_sc,  # single cell模态的维度列表
        dims_st,  # transcriptome模态的维度列表
        dim_z,  # 潜在空间维度
        n_hidden,  # 隐藏层节点数
        weight_penalty,  # 权重惩罚系数
        info_nce_lambda_sc=1.0,
        info_nce_lambda_st=1.0,
        temperature=0.07,  # InfoNCE loss的温度参数
    ):
        super(DualMUSE, self).__init__()
        self.temperature = temperature
        # 创建两个MUSE模型
        self.muse_sc = MUSE(
            dims_sc, dim_z, n_hidden, weight_penalty, info_nce_lambda_sc, temperature
        )
        self.muse_st = MUSE(
            dims_st, dim_z, n_hidden, weight_penalty, info_nce_lambda_st, temperature
        )

    def forward(
        self,
        inputs_sc,  # single cell数据
        inputs_st,  # transcriptome数据
        labels_sc=None,
        labels_st=None,
        triplet_margin=[0, 0],
        triplet_lambda=0,
        info_nce_lambda=1.0,  # InfoNCE loss的权重
    ):
        # 前向传播两个MUSE模型
        (
            z_sc,
            inputs_hat_sc,
            encoded_sc,
            loss_sc,
            recon_loss_sc,
            sparse_penalty_sc,
            trip_loss_sc,
            info_nce_sc,
        ) = self.muse_sc(inputs_sc, labels_sc, triplet_margin[0], triplet_lambda)

        (
            z_st,
            inputs_hat_st,
            encoded_st,
            loss_st,
            recon_loss_st,
            sparse_penalty_st,
            trip_loss_st,
            info_nce_st,
        ) = self.muse_st(inputs_st, labels_st, triplet_margin[1], triplet_lambda)

        # 计算InfoNCE loss
        info_nce = info_nce_lambda * info_nce_loss(z_sc, z_st, self.temperature) if info_nce_lambda > 0 else torch.tensor(0.0, device=z_sc.device)

        # 总损失
        total_loss = loss_sc + loss_st + info_nce

        return (
            (z_sc, z_st),  # 两个模态的潜在表示
            (inputs_hat_sc, inputs_hat_st),  # 重构输出
            (encoded_sc, encoded_st),  # 编码器输出
            total_loss,  # 总损失
            (loss_sc, loss_st),  # 各自的MUSE损失
            info_nce,  # InfoNCE loss
            (recon_loss_sc, recon_loss_st),  # 重构损失
            (sparse_penalty_sc, sparse_penalty_st),  # 稀疏惩罚
            (trip_loss_sc, trip_loss_st),  # 三元组损失
            (info_nce_sc, info_nce_st),  # 各自的InfoNCE损失
        )


def info_nce_loss(z1, z2, temperature=0.07):
    """计算InfoNCE loss"""
    # 归一化特征
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    # import cos_tg.train_tangram as tg_cos

    # # tangram
    # mapping, _ = tg_cos.read_csv_and_run_tangram(
    #     [z1],
    #     [z2],
    #     num_epochs=500,
    #     print_interval=500,
    #     lambda_g1=1,
    #     lambda_g2=1,
    # )
    # mapping = torch.tensor(mapping, dtype=torch.float32, device='cuda:0')
    # 计算相似度矩阵
    logits = torch.mm(z1, z2.t()) / temperature

    # loss = F.cross_entropy(logits, mapping)
    labels = torch.arange(z1.shape[0], device=z1.device)
    loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)

    return loss
