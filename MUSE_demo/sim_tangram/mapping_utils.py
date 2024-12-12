"""
    Mapping helpers
"""

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import logging
from . import mapping_optimizer_CCA as mo

# from . import utils as ut

# from torch.nn.functional import cosine_similarity

logging.getLogger().setLevel(logging.INFO)


def map_cells_to_space(
    single_cell,
    spatial,
    device="cpu",
    lr=0.1,
    num_epochs=1000,
    lambda_d=0,
    lambda_g1=1,
    lambda_g2=1,
    lambda_r=0,
    print_interval=100,
    random_state=None,
    spotwise=True,
    genewise=True,
):

    # S = np.array(
    #     single_cell.cpu().detach().numpy(),
    #     dtype="float32",
    # )
    # G = np.array(spatial.cpu().detach().numpy(), dtype="float32")

    mapper = mo.Mapper(
        S=single_cell,
        G=spatial,
        device=device,
        random_state=random_state,
        lr=lr,
        lambda_d=0,
        lambda_g1=1,
        lambda_g2=1,
        lambda_r=0,
        spotwise=spotwise,
        genewise=genewise,
    )

    mapping_matrix, loss = mapper.train(
        num_epochs=num_epochs,
        print_interval=print_interval,
    )

    return mapping_matrix, loss
