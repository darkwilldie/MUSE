import numpy as np
import pandas as pd
import tangram as tg
from . import mapping_utils
import sys
import cv2
from anndata import AnnData
import torch

# 添加模块路径
sys.path.append("tangram_src")

# 设置随机种子以确保结果可重复
np.random.seed(42)


def read_csv_and_run_tangram(
    all_latent_sc,
    all_latent_st,
    path_output=None,
    num_epochs=200,
    print_interval=50,
    device=[0],
    lambda_d=0,
    lambda_g1=1,
    lambda_g2=1,
    lambda_r=0,
):

    # 读取单细胞和空间数据
    single_cell, spatial = all_latent_sc, all_latent_st
    device = f"cuda:{device[0]}" if torch.cuda.is_available() else "cpu"
    single_cell = [data.unsqueeze(-1) for data in single_cell]
    S = torch.cat(single_cell, dim=-1)
    spatial = [data.unsqueeze(-1) for data in spatial]
    G = torch.cat(spatial, dim=-1)
    # 运行 Tangram 映射
    map_result, loss = mapping_utils.map_cells_to_space(
        S,
        G,
        num_epochs=num_epochs,
        random_state=32,
        device=device,
        print_interval=print_interval,
        lambda_d=lambda_d,
        lambda_g1=lambda_g1,
        lambda_g2=lambda_g2,
        lambda_r=lambda_r,
    )
    print(map_result)

    # 转换结果为 DataFrame
    df_map_result_X = pd.DataFrame(map_result)
    # print(df_map_result_X)

    # 保存结果到文件
    if path_output:
        df_map_result_X.to_csv(path_output)

    return map_result, loss
