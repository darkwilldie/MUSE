import numpy as np
import pandas as pd
import tangram as tg
from sim_tangram import mapping_utils
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
    spotwise=True,
    genewise=True,
):
    """
    读取 CSV 文件并构建假 Anndata 以运行 Tangram。

    Args:
        path_csv_single_cell (str): 单细胞数据的 CSV 文件路径。
        path_csv_spatial (str): 空间数据的 CSV 文件路径。
        path_output (str): 输出结果文件路径，默认为 None。
        path_img_for_adata (str): 图像路径用于 Anndata。
        mode (str): Tangram 模式，默认为 "cells"。
        density_prior (str): 密度先验，默认为 'rna_count_based'。
        num_epochs (int): 训练的 epoch 数量，默认为 50。
        device (str): 计算设备，默认为 'cpu'。
        nrows_single_cell (int): 读取单细胞数据的行数，默认为 None。
        nrows_spatial (int): 读取空间数据的行数，默认为 None。

    Returns:
        pd.DataFrame: 映射结果的 DataFrame。
    """

    # 读取单细胞和空间数据

    single_cell, spatial = all_latent_sc, all_latent_st
    device = f"cuda:{device[0]}" if torch.cuda.is_available() else "cpu"
    # 运行 Tangram 映射
    # print("-" * 50)
    # print([data.shape for data in single_cell])
    # print("-" * 50)
    # print([data.shape for data in spatial])
    single_cell = [data.unsqueeze(-1) for data in single_cell]
    S = torch.cat(single_cell, dim=-1)
    spatial = [data.unsqueeze(-1) for data in spatial]
    G = torch.cat(spatial, dim=-1)
    map_result, loss = mapping_utils.map_cells_to_space(
        S,
        G,
        num_epochs=num_epochs,
        random_state=32,
        device=device,
        print_interval=print_interval,
        spotwise=spotwise,
        genewise=genewise,
    )
    print(map_result)

    # 转换结果为 DataFrame
    df_map_result_X = pd.DataFrame(map_result)
    # print(df_map_result_X)

    # 保存结果到文件
    if path_output:
        df_map_result_X.to_csv(path_output)

    return map_result, loss


# 运行函数
if __name__ == "__main__":
    read_csv_and_run_tangram(
        "/home/yanghl/zhushijia/model_new_zhu/general_model/result/latent_sc.csv",
        "/home/yanghl/zhushijia/model_new_zhu/general_model/result/latent_st.csv",
        path_output="normalization.csv",
    )
