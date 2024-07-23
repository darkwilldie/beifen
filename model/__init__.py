import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle
from .gene_encode import SinglecellNet
from .deep_feature_encode import ImageEncoder
from .voxel_loss import Voxel_loss
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import lr_scheduler
import torch.nn.functional as F
import pandas as pd
import itertools
from sim_tangram.train_tangram import read_csv_and_run_tangram
from scipy.spatial.distance import pdist
import phenograph
from torch.nn import DataParallel
from sklearn.cross_decomposition import CCA
from tqdm import tqdm

def stack_rows_from_all_matrices(matrices):
    """
    Stack rows from all matrices.

    Args:
        matrices (list): A list of matrices.

    Returns:
        list: A list of stacked matrices.

    """
    max_rows = max(matrix.shape[0] for matrix in matrices)
    stacked_matrices = []
    for row_index in range(max_rows):
        selected_rows = [matrix[row_index:row_index+1, :] for matrix in matrices if matrix.shape[0] > row_index]
        if selected_rows:
            stacked_matrix = torch.vstack(selected_rows)
            stacked_matrices.append(stacked_matrix)
    return stacked_matrices

def compute_cca_score(X, Y, n_components=1):
    """
    Compute the CCA (Canonical Correlation Analysis) score between two sets of variables.

    Parameters:
    X (array-like): The first set of variables.
    Y (array-like): The second set of variables.
    n_components (int, optional): The number of canonical components to keep. Default is 1.

    Returns:
    float: The CCA score between X and Y.

    """
    cca = CCA(n_components=n_components)
    cca.fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)
    return np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]

def sort_by_index(content_list, index_list):
    """
    根据索引列表对内容列表进行排序。

    参数:
    - content_list: 要排序的内容列表。
    - index_list: 对应的排序索引列表。

    返回:
    - 排序后的内容列表。
    """
    zipped_lists = zip(index_list, content_list)
    sorted_pairs = sorted(zipped_lists)
    _, sorted_content = zip(*sorted_pairs)
    return list(sorted_content)

def count_matching_elements(matrix1, matrix2):
    """Count the number of elements that are the same in two matrices."""
    if matrix1.shape != matrix2.shape:
        raise ValueError("Both matrices must have the same shape.")
    
    matching_elements = (matrix1 == matrix2).sum().item()
    return matching_elements

def pearson_corr_torch(x, y):
    """Calculate Pearson correlation coefficients for two matrices using PyTorch."""
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    
    numerator = torch.mm(x, y.t())
    
    x_norm = x.norm(dim=1)
    y_norm = y.norm(dim=1)
    
    correlation = numerator / torch.outer(x_norm, y_norm)
    
    # 使用 PyTorch 的 softmax 函数
    softmax_output = F.softmax(correlation, dim=1)
    print(softmax_output)

    return softmax_output

def cos_sim(sc_feature, other_feature):
    """
    Compute the cosine similarity between two feature vectors.

    Args:
        sc_feature (torch.Tensor): The feature vector of the first sample.
        other_feature (torch.Tensor): The feature vector of the second sample.

    Returns:
        torch.Tensor: The cosine similarity matrix between the two feature vectors.
    """
    sc_feature = F.normalize(sc_feature, dim=1)
    other_feature = F.normalize(other_feature, dim=1)
    cos_sim_matrix = torch.matmul(sc_feature, other_feature.t())
    softmax_output = F.softmax(cos_sim_matrix, dim=1)
    print(softmax_output)
    return cos_sim_matrix

def scsm_fit_predict(
        intersection_sc_st_cluster,
        sc_data_not_intersection_cluster,
        st_data_not_intersection_cluster,
        sc_index,
        st_index,
        After_processing_sc_data_shape,
        After_processing_st_data_shape,
        intersection_sc_st,
        sc_data_not_intersection,
        st_data_not_intersection,
        cell_feature,
        latent_dim=100,
        learn_rate=1e-3,
        optimization_epsilon_epoch=100,
        lambda_recon_gene=1,
        lambda_infoNCE=10,
        lambda_recon_image=1,
        device_ids=[2, 3]):  # 指定使用 GPU 2 和 GPU 3
    """
    使用SCSM模型进行多模态单细胞分析的拟合和预测。

    参数：
    intersection_sc_st_cluster (list): 交集的单细胞和空间转录组的聚类结果。
    sc_data_not_intersection_cluster (list): 非交集的单细胞数据的聚类结果。
    st_data_not_intersection_cluster (list): 非交集的空间转录组数据的聚类结果。
    sc_index (list): 单细胞数据的索引。
    st_index (list): 空间转录组数据的索引。
    After_processing_sc_data_shape (list): 经过处理的单细胞数据的形状。
    After_processing_st_data_shape (list): 经过处理的空间转录组数据的形状。
    intersection_sc_st (list): 交集的单细胞和空间转录组数据。
    sc_data_not_intersection (list): 非交集的单细胞数据。
    st_data_not_intersection (list): 非交集的空间转录组数据。
    cell_feature (tensor): 细胞特征。
    latent_dim (int, optional): 潜在空间的维度，默认为100。
    learn_rate (float, optional): 学习率，默认为0.001。
    optimization_epsilon_epoch (int, optional): 优化迭代次数，默认为100。
    lambda_recon_gene (int, optional): 基因重构损失的权重，默认为1。
    lambda_infoNCE (int, optional): infoNCE损失的权重，默认为10。
    lambda_recon_image (int, optional): 图像重构损失的权重，默认为1。
    device_ids (list, optional): 使用的GPU设备ID列表，默认为[2, 3]。

    返回：
    sum_cos_sim (array): 单细胞和空间转录组之间的余弦相似度矩阵。
    reconstructed_data (array): 重构的数据。
    latent_sc (list): 单细胞的潜在空间。
    latent_st (list): 空间转录组的潜在空间。
    latent_image (tensor): 图像的潜在空间。
    """

    n_hidden = 1024

    # device = torch.device(f'cuda:{device_ids[0]}')  # 主设备为 GPU 2
    device = torch.device('cpu')  # 主设备为 GPU 2
    input_data_list = [data.to(device) for data in intersection_sc_st + sc_data_not_intersection + st_data_not_intersection]
    cluster_label_list = [data.to(device) for data in intersection_sc_st_cluster + sc_data_not_intersection_cluster + st_data_not_intersection_cluster]

    feature_dim = [data.shape[1] for data in input_data_list]
    feature_dim_image_feature = cell_feature.shape[1]

    print('++++++++++ SCSM for multi-modality single-cell analysis ++++++++++')
    print('SCSM initialization')

    Model_cell = DataParallel(SinglecellNet(feature_dim, n_hidden, latent_dim).to(device), device_ids=device_ids)
    Model_Image = DataParallel(ImageEncoder(feature_dim_image_feature, n_hidden, latent_dim).to(device), device_ids=device_ids)

    optimizer = optim.Adam([
        {'params': Model_cell.parameters()},
        {'params': Model_Image.parameters()}
    ], lr=learn_rate)

    st_image_loss_fn = Voxel_loss(margin=0.25)
    loss = []
    gene_reconstruction_loss = []
    image_reconstruction_loss = []
    image_loss = []
    trip_loss = []

    for epoch in tqdm(range(optimization_epsilon_epoch)):
        Model_Image.train()
        Model_cell.train()

        z0, reconstructed_data, total_reconstruction_loss_gene, total_sparse_penalty, total_trip_loss_x = Model_cell(input_data_list, cluster_label_list)
        latent_image, _, total_reconstruction_loss_image, _ = Model_Image(cell_feature.to(device))

        latent_sc = [tensor[:shape[0], ] for tensor, shape in zip([z0[0]], After_processing_sc_data_shape)]
        latent_st = [tensor[shape[0]:, ] for tensor, shape in zip([z0[0]], After_processing_sc_data_shape)]

        latent_sc += [z0[1]] if len(z0) > 1 else []
        latent_st += [z0[2]] if len(z0) > 2 else []

        latent_sc = sort_by_index(latent_sc, sc_index)
        latent_st = sort_by_index(latent_st, st_index)
        pairs_sc = list(itertools.combinations(latent_sc, 2))
        pairs_st = list(itertools.combinations(latent_st, 2))

        infoNCE_loss = 0
        for pair in pairs_sc + pairs_st:
            infoNCE_loss += st_image_loss_fn(pair[0], pair[1])
        for pair in latent_st:
            infoNCE_loss += st_image_loss_fn(latent_image, pair)
            
        total_reconstruction_loss_gene = total_reconstruction_loss_gene.sum()
        total_reconstruction_loss_image = total_reconstruction_loss_image.sum()
        total_trip_loss_x = total_trip_loss_x.sum()
        infoNCE_loss = infoNCE_loss.sum()
        total_loss = total_reconstruction_loss_gene * lambda_recon_gene + infoNCE_loss * lambda_infoNCE \
            + total_reconstruction_loss_image * lambda_recon_image + total_trip_loss_x * 0
        # print(total_loss)
        # print(total_reconstruction_loss_gene)
        # print(infoNCE_loss)
        # print(infoNCE_loss * lambda_infoNCE)
        # quit()
        optimizer.zero_grad()

        
        if total_loss.dim() > 0:
            total_loss = total_loss.sum()  # Ensure it's a scalar

        total_loss.backward()
        optimizer.step()
        L_total = total_loss.item()
        loss.append(total_loss.item())
        gene_reconstruction_loss.append(total_reconstruction_loss_gene.item())
        image_reconstruction_loss.append(total_reconstruction_loss_image.item())
        trip_loss.append(total_trip_loss_x.item())
        image_loss.append(infoNCE_loss.item())
        print(f"epoch: {epoch}, total loss: {L_total:.5f}")

    with torch.no_grad():
        z, _, _, _, _ = Model_cell(input_data_list, cluster_label_list)
        latent_image, _, _, _ = Model_Image(cell_feature.to(device))

        latent_sc = [tensor[:shape[0], ] for tensor, shape in zip([z0[0]], After_processing_sc_data_shape)]
        latent_st = [tensor[shape[0]:, ] for tensor, shape in zip([z0[0]], After_processing_sc_data_shape)]

        latent_sc += [z0[1]] if len(z0) > 1 else []
        latent_st += [z0[2]] if len(z0) > 2 else []

        latent_sc = sort_by_index(latent_sc, sc_index)
        latent_st = sort_by_index(latent_st, st_index)

    all_latent_st = latent_st + [latent_image]
    print(len(latent_sc), len(all_latent_st))

    sum_cos_sim =cos_sim(latent_sc[0], all_latent_st[0])
    # sc = latent_sc[0].detach().cpu().numpy()
    # st = all_latent_st[0].detach().cpu().numpy()
    # img = all_latent_st[1].detach().cpu().numpy()

    # n_features_sc = sc.shape[0]
    # n_features_st = st.shape[0]
    # sum_cos_sim = np.zeros((n_features_sc, n_features_st))

    # for j in range(n_features_st):
    #     print(j)
        
    #     Y = np.vstack((st[j, :], img[j, :])).T

    #     for i in range(n_features_sc):
    #         X = sc[i, :].reshape(-1, 1) # 将sc的列转换为列矩阵

    #         # 创建并拟合CCA模型
    #         cca = CCA(n_components=1)
    #         cca.fit(X, Y)

    #         # 计算典型相关系数
    #         X_c, Y_c = cca.transform(X, Y)
    #         correlation_matrix = np.corrcoef(X_c.T, Y_c.T)
    #         sum_cos_sim[i, j] = correlation_matrix[0, 1]  # 取典型相关系数


    df = pd.DataFrame()
    df['total_loss'] = loss
    df['gene_reconstruction_loss'] = gene_reconstruction_loss
    df['image_reconstruction_loss'] = image_reconstruction_loss
    df['trip_loss'] = trip_loss
    df['st_image_loss'] = image_loss
    df.to_csv('loss.csv', index=False, header=True)

    return sum_cos_sim, reconstructed_data[0].detach().cpu().numpy(), latent_sc, latent_st, latent_image
