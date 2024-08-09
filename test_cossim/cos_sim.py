import torch
import numpy as np
import pandas as pd
import argparse
from sklearn.cluster import KMeans
import model as scsm
from sklearn.preprocessing import StandardScaler
import phenograph
import logging
import torch.nn.functional as F
from sklearn.decomposition import PCA


def cos_sim(sc_feature, other_feature):
    sc_feature = F.normalize(sc_feature, dim=1)
    other_feature = F.normalize(other_feature, dim=1)
    cos_sim_matrix = torch.matmul(sc_feature, other_feature.t())
    softmax_output = F.softmax(cos_sim_matrix, dim=1)
    print(softmax_output)
    return cos_sim_matrix


# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
def cluster_data(data, cluster_type, cluster_number):
    """
    Perform clustering on the given data based on the specified clustering algorithm.

    Parameters:
        data (tensor): The data to cluster.
        cluster_type (str): The type of clustering to use ('kmeans' or 'phenograph').
        cluster_number (int): The number of clusters to form.

    Returns:
        numpy.ndarray: Array of cluster labels.
    """
    if cluster_type == "phenograph":
        cluster_labels, _, _ = phenograph.cluster(data)
    else:
        kmeans = KMeans(n_clusters=cluster_number)
        cluster_labels = kmeans.fit(data).labels_
    return cluster_labels


def find_duplicates_with_indexes(list1, list2):
    duplicates = {}
    for index1, name in enumerate(list1):
        if name in list2:
            index2 = list2.index(name)
            duplicates[name] = (index1, index2)
    return duplicates


def load_data(paths):
    loaded_data = []
    for path in paths:
        try:
            #
            data = pd.read_csv(path)
            data.columns.values[0] = "gene"
            print(data.shape)
            loaded_data.append(data)
        except Exception as e:
            logging.error(f"Error loading data from {path}: {e}")
            continue
    return loaded_data


def dataframe_to_tensor(df):
    np_array = df.iloc[:, 1:].values.astype(float)
    return torch.tensor(np_array, dtype=torch.float32)


def process_data(data, columns_to_drop):
    data = data.drop(columns=columns_to_drop)
    print(data)
    np_array = data.iloc[:, 1:].values.astype(float)
    ###
    scaler = StandardScaler()
    np_array = scaler.fit_transform(np_array)
    return torch.tensor(np_array, dtype=torch.float32)


def calculate_columns_to_drop(data, threshold):
    zero_counts = (data == 0).sum()
    zero_percentage = (zero_counts / len(data)) * 100
    return zero_percentage[zero_percentage >= threshold].index.tolist()


def main(params):
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    sing_cell_load_data = load_data(params.sing_cell_multi_omic_path)
    spatial_transcriptome_load_data = load_data(
        params.spatial_transcriptome_multi_omic_path
    )

    intersection_sing_cell_spatial_transcriptome = find_duplicates_with_indexes(
        params.sing_cell_multi_omic, params.spatial_transcriptome_multi_omic
    )
    print(intersection_sing_cell_spatial_transcriptome)
    data_to_process = []
    threshold = 70
    for gene, indexes in list(intersection_sing_cell_spatial_transcriptome.items()):
        data_to_process.append(("intersection", gene, indexes))

    for index in [
        item
        for item in range(len(params.sing_cell_multi_omic))
        if item
        not in [
            value[0] for value in intersection_sing_cell_spatial_transcriptome.values()
        ]
    ]:
        data_to_process.append(("sing_cell", None, [index, None]))

    for index in [
        item
        for item in range(len(params.spatial_transcriptome_multi_omic))
        if item
        not in [
            value[1] for value in intersection_sing_cell_spatial_transcriptome.values()
        ]
    ]:
        data_to_process.append(("spatial_transcriptome", None, [None, index]))

    After_processing_sc_data = [None] * len(params.sing_cell_multi_omic)
    After_processing_st_data = [None] * len(params.spatial_transcriptome_multi_omic)

    for data_type, gene, indexes in data_to_process:
        sc_index, st_index = indexes

        if (
            data_type == "intersection"
            and sc_index is not None
            and st_index is not None
        ):
            sing_cell_data = sing_cell_load_data[sc_index]
            spatial_data = spatial_transcriptome_load_data[st_index]
            columns_to_drop = calculate_columns_to_drop(sing_cell_data, threshold)
            After_processing_sc_data[sc_index] = process_data(
                sing_cell_data, columns_to_drop
            )
            After_processing_st_data[st_index] = process_data(
                spatial_data, columns_to_drop
            )

        else:
            if sc_index is not None:
                sing_cell_data = sing_cell_load_data[sc_index]
                columns_to_drop = calculate_columns_to_drop(sing_cell_data, threshold)
                After_processing_sc_data[sc_index] = process_data(
                    sing_cell_data, columns_to_drop
                )

            if st_index is not None:
                spatial_data = spatial_transcriptome_load_data[st_index]
                columns_to_drop = calculate_columns_to_drop(sing_cell_data, threshold)
                After_processing_st_data[st_index] = process_data(
                    spatial_data, columns_to_drop
                )

    After_processing_sc_data = [
        data for data in After_processing_sc_data if data is not None
    ]
    After_processing_st_data = [
        data for data in After_processing_st_data if data is not None
    ]
    print(After_processing_sc_data[0].shape)
    print(After_processing_st_data[0].shape)

    cell_feature = pd.read_csv(params.spatial_cell_feature_path)
    cell_feature.columns.values[0] = "gene"
    cell_feature = dataframe_to_tensor(cell_feature)
    print("cell_feature shape", cell_feature.shape)

    sum_cos_sim = cos_sim(After_processing_sc_data[0], After_processing_st_data[0])

    print(sum_cos_sim.shape)
    tensor_tuple = sum_cos_sim
    torch.save(tensor_tuple, "test_cossim/original_sc_st_Mouse_brain.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCSM")
    parser.add_argument("--sing_cell_multi_omic", nargs="+", default=["rna"], type=str)
    parser.add_argument(
        "--spatial_transcriptome_multi_omic", nargs="+", default=["rna"], type=str
    )

    # section2  Mouse_brain_hippocampus_STexpr_cellSegmentation
    parser.add_argument(
        "--sing_cell_multi_omic_path",
        nargs="+",
        default=[
            "/home/yanghl/zhushijia/model_new_zhu/data_process/Mouse_brain_hippocampus_STexpr_cellSegmentation/process_sc_rna_data.csv"
        ],
        type=str,
    )

    parser.add_argument(
        "--spatial_transcriptome_multi_omic_path",
        nargs="+",
        default=[
            "/home/yanghl/zhushijia/model_new_zhu/data_process/Mouse_brain_hippocampus_STexpr_cellSegmentation/cell_st_rna_data.csv"
        ],
        type=str,
    )

    parser.add_argument(
        "--spatial_cell_feature_path",
        default="/home/yanghl/zhushijia/model_new_zhu/data_process/Mouse_brain_hippocampus_STexpr_cellSegmentation/new_cell_deep_feature.csv",
        type=str,
    )

    parser.add_argument(
        "--device", default="cuda:2", type=str
    )  # 这里指定默认设备为 GPU 2
    parser.add_argument("--lambda_recon_gene", default=1, type=int)
    parser.add_argument("--lambda_infoNCE", default=100, type=int)
    parser.add_argument("--lambda_recon_image", default=0, type=int)
    parser.add_argument("--latent_dim", default=512, type=int)
    parser.add_argument("--cluster_type", default="k", type=str)
    parser.add_argument("--learn_rate", default=1e-3, type=int)
    parser.add_argument("--optimization_epsilon_epoch", default=200, type=int)
    parser.add_argument("--cluster_number", default=5, type=int)
    params = parser.parse_args()

    main(params)
