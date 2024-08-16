import torch
import numpy as np
import pandas as pd
import argparse
from sklearn.cluster import KMeans
import model as scsm
from sklearn.preprocessing import StandardScaler
import phenograph
import logging
from sklearn.decomposition import PCA
import os


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
            print("data.shape:", data.shape)
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
    zero_counts = (data == 0).sum(axis=0)
    zero_percentage = (zero_counts / len(data)) * 100
    return zero_percentage[zero_percentage >= threshold].index.tolist()


def main(params):
    """
    Process and analyze single-cell and spatial transcriptome data.

    Args:
        params (object): An object containing various parameters for data processing and analysis.

    Returns:
        None
    """
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    data_path = os.path.join(
        params.data_dir, params.dataset, "processed_data", "data.npz"
    )
    # data_path = os.path.join('')

    if os.path.exists(data_path):
        print("-" * 50)
        print("loading data from:", data_path)
        loaded_data = np.load(data_path, allow_pickle=True)
        intersection_sc_st_cluster = [torch.tensor(arr) for arr in loaded_data["arr_0"]]
        sc_data_not_intersection_cluster = [
            torch.tensor(arr) for arr in loaded_data["arr_1"]
        ]
        st_data_not_intersection_cluster = [
            torch.tensor(arr) for arr in loaded_data["arr_2"]
        ]
        sc_index = list(loaded_data["arr_3"])
        st_index = list(loaded_data["arr_4"])
        After_processing_sc_data_shape = list(loaded_data["arr_5"])
        After_processing_st_data_shape = list(loaded_data["arr_6"])
        intersection_sc_st = [torch.tensor(arr) for arr in loaded_data["arr_7"]]
        sc_data_not_intersection = [torch.tensor(arr) for arr in loaded_data["arr_8"]]
        st_data_not_intersection = [torch.tensor(arr) for arr in loaded_data["arr_9"]]
        cell_feature = torch.tensor(loaded_data["arr_10"])

    else:
        print("%" * 50)
        print("data_path not exists:", data_path)
        params.sing_cell_multi_omic_path = [
            os.path.join(params.data_dir, params.dataset, path)
            for path in params.sing_cell_multi_omic_path
        ]
        params.spatial_transcriptome_multi_omic_path = [
            os.path.join(params.data_dir, params.dataset, path)
            for path in params.spatial_transcriptome_multi_omic_path
        ]
        params.spatial_cell_feature_path = os.path.join(
            params.data_dir, params.dataset, params.spatial_cell_feature_path
        )

        # Load single-cell and spatial transcriptome data
        sing_cell_load_data = load_data(params.sing_cell_multi_omic_path)
        spatial_transcriptome_load_data = load_data(
            params.spatial_transcriptome_multi_omic_path
        )

        # print('sing_cell_load_data:', sing_cell_load_data)
        # print('spatial_transcriptome_load_data:', spatial_transcriptome_load_data)

        # Find the intersection of single-cell and spatial transcriptome data
        intersection_sing_cell_spatial_transcriptome = find_duplicates_with_indexes(
            params.sing_cell_multi_omic, params.spatial_transcriptome_multi_omic
        )
        # print('intersection_sing_cell_spatial_transcriptome:', intersection_sing_cell_spatial_transcriptome)

        # Prepare data for processing
        data_to_process = []
        threshold = 70
        for gene, indexes in list(intersection_sing_cell_spatial_transcriptome.items()):
            data_to_process.append(("intersection", gene, indexes))

        for index in [
            item
            for item in range(len(params.sing_cell_multi_omic))
            if item
            not in [
                value[0]
                for value in intersection_sing_cell_spatial_transcriptome.values()
            ]
        ]:
            data_to_process.append(("sing_cell", None, [index, None]))

        for index in [
            item
            for item in range(len(params.spatial_transcriptome_multi_omic))
            if item
            not in [
                value[1]
                for value in intersection_sing_cell_spatial_transcriptome.values()
            ]
        ]:
            data_to_process.append(("spatial_transcriptome", None, [None, index]))

        # Process single-cell and spatial transcriptome data
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
                print("-" * 50)
                print("st_index:", st_index)
                print("spatial_data:", spatial_data)
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
                    columns_to_drop = calculate_columns_to_drop(
                        sing_cell_data, threshold
                    )
                    After_processing_sc_data[sc_index] = process_data(
                        sing_cell_data, columns_to_drop
                    )

                if st_index is not None:
                    spatial_data = spatial_transcriptome_load_data[st_index]
                    # 将sing_cell_data改为spatial_data，原来的sing_cell_data可能笔误
                    columns_to_drop = calculate_columns_to_drop(spatial_data, threshold)
                    After_processing_st_data[st_index] = process_data(
                        spatial_data, columns_to_drop
                    )

        After_processing_sc_data = [
            data for data in After_processing_sc_data if data is not None
        ]
        After_processing_st_data = [
            data for data in After_processing_st_data if data is not None
        ]

        # Cluster single-cell data
        sc_cluster = []
        for data in After_processing_sc_data:
            cluster_labels = cluster_data(
                data.detach().cpu().numpy(), params.cluster_type, params.cluster_number
            )
            sc_cluster.append(torch.tensor(cluster_labels))
            # print(f'After_processing_sc_data Shape: {data.shape}')

        # Cluster spatial transcriptome data
        st_cluster = []
        for data in After_processing_st_data:
            cluster_labels = cluster_data(
                data.detach().cpu().numpy(), params.cluster_type, params.cluster_number
            )
            st_cluster.append(torch.tensor(cluster_labels))
            # print(f'After_processing_st_data Data Shape: {data.shape}')

        # Load cell features
        cell_feature = pd.read_csv(params.spatial_cell_feature_path)
        cell_feature.columns.values[0] = "gene"
        cell_feature = dataframe_to_tensor(cell_feature)
        # print('cell_feature shape:', cell_feature.shape)

        # Process intersection of single-cell and spatial transcriptome data
        intersection_sc_st = []
        After_processing_sc_data_shape = []
        After_processing_st_data_shape = []
        intersection_sc_st_cluster = []
        sc_index = []
        st_index = []

        for gene, indexes in list(intersection_sing_cell_spatial_transcriptome.items()):
            intersection_sc_st.append(
                torch.cat(
                    (
                        After_processing_sc_data[indexes[0]],
                        After_processing_st_data[indexes[1]],
                    ),
                    dim=0,
                )
            )
            intersection_sc_st_cluster.append(
                torch.cat((sc_cluster[indexes[0]], st_cluster[indexes[1]]), dim=0)
            )
            After_processing_sc_data_shape.append(
                After_processing_sc_data[indexes[0]].shape
            )
            After_processing_st_data_shape.append(
                After_processing_st_data[indexes[1]].shape
            )
            sc_index.append(indexes[0])
            st_index.append(indexes[1])

        # Process non-intersecting single-cell data
        sc_data_not_intersection_cluster = []
        sc_data_not_intersection = []
        for index in [
            item
            for item in range(len(params.sing_cell_multi_omic))
            if item
            not in [
                value[0]
                for value in intersection_sing_cell_spatial_transcriptome.values()
            ]
        ]:
            sc_data_not_intersection.append(After_processing_sc_data[index])
            sc_data_not_intersection_cluster.append(sc_cluster[index])
            sc_index.append(index)

        # Process non-intersecting spatial transcriptome data
        st_data_not_intersection_cluster = []
        st_data_not_intersection = []
        for index in [
            item
            for item in range(len(params.spatial_transcriptome_multi_omic))
            if item
            not in [
                value[1]
                for value in intersection_sing_cell_spatial_transcriptome.values()
            ]
        ]:
            st_data_not_intersection.append(After_processing_st_data[index])
            st_data_not_intersection_cluster.append(st_cluster[index])
            st_index.append(index)

        # print('intersection_sc_st:', intersection_sc_st)
        # print('sc_data_not_intersection:', sc_data_not_intersection)
        # print('st_data_not_intersection:', st_data_not_intersection)

        # save the processed data as numpy arrays
        save_path = os.path.join(params.data_dir, params.dataset, "processed_data")
        os.makedirs(save_path, exist_ok=True)
        # print('After_processing_sc_data_shape:', After_processing_sc_data_shape)
        np.savez(
            os.path.join(save_path, "data.npz"),
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
        )

    # print(intersection_sc_st, sc_data_not_intersection, st_data_not_intersection)

    # Perform data analysis using the processed data
    sum_cos_sim, reconstructed_data, latent_sc, latent_st, latent_image = (
        scsm.scsm_fit_predict(
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
            latent_dim=params.latent_dim,
            learn_rate=params.learn_rate,
            optimization_epsilon_epoch=params.epochs,
            lambda_recon_gene=params.lambda_recon_gene,
            lambda_infoNCE=params.lambda_infoNCE,
            lambda_recon_image=params.lambda_recon_image,
            device_ids=[
                0,
            ],  # Specify GPU devices to use
        )
    )
    print("reconstructed_data.shape:", reconstructed_data.shape)

    print("sum_cos_sim.shape:", sum_cos_sim.shape)
    tensor_tuple = sum_cos_sim
    torch.save(tensor_tuple, params.weight_name + ".pt")

    df_cos_sim_sc_image = pd.DataFrame(latent_sc[0].cpu().detach().numpy())
    df_cos_sim_sc_st = pd.DataFrame(latent_st[0].cpu().detach().numpy())
    df_latent_image = pd.DataFrame(latent_image.cpu().detach().numpy())

    result_dir = "result"
    os.makedirs(result_dir, exist_ok=True)

    df_cos_sim_sc_image.to_csv(
        os.path.join(result_dir, "latent_sc.csv"), index=False, header=True
    )
    df_cos_sim_sc_st.to_csv(
        os.path.join(result_dir, "latent_st.csv"), index=False, header=True
    )
    df_latent_image.to_csv(
        os.path.join(result_dir, "latent_image.csv"), index=False, header=True
    )


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
        default=["process_sc_rna_data.csv"],
        type=str,
    )

    parser.add_argument(
        "--spatial_transcriptome_multi_omic_path",
        nargs="+",
        default=["cell_st_rna_data.csv"],
        type=str,
    )

    parser.add_argument(
        "--spatial_cell_feature_path", default="new_cell_deep_feature.csv", type=str
    )

    # parser.add_argument('--device', default='cuda:2', type=str)  # 这里指定默认设备为 GPU 2
    parser.add_argument("--lambda_recon_gene", default=1, type=int)
    parser.add_argument("--lambda_infoNCE", default=100, type=int)
    parser.add_argument("--lambda_recon_image", default=0, type=int)
    parser.add_argument("--latent_dim", default=512, type=int)
    parser.add_argument("--cluster_type", default="k", type=str)
    parser.add_argument("--learn_rate", default=1e-3, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--cluster_number", default=5, type=int)
    parser.add_argument("--weight_name", default="section2", type=str)
    parser.add_argument(
        "--dataset",
        default="section2",
        type=str,
    )
    parser.add_argument("--data_dir", default="/root/beifen/data", type=str)
    params = parser.parse_args()

if __name__ == "__main__":
    main(params)
