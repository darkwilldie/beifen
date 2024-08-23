import numpy as np
import pandas as pd
import torch
import os
from os.path import join

type = "sum"
dataset = "section2"
# torch_path = f'{name}.pt'
torch_path = f"section2_tangram_cca.pt"
data_dir = "/root/beifen/data"
metadata_path = (
    "/root/beifen/single_cell/metadata/CLUSTER_AND_SUBCLUSTER_INDEX.txt"
)

tensor_tuple = torch.load(torch_path, map_location="cpu")
sum_tensor = tensor_tuple

try:
    sum_tensor = tensor_tuple.detach().cpu().numpy()
except AttributeError:
    sum_tensor = tensor_tuple

# 赋余弦相似度的符号给tangram的张量
# torch_path1 = f"section2_cos.pt"
# # torch_path1 = f"E:/Omics/beifen/test_cossim/original_sc_st_Mouse_brain.pt"
# tensor_tuple1 = torch.load(torch_path1, map_location="cpu")
# signs_a = torch.sign(tensor_tuple1).detach().cpu().numpy()
# sum_tensor = signs_a * np.abs(sum_tensor)

# print(sum_tensor)
# quit()
print(sum_tensor.shape)
# print(cos_sim_sc_image,cos_sim_sc_st,sum_tensor)

# sc_rna_path = f"E:/Omics/beifen/datasets/{name}/process_sc_rna_data.csv"  # [1289, 6025]
sc_rna_path = join(data_dir, dataset, "process_sc_rna_data.csv")
sc_rna = pd.read_csv(sc_rna_path)
# st_rna_path = f"E:/Omics/beifen/datasets/{name}/cell_st_rna_data.csv"  # [189, 6025]
st_rna_path = join(data_dir, dataset, "cell_st_rna_data.csv")
st_rna = pd.read_csv(st_rna_path)
# cell_feature_path = f"E:/Omics/beifen/datasets/{name}/new_cell_deep_feature.csv"
cell_feature_path = join(data_dir, dataset, "new_cell_deep_feature.csv")
cell_feature = pd.read_csv(cell_feature_path)
# cell_id_path = f"E:/Omics/beifen/datasets/{name}/cell_id.csv"
cell_id_path = join(data_dir, dataset, "cell_id.csv")
cell_id = pd.read_csv(cell_id_path)

sc_rna.columns.values[0] = "spot"
st_rna.columns.values[0] = "spot"
cell_feature.columns.values[0] = "spot"

sc_rna_name = sc_rna["spot"].tolist()
st_rna_name = st_rna["spot"].tolist()
cell_feature_name = cell_feature["spot"].tolist()
cell_id_spot = cell_id["spot"].tolist()
cell_id_name = cell_id["cell_id"].tolist()

print(st_rna_name == cell_id_spot)
print(cell_feature_name == cell_id_spot)
##############
sum_tensor = pd.DataFrame(sum_tensor)
# print(sum_tensor)
# quit()
sum_tensor = sum_tensor.rename(
    index=dict(zip(sum_tensor.index, sc_rna_name)),
    columns=dict(zip(sum_tensor.columns, cell_id_name)),
)

sum_tensor.reset_index(level=0, inplace=True)
sum_tensor.columns.values[0] = "sc_name"
# print(sum_tensor)


cluster_type_to_cell_csv = pd.read_csv(
    # "E:/Omics/data_and_code/Single cell/metadata/CLUSTER_AND_SUBCLUSTER_INDEX.txt",
    metadata_path,
    sep="\t",
)
print(cluster_type_to_cell_csv)

cluster_type_to_cell_csv = cluster_type_to_cell_csv.drop([0])
cluster_type_to_cell_csv = cluster_type_to_cell_csv.rename(
    columns={cluster_type_to_cell_csv.columns[0]: "sc_name"}
)

print(cluster_type_to_cell_csv[["sc_name", "CLUSTER"]])
merged_df = pd.merge(
    sum_tensor,
    cluster_type_to_cell_csv[["sc_name", "CLUSTER"]],
    on="sc_name",
    # how="right",
    how="left",
).dropna()
print(merged_df)

####
grouped_df = merged_df.drop(columns=["sc_name"]).groupby("CLUSTER").mean()
print(grouped_df)
grouped_df = grouped_df.drop(["Ependymal", "GABAergic"], axis=0)


max_index_per_column = grouped_df.idxmax()
max_index_df = pd.DataFrame(max_index_per_column, columns=["sc_name"])
max_index_df.reset_index(inplace=True)
max_index_df.rename(columns={"index": "cell_id", "sc_name": "type"}, inplace=True)
print(max_index_df)
max_index_df = max_index_df.sort_values(by="cell_id")
# quit()
folder_path = f"plot_csv/{dataset}/"
if not os.path.exists(folder_path):
    # 如果不存在，则创建文件夹
    os.makedirs(folder_path)
    print("文件夹不存在，已创建")
else:
    print("文件夹已存在")
max_index_df.to_csv(f"plot_csv/{dataset}/{type}_spot_cell_type.csv", index=False)
