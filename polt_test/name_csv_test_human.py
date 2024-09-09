import numpy as np
import pandas as pd
import torch
import os

type = "sum"
name = "test_1000"
# sum
torch_path = f"test_1000_test_tangram_cca.pt"
metadata_path = "/root/beifen/ST_image/human/metadata.csv"

tensor_tuple = torch.load(torch_path, map_location="cpu")
sum_tensor = tensor_tuple

try:
    sum_tensor = tensor_tuple.detach().cpu().numpy()
except AttributeError:
    sum_tensor = tensor_tuple

# 赋余弦相似度的符号给tangram的张量
torch_path1 = f"test_1000_cos.pt"
# torch_path1 = f"E:/Omics/beifen/test_cossim/original_sc_st_Mouse_brain.pt"
tensor_tuple1 = torch.load(torch_path1, map_location="cpu")
print('tangram shape',sum_tensor.shape)
print('cos sim shape',tensor_tuple1.shape)
# assert False
signs_a = torch.sign(tensor_tuple1).detach().cpu().numpy()
sum_tensor = signs_a * np.abs(sum_tensor)
# print(sum_tensor)
# print(sum_tensor.shape)
# print(cos_sim_sc_image,cos_sim_sc_st,sum_tensor)

sc_rna_path = f"/root/beifen/data/{name}/process_sc_rna_data.csv"  # [1289, 6025]
sc_rna = pd.read_csv(sc_rna_path)

cell_id_path = f"/root/beifen/data/{name}/cell_id.csv"
cell_id = pd.read_csv(cell_id_path)

sc_rna.columns.values[0] = "spot"

sc_rna_name = sc_rna["spot"].tolist()
cell_id_spot = cell_id["spot"].tolist()
cell_id_name = cell_id["cell_id"].tolist()

print(len(sc_rna_name))

print(len(cell_id_spot))

##############
sum_tensor = pd.DataFrame(sum_tensor)
print(sum_tensor)
# quit()
sum_tensor = sum_tensor.rename(
    index=dict(zip(sum_tensor.index, sc_rna_name)),
    columns=dict(zip(sum_tensor.columns, cell_id_name)),
)

sum_tensor.reset_index(level=0, inplace=True)
sum_tensor.columns.values[0] = "sc_name"
print(sum_tensor)

cluster_type_to_cell_csv = pd.read_csv(
    # "/root/beifen/ST image/human/Wu_etal_2021_BRCA_scRNASeq/metadata.csv"
    metadata_path
)
cluster_type_to_cell_csv = cluster_type_to_cell_csv[["Unnamed: 0", "celltype_major"]]
cluster_type_to_cell_csv = cluster_type_to_cell_csv.rename(
    columns={
        cluster_type_to_cell_csv.columns[0]: "sc_name",
        cluster_type_to_cell_csv.columns[1]: "CLUSTER",
    }
)
print(cluster_type_to_cell_csv)

merged_df = pd.merge(
    sum_tensor, cluster_type_to_cell_csv, on="sc_name", how="right"
).dropna()
print(merged_df)

duplicates = merged_df["sc_name"].duplicated()
if duplicates.any():
    print("ID列存在重复值")
    # 输出重复的ID值
    print("重复的ID值有：")
    print(merged_df["sc_name"][duplicates])
else:
    print("ID列没有重复值")


grouped_df = merged_df.drop(columns=["sc_name"]).groupby("CLUSTER").mean()
print(grouped_df)

max_index_per_column = grouped_df.idxmax()
max_index_df = pd.DataFrame(max_index_per_column, columns=["sc_name"])
max_index_df.reset_index(inplace=True)
max_index_df.rename(columns={"index": "cell_id", "sc_name": "type"}, inplace=True)
print(max_index_df)
duplicates = max_index_df["cell_id"].duplicated()
if duplicates.any():
    print("ID列存在重复值")
    # 输出重复的ID值
    print("重复的ID值有：")
    print(max_index_df["cell_id"][duplicates])
else:
    print("ID列没有重复值")
max_index_df = max_index_df.sort_values(by="cell_id")
# quit()
folder_path = f"plot_csv/{name}/"
if not os.path.exists(folder_path):
    # 如果不存在，则创建文件夹
    os.makedirs(folder_path)
    print("文件夹不存在，已创建")
else:
    print("文件夹已存在")
duplicates = max_index_df["cell_id"].duplicated()
if duplicates.any():
    print("ID列存在重复值")
    # 输出重复的ID值
    print("重复的ID值有：")
    print(max_index_df["cell_id"][duplicates])
else:
    print("ID列没有重复值")

max_index_df.to_csv(f"plot_csv/{name}/{type}_spot_cell_type.csv", index=False)
