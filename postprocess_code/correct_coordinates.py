import pandas as pd
import os
from os.path import join
from tqdm import tqdm
import argparse


def main(params):
    type = params.type
    dataset = params.dataset
    result_path = params.result_path
    ST_data_dir = params.ST_data_dir
    # cell_contour_path = f"E:/Omics/data_and_code/ST image/human/cell_contour"
    cell_coor_path = join(result_path, dataset, f"{type}_cell_coordinates.csv")
    if dataset == "Mouse_brain_hippocampus_STexpr_cellSegmentation":
        cell_contour_path = join(ST_data_dir, dataset, "cell_contour")
    else:
        cell_contour_path = join(ST_data_dir, "human", "cell_contour")

    # cell_coor = pd.read_csv(f"{result_path}/{dataset}/{type}_cell_coordinates.csv")
    cell_coor = pd.read_csv(cell_coor_path)
    # print(cell_coor)
    pixel_x = []
    pixel_y = []

    print(f"correct cell coordinates for {dataset}".center(100, "*"))
    for cell_id in tqdm(cell_coor["cell_id"]):

        filename = f"cell_{cell_id}.csv"
        file_path = join(cell_contour_path, filename)
        if not os.path.exists(file_path):
            print(filename)
            continue

        pixel = pd.read_csv(file_path)
        # print(pixel["x"][0])
        pixel_x.append(pixel["x"][0] * 0.1)
        pixel_y.append(pixel["y"][0] * 0.1)

    cell_coor["X"] = pixel_x
    cell_coor["Y"] = pixel_y
    cell_coor.to_csv(
        cell_coor_path,
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        default="sum",
        type=str,
    )
    parser.add_argument(
        "--dataset",
        default="Mouse_brain_hippocampus_STexpr_cellSegmentation",
        type=str,
    )
    parser.add_argument(
        "--result_path",
        default="final_res",
        type=str,
    )
    parser.add_argument(
        "--ST_data_dir",
        default="../ST_image",
        type=str,
    )
    params = parser.parse_args()
    main(params)