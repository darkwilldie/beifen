import argparse
import pandas as pd
from os.path import join
from PIL import Image


def main(params):
    type = params.type
    dataset = params.dataset
    result_path = params.result_path
    ST_data_dir = params.ST_data_dir

    if dataset == "section1":
        image_filename = "CytAssist_FFPE_Mouse_Brain_Rep1_tissue_image.tif"
    elif dataset == "section2":
        image_filename = "CytAssist_FFPE_Mouse_Brain_Rep2_tissue_image.tif"
    elif dataset == "Mouse_brain_hippocampus_STexpr_cellSegmentation":
        image_filename = "Visium_FFPE_Mouse_Brain_image.jpg"

    image_resize = 0.1
    cell_coordinates_csv = pd.read_csv(
        join(ST_data_dir, dataset, "cells_on_spot.csv"),
        usecols=["cell_id", "spot", "pixel_x", "pixel_y"],
    )
    print('resize image and save cell coordinates'.center(100, '*'))
    cell_coordinates_csv = cell_coordinates_csv.drop_duplicates(subset=["cell_id"])
    cell_coordinates_csv["pixel_x"] *= image_resize
    cell_coordinates_csv["pixel_y"] *= image_resize
    # 输出去重后的 DataFrame

    cell_coordinates_csv.rename(columns={"pixel_x": "X", "pixel_y": "Y"}, inplace=True)
    # cell_coordinates_csv = cell_coordinates_csv.sort_values(by='cell_id')

    # print(cell_coordinates_csv)

    cell_type_csv = pd.read_csv(
        # f"E:/Omics/beifen/final_res/{dataset}/{type}_spot_cell_type.csv",
        join(result_path, dataset, f"{type}_spot_cell_type.csv"),
        usecols=["cell_id"],
    )

    merged_df = pd.merge(
        cell_type_csv, cell_coordinates_csv, on="cell_id", how="left"
    ).dropna()
    # print(merged_df)
    print("saved", join(result_path, dataset, f"{type}_cell_coordinates.csv"))
    merged_df.to_csv(
        # f"{result_path}/{dataset}/{type}_cell_coordinates.csv",
        join(result_path, dataset, f"{type}_cell_coordinates.csv"),
        index=False
    )


    Image.MAX_IMAGE_PIXELS = 1115134080

    # 打开图像文件   Visium_FFPE_Mouse_Brain_image.jpg   CytAssist_FFPE_Mouse_Brain_Rep1_tissue_image.tif  CytAssist_FFPE_Mouse_Brain_Rep2_tissue_image.tif

    image = Image.open(
        # f"E:/Omics/data_and_code/ST image/{dataset}/CytAssist_FFPE_Mouse_Brain_Rep2_tissue_image.tif"
        join(ST_data_dir, dataset, image_filename)
    )
    # print(image.width, image.height)
    new_width = int(image.width * image_resize)
    new_height = int(image.height * image_resize)
    # print(new_width, new_height)
    resized_image = image.resize((new_width, new_height))

    resized_image.save(
        # f"final_res/{dataset}/{type}_CytAssist_FFPE_Mouse_Brain_Rep1_tissue_image.jpg"
        join(result_path, dataset, f"{type}_CytAssist_FFPE_Mouse_Brain_Rep1_tissue_image.jpg")
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