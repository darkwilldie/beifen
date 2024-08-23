import pandas as pd
import os
from os.path import join
from tqdm import tqdm

type = "sum"
name = "test_1000"
ST_data_dir = "/root/beifen/ST_image/human"

cell_coor = pd.read_csv(f"plot_csv/{name}/{type}_cell_coordinates.csv")
print(cell_coor)
# cell_contour_path = f"E:/Omics/data_and_code/ST image/human/cell_contour"
cell_contour_path = join(ST_data_dir, "cell_contour")
pixel_x = []
pixel_y = []

for cell_id in tqdm(cell_coor["cell_id"]):
    # print((cell_id))

    filename = f"cell_{cell_id}.csv"
    # for filename in os.listdir(cell_contour_path):
    # parts = os.path.splitext(filename)[0].split("_")[1]
    # quit()/
    # if int(parts) == int(cell_id):
    if not os.path.exists(os.path.join(cell_contour_path, filename)):
        print(filename)
        continue

    pixel = pd.read_csv(os.path.join(cell_contour_path, filename))
    # print(pixel["x"][0])
    pixel_x.append(pixel["x"][0] * 0.1)
    pixel_y.append(pixel["y"][0] * 0.1)

cell_coor["X"] = pixel_x
cell_coor["Y"] = pixel_y
cell_coor.to_csv(
    f"plot_csv/{name}/{type}_cell_coordinates.csv",
    index=False,
)
