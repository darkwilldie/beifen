import pandas as pd
import os

type = "sum"
name = "test_500"

cell_coor = pd.read_csv(
    f"/home/yanghl/zhushijia/model_new_zhu/model_test1/plot_csv/{name}/{type}_cell_coordinates.csv"
)
print(cell_coor)
cell_contour_path = f"/home/yanghl/zhushijia/data/ST image/human/cell_contour/"
pixel_x = []
pixel_y = []

for cell_id in cell_coor["cell_id"]:
    print((cell_id))

    for filename in os.listdir(cell_contour_path):
        parts = os.path.splitext(filename)[0].split("_")[1]
        # quit()/
        if int(parts) == int(cell_id):
            pixel = pd.read_csv(os.path.join(cell_contour_path, filename))
            print(pixel["x"][0])
            pixel_x.append(pixel["x"][0] * 0.1)
            pixel_y.append(pixel["y"][0] * 0.1)

cell_coor["X"] = pixel_x
cell_coor["Y"] = pixel_y
cell_coor.to_csv(
    f"/home/yanghl/zhushijia/model_new_zhu/model_test1/plot_csv/{name}/{type}_cell_coordinates.csv",
    index=False,
)
