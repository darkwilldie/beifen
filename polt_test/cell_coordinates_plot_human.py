import pandas as pd

type = "sum"
name = "section2"

image_resize = 0.1
cell_coordinates_csv = pd.read_csv(
    f"/home/yanghl/zhushijia/data/ST image/human/cells_on_spot.csv",
    usecols=["cell_id", "spot", "pixel_x", "pixel_y"],
)
cell_coordinates_csv = cell_coordinates_csv.drop_duplicates(subset=["cell_id"])
cell_coordinates_csv["pixel_x"] *= image_resize
cell_coordinates_csv["pixel_y"] *= image_resize
# 输出去重后的 DataFrame

cell_coordinates_csv.rename(columns={"pixel_x": "X", "pixel_y": "Y"}, inplace=True)
# cell_coordinates_csv = cell_coordinates_csv.sort_values(by='cell_id')

print(cell_coordinates_csv)

cell_type_csv = pd.read_csv(
    f"plot_csv/{name}/{type}_spot_cell_type.csv", usecols=["cell_id"]
)

merged_df = pd.merge(
    cell_type_csv, cell_coordinates_csv, on="cell_id", how="left"
).dropna()
print(merged_df)
merged_df.to_csv(f"plot_csv/{name}/{type}_cell_coordinates.csv", index=False)

# quit()

from PIL import Image

Image.MAX_IMAGE_PIXELS = 1115134080
from PIL import Image, ImageDraw

# 打开图像文件   Visium_FFPE_Mouse_Brain_image.jpg   CytAssist_FFPE_Mouse_Brain_Rep1_tissue_image.tif
image = Image.open(
    f"/home/yanghl/zhushijia/data/ST image/human/CytAssist_FFPE_Mouse_Brain_Rep2_tissue_image.tif"
)
new_width = int(image.width * image_resize)
new_height = int(image.height * image_resize)
resized_image = image.resize((new_width, new_height))

resized_image.save(
    f"plot_csv/{name}/{type}_CytAssist_FFPE_Mouse_Brain_Rep1_tissue_image.jpg"
)
