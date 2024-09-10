import argparse
import correct_coordinates as correct_coordinates, name_csv_test, name_csv_test_human, cell_coordinates_plot, cell_coordinates_plot_human

def main(params):
    if 'test' in params.dataset:
        name_csv_test_human.main(params)
        cell_coordinates_plot_human.main(params)
        correct_coordinates.main(params)
    else:
        name_csv_test.main(params)
        cell_coordinates_plot.main(params)
        if 'Mouse' in params.dataset:
            correct_coordinates.main(params)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        default="sum",
        type=str,
    )
    parser.add_argument(
        "--dataset",
        default="section2",
        type=str,
    )
    parser.add_argument(
        "--data_dir",
        default="../datasets",
        type=str,
    )
    parser.add_argument(
        "--metadata_path",
        default="../single_cell/metadata/CLUSTER_AND_SUBCLUSTER_INDEX.txt",
        type=str,
    )
    parser.add_argument(
        "--torch_path",
        default="section2_tangram_cca.pt",
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