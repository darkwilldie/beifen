python main.py --weight_name test_1000_cca --dataset test_1000

python main.py --dataset section2 --weight_name section2_every_epoch --tangram_epochs 120

python polt_test/name_csv_test_human.py && python polt_test/cell_coordinates_plot_human.py && python plot_csv/pixel_not_human.py && Rscript plot_code/plot_sub_image_2.R

python polt_test/name_csv_test.py && python polt_test/cell_coordinates_plot.py && Rscript plot_code/plot_sub_image_2.R