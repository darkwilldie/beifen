#@ test_1000
python main.py --weight_name test_1000_cca --dataset test_1000
#@ section2
python main.py --dataset section2 --weight_name section2 --tangram_epochs 200 --data_dir /home/ljc/CLIMA/datasets/
#@ section1
python main.py --dataset section1 --weight_name section1 --tangram_epochs 100 --data_dir /home/ljc/CLIMA/datasets/
#@ Mouse_brain_hippocampus_STexpr_cellSegmentation
python main.py --dataset Mouse_brain_hippocampus_STexpr_cellSegmentation --weight_name Mouse --tangram_epochs 100 --data_dir /home/ljc/CLIMA/datasets/
#@ human
python postprocess_code/postprocess.py --type sum --data_dir ../datasets/ --metadata_path ../ST_image/human/Wu_etal_2021_BRCA_scRNASeq/metadata.csv --result_path final_res --ST_data_dir ../ST_image/ --dataset test_1000 --torch_path test_1000_tangram_cca.pt 
#@ section1
python postprocess_code/postprocess.py --type sum --data_dir ../datasets/ --metadata_path ../single_cell/metadata/CLUSTER_AND_SUBCLUSTER_INDEX.txt --result_path final_res --ST_data_dir ../ST_image/ --dataset section1 --torch_path section1_tangram_cca.pt 
#@ section2
python postprocess_code/postprocess.py --type sum --data_dir ../datasets/ --metadata_path ../single_cell/metadata/CLUSTER_AND_SUBCLUSTER_INDEX.txt --result_path final_res --ST_data_dir ../ST_image/ --dataset section2 --torch_path section2_tangram_cca.pt 
#@ Mouse_brain_hippocampus_STexpr_cellSegmentation
python postprocess_code/postprocess.py --type sum --data_dir ../datasets/ --metadata_path ../single_cell/metadata/CLUSTER_AND_SUBCLUSTER_INDEX.txt --result_path final_res --ST_data_dir ../ST_image/ --dataset Mouse_brain_hippocampus_STexpr_cellSegmentation --torch_path Mouse_tangram_cca.pt 
