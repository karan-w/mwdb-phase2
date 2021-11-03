#!/bin/bash
# Script to run task 1 on Karan's laptop

model=ELBP
subject_id=3
k_value=2
dimensionality_reduction_technique=PCA
images_folder_path="/Users/karanwadhwani/Documents/ASU/Fall2021/CSE515_MWDB/Project/mwdb-phase2/Inputs/tests_v2"
output_folder_path="/Users/karanwadhwani/Documents/ASU/Fall2021/CSE515_MWDB/Project/mwdb-phase2/Submission/Outputs/Task2"

python3.7 tasks/Task2.py --model $model --x $subject_id --k $k_value --dimensionality_reduction_technique $dimensionality_reduction_technique --images_folder_path "${images_folder_path}" --output_folder_path "${output_folder_path}"
# echo "python3.7 tasks/Task1_KW.py --model $model --x $image_type --k $k_value --dimensionality_reduction_technique $dimensionality_reduction_technique --images_folder_path "${images_folder_path}" --output_folder_path "${output_folder_path}""