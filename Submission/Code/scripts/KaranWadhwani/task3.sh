#!/bin/bash
# Script to run task 1 on Karan's laptop

model=ELBP
k_value=4
dimensionality_reduction_technique=PCA
images_folder_path="/Users/karanwadhwani/Documents/ASU/Fall2021/CSE515_MWDB/Project/mwdb-phase2/Inputs/tests_v2"
output_folder_path="/Users/karanwadhwani/Documents/ASU/Fall2021/CSE515_MWDB/Project/mwdb-phase2/Submission/Outputs/Task3"

python3 tasks/Task3.py --model $model --k $k_value --dimensionality_reduction_technique $dimensionality_reduction_technique --images_folder_path "${images_folder_path}" --output_folder_path "${output_folder_path}"