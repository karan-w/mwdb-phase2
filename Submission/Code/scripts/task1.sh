#!/bin/bash
# Script to run task 1 on Karan's laptop

model=ELBP
image_type=cc
k_value=4
dimensionality_reduction_technique=PCA
images_folder_path="/Users/harshilgandhi/Documents/MWDB_Project/mwdb-phase2/data"
output_folder_path="/Users/harshilgandhi/Documents/MWDB_Project/mwdb-phase2/Submission/Outputs/Task4"

python3 /Users/harshilgandhi/Documents/MWDB_Project/mwdb-phase2/Submission/Code/tasks/Task1.py --model $model --x $image_type --k $k_value --dimensionality_reduction_technique $dimensionality_reduction_technique --images_folder_path "${images_folder_path}" --output_folder_path "${output_folder_path}"