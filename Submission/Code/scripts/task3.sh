#!/bin/bash
# Script to run task 1 on Harshil's laptop

model=ELBP
k_value=4
dimensionality_reduction_technique=PCA
images_folder_path="/Users/harshilgandhi/Documents/MWDB_Project/mwdb-phase2/data"
output_folder_path="/Users/harshilgandhi/Documents/MWDB_Project/mwdb-phase2/Submission/Outputs/Task3"

python3 /Users/harshilgandhi/Documents/MWDB_Project/mwdb-phase2/Submission/Code/tasks/Task3.py --model $model --k $k_value --dimensionality_reduction_technique $dimensionality_reduction_technique --images_folder_path "${images_folder_path}" --output_folder_path "${output_folder_path}"