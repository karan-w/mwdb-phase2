#!/bin/bash
# Script to run task 1 on Karan's laptop

model=HOG
image_type=cc
k_value=2
dimensionality_reduction_technique=PCA
images_folder_path="/Users/karanwadhwani/Documents/ASU/Fall 2021/CSE 515 - MWDB/Project/mwdb-phase2/data"
output_folder_path="/Users/karanwadhwani/Documents/ASU/Fall 2021/CSE 515 - MWDB/Project/mwdb-phase2/Submission/Outputs/Task1"


python tasks/Task1_KW.py --model $model --x $image_type --k $k_value --dimensionality_reduction_technique $dimensionality_reduction_technique --images_folder_path "${images_folder_path}" --output_folder_path "${output_folder_path}"

# python tasks/Task1_KW.py --model CM --x cc --k 2 --dimensionality_reduction_technique PCA --images_folder_path "/Users/karanwadhwani/Documents/ASU/Fall 2021/CSE 515 - MWDB/Project/mwdb-phase2/data" --output_folder_path "/Users/karanwadhwani/Documents/ASU/Fall 2021/CSE 515 - MWDB/Project/mwdb-phase2/Submission/Outputs/Task1"