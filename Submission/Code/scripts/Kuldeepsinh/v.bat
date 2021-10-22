#!/bin/bash
# Script to run task 1 on Karan's laptop

set model=HOG
set image_type=cc
set /A k_value = 2
set dimensionality_reduction_technique=kmeans
set images_folder_path=E:/projects/workspace/mwdb-phase2/Submission/Code/tasks/all

set output_folder_path=E:/projects/workspace/mwdb-phase2/Submission/Outputs/Task1

python tasks/Task1_KW.py --model %model% --x %image_type% --k %k_value% --dimensionality_reduction_technique %dimensionality_reduction_technique% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%