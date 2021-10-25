#!/bin/bash

set model=HOG
set k_value=7
set dimensionality_reduction_technique=LDA
set images_folder_path=D:/MWDB/mwdb-phase2/Submission/Code/tasks/all
set output_folder_path=D:/MWDB/mwdb-phase2/Submission/Outputs/Task3
python tasks/Task3.py --model %model% --k %k_value% --dimensionality_reduction_technique %dimensionality_reduction_technique% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%