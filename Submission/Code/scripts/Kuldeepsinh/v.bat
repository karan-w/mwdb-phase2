#!/bin/bash

set model=HOG
set image_type=cc
set k_value=7
set dimensionality_reduction_technique=LDA
set images_folder_path=E:/projects/workspace/mwdb-phase2/Submission/Code/tasks/all

set output_folder_path=E:/projects/workspace/mwdb-phase2/Submission/Outputs/Task1

python tasks/Task1.py --model %model% --x %image_type% --k %k_value% --dimensionality_reduction_technique %dimensionality_reduction_technique% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%

:: python tasks/Task3_modular.py --model %model% --k %k_value% --dimensionality_reduction_technique %dimensionality_reduction_technique% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%

:: python tasks/Task3.py --model %model% --k %k_value% --dimensionality_reduction_technique %dimensionality_reduction_technique% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%

:: python tasks/Task3.py --model %model% --k %k_value% --dimensionality_reduction_technique %dimensionality_reduction_technique% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%
:: python tasks/Task4.py --model %model% --k %k_value% --dimensionality_reduction_technique %dimensionality_reduction_technique% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%