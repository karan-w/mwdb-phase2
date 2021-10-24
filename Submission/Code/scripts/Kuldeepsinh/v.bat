#!/bin/bash

set model=HOG
set image_type=cc

set k_value=30
set dimensionality_reduction_technique=PCA
set images_folder_path=E:/projects/workspace/mwdb-phase2/Submission/Code/tasks/all

set output_folder_path=E:/projects/workspace/mwdb-phase2/Submission/Outputs/Task2
set query_image=E:/projects/workspace/mwdb-phase2/Submission/Outputs/Task1

set subject_id=1
:: python tasks/Task1.py --model %model% --x %image_type% --k %k_value% --dimensionality_reduction_technique %dimensionality_reduction_technique% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%

python tasks/Task2.py --model %model% --x %subject_id% --k %k_value% --dimensionality_reduction_technique %dimensionality_reduction_technique% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%

:: python tasks/Task3_modular.py --model %model% --k %k_value% --dimensionality_reduction_technique %dimensionality_reduction_technique% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%

:: python tasks/Task3.py --model %model% --k %k_value% --dimensionality_reduction_technique %dimensionality_reduction_technique% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%

:: python tasks/Task3.py --model %model% --k %k_value% --dimensionality_reduction_technique %dimensionality_reduction_technique% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%
:: python tasks/Task4.py --model %model% --k %k_value% --dimensionality_reduction_technique %dimensionality_reduction_technique% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%

:: python tasks/Task7.py --query_image %query_image% --latent_semantics_file %latent_semantic_file% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%