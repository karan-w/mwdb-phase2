#!/bin/bash

set n=3
set m=10
set input_subjects=10,21,30
set model=HOG
set image_type=cc
set k_value=7
set dimensionality_reduction_technique=LDA
set sub_sub_matrix_path=D:\MWDB\mwdb-phase2\Submission\Outputs\Task4\output.json
set images_folder_path=D:/MWDB/mwdb-phase2/Submission/Code/tasks/all
set output_folder_path=D:/MWDB/mwdb-phase2/Submission/Outputs/Task9
python tasks/Task9.py --n %n% --m %m% --input_subjects %input_subjects% --output_folder_path %output_folder_path% --sub_sub_matrix_path %sub_sub_matrix_path%
::python tasks/Task1.py --model %model% --x %image_type% --k %k_value% --dimensionality_reduction_technique %dimensionality_reduction_technique% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%