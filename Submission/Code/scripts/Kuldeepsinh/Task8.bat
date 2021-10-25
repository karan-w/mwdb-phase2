#!/bin/bash

set subject_subject_similarity_matrix_filepath=E:\projects\workspace\mwdb-phase2\Submission\Outputs\Task4\2021-10-24_23-17-29\output.json
set number_of_subjects=40
set n=5
set m=10
set images_folder_path=E:/projects/workspace/mwdb-phase2/Submission/Code/tasks/all
set output_folder_path=E:/projects/workspace/mwdb-phase2/Submission/Outputs/Task8
python tasks/Task8.py --subject_subject_similarity_matrix_filepath %subject_subject_similarity_matrix_filepath% --number_of_subjects %number_of_subjects% --n %n% --m %m%  --output_folder_path %output_folder_path%