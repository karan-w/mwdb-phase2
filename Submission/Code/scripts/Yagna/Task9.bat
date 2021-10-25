#!/bin/bash

set n=3
set m=10
set input_subjects=10,21,30
set sub_sub_matrix_path=D:\MWDB\mwdb-phase2\Submission\Outputs\Task4\2021-10-24_23-17-29\output.json
set images_folder_path=D:/MWDB/mwdb-phase2/Submission/Code/tasks/all
set output_folder_path=D:/MWDB/mwdb-phase2/Submission/Outputs/Task9
python tasks/Task9.py --n %n% --m %m% --input_subjects %input_subjects% --output_folder_path %output_folder_path% --sub_sub_matrix_path %sub_sub_matrix_path%