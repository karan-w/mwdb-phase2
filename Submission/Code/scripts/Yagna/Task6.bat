#!/bin/bash

set query_image=D:/MWDB/mwdb-phase2/Submission/Code/tasks/all/image-cc-1-1.png
set n=7
set latent_semantics_file=D:\MWDB\mwdb-phase2\Submission\Outputs\Task1\2021-10-24_00-09-02\output.json
set images_folder_path=D:/MWDB/mwdb-phase2/Submission/Code/tasks/all
set output_folder_path=D:/MWDB/mwdb-phase2/Submission/Outputs/Task6
python tasks/Task6.py --query_image %query_image% --n %n% --latent_semantics_file %latent_semantics_file% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%