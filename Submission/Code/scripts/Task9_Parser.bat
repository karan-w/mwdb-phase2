#!/bin/bash

set n=3
set m=10
set input_subjects=10,21,30
set output_folder_path=E:/projects/workspace/mwdb-phase2/Submission/Outputs/Task9
python tasks/Task9_Parsers.py --n %n% --m %m% --input_subjects %input_subjects% --output_folder_path %output_folder_path%
