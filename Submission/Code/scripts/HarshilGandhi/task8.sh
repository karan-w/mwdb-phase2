#!/bin/bash
# Script to run task 8 on Harshil's laptop

subject_subject_similarity_matrix_filepath="/Users/harshilgandhi/Documents/MWDB_Project/mwdb-phase2/Submission/Outputs/Task4/2021-10-24_20-28-58/output.json"
number_of_subjects=4
n=2
m=10
output_folder_path="/Users/harshilgandhi/Documents/MWDB_Project/mwdb-phase2/Submission/Outputs/Task8"

python3 /Users/harshilgandhi/Documents/MWDB_Project/mwdb-phase2/Submission/Code/tasks/Task8.py --subject_subject_similarity_matrix_filepath "${subject_subject_similarity_matrix_filepath}" --number_of_subjects $number_of_subjects --n $n --m $m --output_folder_path "${output_folder_path}"
