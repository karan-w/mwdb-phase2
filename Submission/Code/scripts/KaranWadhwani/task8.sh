#!/bin/bash
# Script to run task 8 on Karan's laptop


# subject_subject_similarity_matrix_filepath, n, m 

subject_subject_similarity_matrix_filepath="/Users/karanwadhwani/Documents/ASU/Fall2021/CSE515_MWDB/Project/mwdb-phase2/Submission/Outputs/Task4/2021-10-24_20-28-58/output.json"
number_of_subjects=40
n=10
m=10
output_folder_path="/Users/karanwadhwani/Documents/ASU/Fall2021/CSE515_MWDB/Project/mwdb-phase2/Submission/Outputs/Task8"

python3.7 tasks/Task8.py --subject_subject_similarity_matrix_filepath "${subject_subject_similarity_matrix_filepath}" --number_of_subjects $number_of_subjects --n $n --m $m --output_folder_path="${output_folder_path}"
