#!/bin/bash
# Script to run task 5 on Karan's laptop

query_image="/Users/karanwadhwani/Documents/ASU/Fall2021/CSE515_MWDB/Project/mwdb-phase2/data/image-poster-19-1.png"
latent_semantics_file="/Users/karanwadhwani/Documents/ASU/Fall2021/CSE515_MWDB/Project/mwdb-phase2/Submission/Outputs/Task1/2021-11-01_08-48-57/output.json"
n=10
images_folder_path="/Users/karanwadhwani/Documents/ASU/Fall2021/CSE515_MWDB/Project/mwdb-phase2/Inputs/tests_v2"
output_folder_path="/Users/karanwadhwani/Documents/ASU/Fall2021/CSE515_MWDB/Project/mwdb-phase2/Submission/Outputs/Task5"

python3 tasks/Task5.py --query_image "${query_image}" --latent_semantics_file "${latent_semantics_file}" --n $n --images_folder_path "${images_folder_path}" --output_folder_path "${output_folder_path}"