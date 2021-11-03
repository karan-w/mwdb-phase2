#!/bin/bash
# Script to run task 6 on Karan's laptop

query_image="/Users/karanwadhwani/Documents/ASU/Fall2021/CSE515_MWDB/Project/mwdb-phase2/data/image-poster-19-1.png"
latent_semantics_file="/Users/karanwadhwani/Documents/ASU/Fall2021/CSE515_MWDB/Project/mwdb-phase2/Submission/Outputs/Task1/2021-10-24_16-48-10/output.json"
images_folder_path="/Users/karanwadhwani/Documents/ASU/Fall2021/CSE515_MWDB/Project/mwdb-phase2/Inputs/tests_v2"
output_folder_path="/Users/karanwadhwani/Documents/ASU/Fall2021/CSE515_MWDB/Project/mwdb-phase2/Submission/Outputs/Task6"

python3 tasks/Task6.py --query_image "${query_image}" --latent_semantics_file "${latent_semantics_file}" --images_folder_path "${images_folder_path}" --output_folder_path "${output_folder_path}"
