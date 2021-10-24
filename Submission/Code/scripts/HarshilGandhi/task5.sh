#!/bin/bash
# Script to run task 5 on Harshil's laptop

query_image="/Users/harshilgandhi/Documents/MWDB_Project/mwdb-phase2/data/image-poster-19-1.png"
latent_semantics_file="/Users/harshilgandhi/Documents/MWDB_Project/mwdb-phase2/Submission/Outputs/Task1/2021-10-24_12-45-57/output.json"
n=10
images_folder_path="/Users/harshilgandhi/Documents/MWDB_Project/mwdb-phase2/data"
output_folder_path="/Users/harshilgandhi/Documents/MWDB_Project/mwdb-phase2/Submission/Outputs/Task5"

python3 /Users/harshilgandhi/Documents/MWDB_Project/mwdb-phase2/Submission/Code/tasks/Task5.py --query_image "${query_image}" --latent_semantics_file "${latent_semantics_file}" --n $n --images_folder_path "${images_folder_path}" --output_folder_path "${output_folder_path}"