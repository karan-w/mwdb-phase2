#!/bin/bash
# Script to run task 7 on Harshil's laptop

query_image="/Users/harshilgandhi/Documents/MWDB_Project/mwdb-phase2/data/image-poster-19-1.png"
latent_semantics_file="/Users/harshilgandhi/Documents/MWDB_Project/mwdb-phase2/Submission/Outputs/Task1/2021-10-24_22-02-54/output.json"
images_folder_path="/Users/harshilgandhi/Documents/MWDB_Project/mwdb-phase2/data"
output_folder_path="/Users/harshilgandhi/Documents/MWDB_Project/mwdb-phase2/Submission/Outputs/Task5"

python3 /Users/harshilgandhi/Documents/MWDB_Project/mwdb-phase2/Submission/Code/tasks/Task7.py --query_image "${query_image}" --latent_semantics_file "${latent_semantics_file}" --images_folder_path "${images_folder_path}" --output_folder_path "${output_folder_path}"
