
#!/bin/bash
# Script to run task 9 on Harshil's laptop

n=2
m=10
input_subjects=1,2,3
sub_sub_matrix_path="/Users/harshilgandhi/Documents/MWDB_Project/mwdb-phase2/Submission/Outputs/Task4/2021-10-24_20-28-58/output.json"
output_folder_path="/Users/harshilgandhi/Documents/MWDB_Project/mwdb-phase2/Submission/Outputs/Task9"

python3 /Users/harshilgandhi/Documents/MWDB_Project/mwdb-phase2/Submission/Code/tasks/Task9.py --n $n --m $m --input_subjects $input_subjects --sub_sub_matrix_path "${sub_sub_matrix_path}" --output_folder_path "${output_folder_path}"
