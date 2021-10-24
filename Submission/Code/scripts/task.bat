set model=HOG
set image_type=cc
set /A k_value=7
set dimensionality_reduction_technique=PCA
set query_image=C:\Users\sshah96\Downloads\phase2_data_revised\all\image-cc-1-1.png
set latent_semantics_file=C:\Users\sshah96\Desktop\GitHub\mwdb-phase2\Submission\Outputs\Task1\2021-10-23_18-40-00\output.json
set n=10
set images_folder_path=C:\Users\sshah96\Downloads\phase2_data_revised\all

set output_folder_path=C:\Users\sshah96\Desktop\GitHub\mwdb-phase2\Submission\Outputs\Task1

:: python tasks/Task1.py --model %model% --x %image_type% --k %k_value% --dimensionality_reduction_technique %dimensionality_reduction_technique% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%

:: python tasks/Task3_modular.py --model %model% --k %k_value% --dimensionality_reduction_technique %dimensionality_reduction_technique% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%

:: python tasks/Task4.py --model %model% --k %k_value% --dimensionality_reduction_technique %dimensionality_reduction_technique% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%

python tasks/Task5.py --query_image %query_image% --latent_semantics_file %latent_semantics_file% --n %n% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%