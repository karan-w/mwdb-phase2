set model=HOG
set x=1
set image_type=cc
set k_value=30
set dimensionality_reduction_technique=PCA

set query_image=E:/projects/workspace/mwdb-phase2/Submission/Code/tasks/all/image-cc-1-1.png
set latent_semantics_file=E:/projects/workspace/mwdb-phase2/Submission/Outputs/Task2/2021-10-24_03-44-42/output.json
set n=10

set output_folder_path=E:/projects/workspace/mwdb-phase2/Submission/Outputs/Task7
::set query_image=E:/projects/workspace/mwdb-phase2/Submission/Outputs/Task1

:: python tasks/Task1.py --model %model% --x %image_type% --k %k_value% --dimensionality_reduction_technique %dimensionality_reduction_technique% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%

:: python tasks/Task2.py --model %model% --x %x% --k %k_value% --dimensionality_reduction_technique %dimensionality_reduction_technique% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%

:: python tasks/Task3_modular.py --model %model% --k %k_value% --dimensionality_reduction_technique %dimensionality_reduction_technique% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%

:: python tasks/Task4.py --model %model% --k %k_value% --dimensionality_reduction_technique %dimensionality_reduction_technique% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%

python tasks/Task7.py --query_image %query_image% --latent_semantics_file %latent_semantics_file% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%
:: python tasks/Task5.py --query_image %query_image% --latent_semantics_file %latent_semantics_file% --n %n% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%

python tasks/Task6.py --query_image %query_image% --latent_semantics_file %latent_semantics_file% --n %n% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%
