import os
import shutil

# Set paths
brats_root = "dataset and backend\Training1\BraTS20"  # Replace with the path to the BraTS dataset
seg_output_folder = "dataset and backend\seg"  # Replace with your desired output folder
flair_output_folder = "dataset and backend\Flair"  # Replace with your desired output folder

# Create output folders if they don't exist
os.makedirs(seg_output_folder, exist_ok=True)
os.makedirs(flair_output_folder, exist_ok=True)

# Loop through each patient folder
for patient_folder in os.listdir(brats_root):
    print(patient_folder)
    patient_path = os.path.join(brats_root, patient_folder)
    
    if os.path.isdir(patient_path):
        for file in os.listdir(patient_path):
            if file.endswith("_seg.nii"):
                shutil.copy(os.path.join(patient_path, file), os.path.join(seg_output_folder, file))
            elif file.endswith("_flair.nii"):
                shutil.copy(os.path.join(patient_path, file), os.path.join(flair_output_folder, file))

print("Done moving segmentation and flair files.")
