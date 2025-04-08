# import nibabel as nib
# import numpy as np
# import pandas as pd
# import os
# from skimage.measure import marching_cubes

# # Define file paths
# base_path = r"C:\Users\gowri\OneDrive\Desktop\FE\Images"  # Change this to your dataset path
# main_path = r"C:\Users\gowri\OneDrive\Desktop\FE"
# patient_id = "BraTS20_Training_001"

# # Load all NIfTI files
# modalities = ["t1", "t1ce", "t2", "flair", "seg"]
# nii_files = {mod: os.path.join(base_path, f"{patient_id}_{mod}.nii") for mod in modalities}
# nii_data = {mod: nib.load(nii_files[mod]).get_fdata() for mod in modalities}

# # Load segmentation mask
# mask = nii_data["seg"]
# voxel_size = np.prod(nib.load(nii_files["seg"]).header.get_zooms())  # Get voxel volume in mm³

# # ---------------------- Feature Extraction ----------------------

# # Compute Tumor Volume & Surface Area
# tumor_voxels = np.sum(mask > 0)  # Number of voxels in the tumor
# tumor_volume = tumor_voxels * voxel_size  # Total volume in mm³

# # Compute tumor surface area using the Marching Cubes algorithm
# verts, faces, _, _ = marching_cubes(mask, level=0.5)
# tumor_surface_area = len(faces) * voxel_size ** (2/3)  # Approximate surface area in mm²

# # Compute Sphericity
# sphericity = (np.pi ** (1/3) * (6 * tumor_volume) ** (2/3)) / tumor_surface_area

# # Compute Bounding Box Dimensions
# coords = np.argwhere(mask > 0)
# min_coords, max_coords = coords.min(axis=0), coords.max(axis=0)
# bbox_size = (max_coords - min_coords) * nib.load(nii_files["seg"]).header.get_zooms()
# bbox_size_str = f"{bbox_size[0]:.2f} x {bbox_size[1]:.2f} x {bbox_size[2]:.2f} mm"

# # 2️⃣ Extract Intensity Features
# def extract_intensity_features(image, mask):
#     tumor_region = image[mask > 0]  # Extract values only from the tumor region
#     return {
#         "mean": np.mean(tumor_region),
#         "std": np.std(tumor_region),
#         "min": np.min(tumor_region),
#         "max": np.max(tumor_region),
#         "percentile_25": np.percentile(tumor_region, 25),
#         "percentile_50": np.percentile(tumor_region, 50),
#         "percentile_75": np.percentile(tumor_region, 75),
#     }

# # Compute intensity features for each MRI modality
# intensity_features = {mod: extract_intensity_features(nii_data[mod], mask) for mod in ["t1", "t1ce", "t2", "flair"]}

# # ---------------------- Save Features to CSV ----------------------

# # Create a dictionary for DataFrame
# feature_dict = {
#     "Patient_ID": patient_id,
#     "Tumor_Volume_mm3": tumor_volume,
#     "Tumor_Surface_Area_mm2": tumor_surface_area,
#     "Sphericity": sphericity,
#     "Bounding_Box_Dimensions": bbox_size_str,
# }

# # Add intensity features
# for mod, features in intensity_features.items():
#     for key, value in features.items():
#         feature_dict[f"{mod}_{key}"] = value

# # Convert to DataFrame and Save to CSV
# df = pd.DataFrame([feature_dict])
# csv_path = os.path.join(main_path, "tumor_features.csv")
# df.to_csv(csv_path, index=False)

# print(f"✅ Features saved to: {csv_path}")


import nibabel as nib
import numpy as np
import pandas as pd
import os
from skimage.measure import marching_cubes

# Define file paths
base_path = r"C:\Users\gowri\OneDrive\Desktop\FE\Images"  # Folder containing patient subfolders
main_path = r"C:\Users\gowri\OneDrive\Desktop\FE"
csv_path = os.path.join(main_path, "tumor_features.csv")

# List all patient folders
patient_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

# Store all extracted features
all_features = []

for patient_id in patient_folders:
    patient_path = os.path.join(base_path, patient_id)
    modalities = ["t1", "t1ce", "t2", "flair", "seg"]
    nii_files = {mod: os.path.join(patient_path, f"{patient_id}_{mod}.nii") for mod in modalities}
    
    # Ensure all required files exist
    if not all(os.path.exists(nii_files[mod]) for mod in modalities):
        print(f"❌ Missing files for {patient_id}, skipping...")
        continue
    
    nii_data = {mod: nib.load(nii_files[mod]).get_fdata() for mod in modalities}
    
    # Load segmentation mask
    mask = nii_data["seg"]
    voxel_size = np.prod(nib.load(nii_files["seg"]).header.get_zooms())  # Get voxel volume in mm³

    # ---------------------- Feature Extraction ----------------------

    # Compute Tumor Volume & Surface Area
    tumor_voxels = np.sum(mask > 0)  # Number of voxels in the tumor
    tumor_volume = tumor_voxels * voxel_size  # Total volume in mm³
    
    # Compute tumor surface area using the Marching Cubes algorithm
    verts, faces, _, _ = marching_cubes(mask, level=0.5)
    tumor_surface_area = len(faces) * voxel_size ** (2/3)  # Approximate surface area in mm²

    # Compute Sphericity
    sphericity = (np.pi ** (1/3) * (6 * tumor_volume) ** (2/3)) / tumor_surface_area
    
    # Compute Bounding Box Dimensions
    coords = np.argwhere(mask > 0)
    min_coords, max_coords = coords.min(axis=0), coords.max(axis=0)
    bbox_size = (max_coords - min_coords) * nib.load(nii_files["seg"]).header.get_zooms()
    bbox_size_str = f"{bbox_size[0]:.2f} x {bbox_size[1]:.2f} x {bbox_size[2]:.2f} mm"

    # 2️⃣ Extract Intensity Features
    def extract_intensity_features(image, mask):
        tumor_region = image[mask > 0]  # Extract values only from the tumor region
        return {
            "mean": np.mean(tumor_region),
            "std": np.std(tumor_region),
            "min": np.min(tumor_region),
            "max": np.max(tumor_region),
            "percentile_25": np.percentile(tumor_region, 25),
            "percentile_50": np.percentile(tumor_region, 50),
            "percentile_75": np.percentile(tumor_region, 75),
        }

    # Compute intensity features for each MRI modality
    intensity_features = {mod: extract_intensity_features(nii_data[mod], mask) for mod in ["t1", "t1ce", "t2", "flair"]}

    # ---------------------- Store Extracted Features ----------------------

    feature_dict = {
        "Patient_ID": patient_id,
        "Tumor_Volume_mm3": tumor_volume,
        "Tumor_Surface_Area_mm2": tumor_surface_area,
        "Sphericity": sphericity,
        "Bounding_Box_Dimensions": bbox_size_str,
    }

    # Add intensity features
    for mod, features in intensity_features.items():
        for key, value in features.items():
            feature_dict[f"{mod}_{key}"] = value

    # Append features to the list
    all_features.append(feature_dict)

# Convert to DataFrame and Save to CSV
if all_features:
    df = pd.DataFrame(all_features)
    df.to_csv(csv_path, index=False)
    print(f"✅ Features saved to: {csv_path}")
else:
    print("❌ No valid patient data found!")