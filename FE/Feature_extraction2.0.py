import nibabel as nib
import numpy as np
import pandas as pd
import os
from skimage.measure import marching_cubes

# Define folders
flair_folder = r"C:\Users\gowri\OneDrive\Desktop\FE\FLAIR"
seg_folder = r"C:\Users\gowri\OneDrive\Desktop\FE\seg"
main_path = r"C:\Users\gowri\OneDrive\Desktop\FE"
csv_path = os.path.join(main_path, "tumor_features.csv")

# Collect all FLAIR files
flair_files = [f for f in os.listdir(flair_folder) if f.endswith(".nii") and "flair" in f.lower()]

all_features = []

for flair_file in flair_files:
    #patient_id = flair_file.split("_")[0]  # Assumes format: PatientID_flair.nii
    patient_id = flair_file.replace("_flair.nii", "")
    flair_path = os.path.join(flair_folder, flair_file)
    
    seg_file = f"{patient_id}_seg.nii"
    seg_path = os.path.join(seg_folder, seg_file)

    if not os.path.exists(seg_path):
        print(f"❌ Missing segmentation for {patient_id}, skipping...")
        continue

    # Load images
    flair_img = nib.load(flair_path).get_fdata()
    seg_img = nib.load(seg_path)
    mask = seg_img.get_fdata()
    voxel_size = np.prod(seg_img.header.get_zooms())

    # Tumor volume
    tumor_voxels = np.sum(mask > 0)
    tumor_volume = tumor_voxels * voxel_size

    # Tumor surface area
    verts, faces, _, _ = marching_cubes(mask, level=0.5)
    tumor_surface_area = len(faces) * voxel_size ** (2 / 3)

    # Sphericity
    sphericity = (np.pi ** (1/3) * (6 * tumor_volume) ** (2/3)) / tumor_surface_area

    # Bounding box
    coords = np.argwhere(mask > 0)
    min_coords, max_coords = coords.min(axis=0), coords.max(axis=0)
    bbox_size = (max_coords - min_coords) * seg_img.header.get_zooms()
    bbox_str = f"{bbox_size[0]:.2f} x {bbox_size[1]:.2f} x {bbox_size[2]:.2f} mm"

    # Intensity features from FLAIR
    tumor_region = flair_img[mask > 0]
    intensity = {
        "mean": np.mean(tumor_region),
        "std": np.std(tumor_region),
        "min": np.min(tumor_region),
        "max": np.max(tumor_region),
        "percentile_25": np.percentile(tumor_region, 25),
        "percentile_50": np.percentile(tumor_region, 50),
        "percentile_75": np.percentile(tumor_region, 75),
    }

    feature_dict = {
        "Patient_ID": patient_id,
        "Tumor_Volume_mm3": tumor_volume,
        "Tumor_Surface_Area_mm2": tumor_surface_area,
        "Sphericity": sphericity,
        "Bounding_Box_Dimensions": bbox_str,
    }

    for key, value in intensity.items():
        feature_dict[f"flair_{key}"] = value

    all_features.append(feature_dict)

# Save to CSV
if all_features:
    df = pd.DataFrame(all_features)
    df.to_csv(csv_path, index=False)
    print(f"✅ Features saved to: {csv_path}")
else:
    print("❌ No valid FLAIR + segmentation pairs found!")
