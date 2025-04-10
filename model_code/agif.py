import os
import cv2
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import imageio

# Load Richards growth parameters from CSV
df_params = pd.read_csv(r"model_code\global_richards_params.csv")
K = df_params['Value'][0]
B = df_params['Value'][1]
Q = df_params['Value'][2]
M = df_params['Value'][3]
nu = df_params['Value'][4]

# --- Richards Growth Function ---
def richards_growth(t, K, B, Q, M, nu):
    return K * (1 + Q * np.exp(-B * (t - M))) ** (-1 / nu)

# --- Segment Tumor (for initial mask if needed) ---
def segment_tumor(feature_slice):
    norm_map = cv2.normalize(feature_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary_mask = cv2.threshold(norm_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return binary_mask

# --- Calculate Tumor Area ---
def get_tumor_area(binary_mask):
    return np.sum(binary_mask > 0)

# --- Simulate Tumor Growth ---
def generate_growth_from_volume(original_slice, tumor_mask, V0, time_steps):
    images = []
    
    # Calculate initial area
    initial_area = get_tumor_area(tumor_mask)
    
    for t in range(time_steps):
        # Calculate target volume using Richards growth
        target_volume = richards_growth(t, K, B, Q, M, nu)
        
        # Calculate scale factor (how much to grow from initial)
        scale = target_volume / V0
        
        # Estimate morph dilation steps based on area scaling
        iterations = int(np.sqrt(scale))  # assuming area scales with square root
        
        # Apply dilation
        grown_mask = cv2.dilate(tumor_mask.copy(), np.ones((3, 3), np.uint8), iterations=iterations)
        
        # Create overlay
        overlay = original_slice.copy()
        if len(overlay.shape) == 2 or overlay.shape[2] == 1:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
        
        overlay = overlay / 255.0
        overlay[grown_mask > 0] = [1, 0, 0]
        images.append(overlay)
    
    return images

# --- Save Time-Series as images in a folder ---
def save_as_images(images, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    for i, img in enumerate(images):
        img_path = os.path.join(folder_path, f"frame_{i}.png")
        img_uint8 = (img * 255).astype(np.uint8)
        cv2.imwrite(img_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
    print(f"[✓] Saved time-series images to: {folder_path}")

# --- MAIN PIPELINE ---
def simulate_from_csv(csv_path, nii_dir, mask_dir, output_dir, time_steps=10):
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    for _, row in df.iterrows():
        patient_id = str(row['Patient_ID'])
        print(patient_id)

        # Load brain slice and mask
        img_path = os.path.join(nii_dir, f"{patient_id}_flair_preprocessed_full.nii")
        mask_path = os.path.join(mask_dir, f"{patient_id}_seg.nii")

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print(f"[!] Skipping {patient_id} — Missing NIfTI file.")
            continue

        # Load slices
        img_nii = nib.load(img_path)
        mask_nii = nib.load(mask_path)
        img = img_nii.get_fdata()
        mask = mask_nii.get_fdata()

        # Pick middle slice
        slice_index = img.shape[2] // 2
        brain_slice = img[:, :, slice_index]
        tumor_mask = (mask[:, :, slice_index] > 0).astype(np.uint8) * 255

        # Normalize brain slice for display
        brain_slice = cv2.normalize(brain_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Get initial tumor volume from CSV
        V0 = row["Tumor_Volume_mm3"]

        # Simulate growth using Richards model
        images = generate_growth_from_volume(brain_slice, tumor_mask, V0, time_steps)
        patient_folder = os.path.join(output_dir, patient_id)
        save_as_images(images, patient_folder)

# Example usage:
simulate_from_csv(
    csv_path=r"FE\tumor_features.csv",
    nii_dir=r"preprocessed_images",
    mask_dir=r"dataset and backend\seg",
    output_dir=r"synthetic images",
    time_steps=12
)