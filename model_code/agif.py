import os
import cv2
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import imageio

df = pd.read_csv(r"model_code\global_richards_params.csv")
K = df['Value'][0]
B = df['Value'][1]
Q = df['Value'][2]
M = df['Value'][3]
nu = df['Value'][4]
# --- Richards Growth Function (Parameter-based) ---
def richards_growth_param(t, K, B, Q, M, nu):
    return K * (1 + Q * np.exp(-B * (t - M))) ** (-1 / nu)

# --- Segment Tumor (if needed) ---
def segment_tumor(feature_slice):
    norm_map = cv2.normalize(feature_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary_mask = cv2.threshold(norm_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return binary_mask

# --- Calculate Tumor Area ---
def get_tumor_area(binary_mask):
    return np.sum(binary_mask > 0)

# --- Simulate Tumor Growth (Biologically realistic) ---
def generate_growth_from_volume(original_slice, tumor_mask, V0, V_max, r, time_steps):
    images = []

    # Assume spherical tumor: V ∝ r³
    r0 = V0 ** (1 / 3)
    pixels_per_mm = 1.0  # adjust if you know pixel spacing from .nii header

    for t in range(time_steps):
        Vt = richards_growth_param(t, K, B, Q, M, nu)
        rt = Vt ** (1 / 3)
        delta_r = rt - r0
        iterations = max(1, int(delta_r * pixels_per_mm))

        kernel = np.ones((3, 3), np.uint8)
        grown_mask = cv2.dilate(tumor_mask.copy(), kernel, iterations=iterations)

        overlay = original_slice.copy()
        if len(overlay.shape) == 2 or overlay.shape[2] == 1:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
        overlay = overlay / 255.0

        # Color tumor region red
        overlay[grown_mask > 0] = [1, 0, 0]
        images.append(overlay)

    return images

# --- Save Images ---
def save_as_images(images, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    for i, img in enumerate(images):
        img_path = os.path.join(folder_path, f"frame_{i}.png")
        img_uint8 = (img * 255).astype(np.uint8)
        cv2.imwrite(img_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
    print(f"[✓] Saved time-series images to: {folder_path}")

# --- Save as GIF ---
def save_growth_as_gif(images, output_path, duration=0.5):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames = [(img * 255).astype(np.uint8) for img in images]
    imageio.mimsave(output_path, frames, format='GIF', duration=duration)
    print(f"[✓] Saved tumor growth GIF to: {output_path}")

# --- Main Pipeline ---
def simulate_from_csv(csv_path, nii_dir, mask_dir, output_dir, time_steps=10):
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    for _, row in df.iterrows():
        patient_id = str(row['Patient_ID'])
        print(f"\nProcessing Patient: {patient_id}")

        img_path = os.path.join(nii_dir, f"{patient_id}_flair_preprocessed_full.nii")
        mask_path = os.path.join(mask_dir, f"{patient_id}_seg.nii")

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print(f"[!] Skipping {patient_id} — Missing NIfTI file.")
            continue

        img_nii = nib.load(img_path)
        mask_nii = nib.load(mask_path)
        img = img_nii.get_fdata()
        mask = mask_nii.get_fdata()

        slice_index = img.shape[2] // 2
        brain_slice = img[:, :, slice_index]
        tumor_mask = (mask[:, :, slice_index] > 0).astype(np.uint8) * 255

        brain_slice = cv2.normalize(brain_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        V0 = row["Tumor_Volume_mm3"]
        V_max = V0 * 2.5
        r = 0.1 if row["Sphericity"] > 0.5 else 0.05

        images = generate_growth_from_volume(brain_slice, tumor_mask, V0, V_max, r, time_steps)

        patient_folder = os.path.join(output_dir, patient_id)
        save_as_images(images, patient_folder)

        gif_path = os.path.join(patient_folder, "tumor_growth.gif")
        save_growth_as_gif(images, gif_path, duration=0.5)

# -------------------
# 🔁 Example usage:
simulate_from_csv(
    csv_path="FE\\tumor_features.csv",
    nii_dir="preprocessed_images",
    mask_dir="dataset and backend\\seg",
    output_dir="synthetic images",
    time_steps=12
)
