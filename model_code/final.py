import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import curve_fit
from glob import glob

# Step 1: Estimate logistic model parameters from tumor_features.csv
def logistic(t, K, r, A):
    return K / (1 + A * np.exp(-r * t))

df = pd.read_csv(r"FE\tumor_features.csv")
volumes = df["Tumor_Volume_mm3"].values

# Normalize for fitting
volumes_norm = (volumes - volumes.min()) / (volumes.max() - volumes.min())
pseudo_t = np.linspace(0, 1, len(volumes))
initial_guess = [1.0, 5.0, 1.0]

params, _ = curve_fit(logistic, pseudo_t, volumes_norm, p0=initial_guess, maxfev=10000)
K_norm, r, A = params
V_max = volumes.max()
V_min = volumes.min()
K = K_norm * (V_max - V_min) + V_min
V0 = K / (A + 1)

print(f"Estimated Parameters:\nV0 = {V0:.2f}, V_max = {K:.2f}, r = {r:.4f}")

# Step 2: Logistic growth model
def logistic_growth(V0, V_max, r, t):
    return V_max / (1 + ((V_max - V0) / V0) * np.exp(-r * t))

# Step 3: Synthetic data generation
def generate_synthetic_data(image, time_steps, V0, V_max, r):
    generated_images = []
    for t in range(time_steps):
        tumor_area = logistic_growth(V0, V_max, r, t)
        tumor_mask = np.zeros_like(image, dtype=np.uint8)
        tumor_mask = np.ascontiguousarray(tumor_mask)
        radius = int(np.sqrt(tumor_area / np.pi))
        center = (image.shape[1] // 2, image.shape[0] // 2)
        cv2.circle(tumor_mask, center, radius, 1, -1)

        synthetic_image = image.copy().astype(np.float32)
        synthetic_image = (synthetic_image - synthetic_image.min()) / (np.ptp(synthetic_image) + 1e-5)
        synthetic_image[tumor_mask == 1] = 1.0  # highlight tumor area

        generated_images.append(synthetic_image)
    return generated_images

# Step 4: Process each image
flair_files = glob("preprocessed_images/*_flair_preprocessed_full.nii")

output_folder = "synthetic_outputs"
os.makedirs(output_folder, exist_ok=True)

for file_path in flair_files:
    file_name = os.path.basename(file_path).replace(".nii", "")
    print(f"Processing: {file_name}")

    # Load image
    nii = nib.load(file_path)
    flair_data = nii.get_fdata()
    slice_idx = flair_data.shape[2] // 2
    flair_slice = flair_data[:, :, slice_idx]

    # Generate synthetic images
    synthetic_images = generate_synthetic_data(flair_slice, time_steps=10, V0=V0, V_max=K, r=r)

    # Save as PNG sequence
    patient_folder = os.path.join(output_folder, file_name)
    os.makedirs(patient_folder, exist_ok=True)

    for i, img in enumerate(synthetic_images):
        plt.imsave(f"{patient_folder}/step_{i}.png", img, cmap="gray")

print("✅ Synthetic tumor progression images saved.")
