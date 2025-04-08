import nibabel as nib
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import kurtosis, skew

# Load NIfTI files
def load_nifti(file_path):
    return nib.load(file_path).get_fdata()

# File paths (Replace with actual paths)
flair = load_nifti("path_to_flair.nii")
t1 = load_nifti("path_to_t1.nii.gz")
t1ce = load_nifti("path_to_t1ce.nii.gz")
t2 = load_nifti("path_to_t2.nii.gz")
seg = load_nifti("path_to_seg.nii.gz")  # Segmentation map

def extract_histogram_features(image):
    values = image[image > 0]  # Exclude background
    return np.mean(values), np.var(values), skew(values.flatten()), kurtosis(values.flatten())

def extract_glcm_features(image):
    image = (image / np.max(image) * 255).astype(np.uint8)  # Normalize to 8-bit
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return contrast, correlation, energy, homogeneity

def extract_edge_features(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.mean(edge_magnitude), np.var(edge_magnitude)

def extract_tumor_features(image, segmentation):
    tumor_region = image[segmentation > 0]
    if len(tumor_region) == 0:
        return 0, 0  # Avoid empty tumor regions
    return np.mean(tumor_region), np.var(tumor_region)

# Extract features
mean_flair, var_flair, skew_flair, kurt_flair = extract_histogram_features(flair)
contrast_flair, correlation_flair, energy_flair, homogeneity_flair = extract_glcm_features(flair)
edge_mean_flair, edge_var_flair = extract_edge_features(flair)
tumor_mean_flair, tumor_var_flair = extract_tumor_features(flair, seg)

# Compute GLISTR Parameters
D = (mean_flair + var_flair) / (edge_mean_flair + 1e-6)  # Diffusion Coefficient
rho = (tumor_mean_flair + tumor_var_flair) / (homogeneity_flair + 1e-6)  # Proliferation Rate
u0 = mean_flair + contrast_flair + energy_flair  # Initial Tumor Distribution
R_x = (correlation_flair + kurt_flair) / (edge_var_flair + 1e-6)  # Resistance Function

# Print Results
print(f"Diffusion Coefficient (D): {D}")
print(f"Proliferation Rate (ρ): {rho}")
print(f"Initial Tumor Distribution (u₀): {u0}")
print(f"Resistance Function (R(x)): {R_x}")
