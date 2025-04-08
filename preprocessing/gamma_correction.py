import nibabel as nib
import numpy as np
import cv2
import os

def compute_lambda(image):
    """ Compute lambda based on image brightness (mean intensity) """
    mean_intensity = np.mean(image) / np.max(image)  # Normalize to [0,1]
    return 1 - mean_intensity  # Higher brightness → Lower lambda, Lower brightness → Higher lambda

def agcwd(image):
    """ Adaptive Gamma Correction with Weighted Distribution (AGCWD) for a single 2D slice """
    lambda_value = compute_lambda(image)  # Compute adaptive lambda

    # Normalize image to [0,1]
    img = image.astype(np.float32) / np.max(image)

    # Compute histogram and PDF
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 1))
    pdf = hist / np.sum(hist)

    # Compute CDF and weighted CDF
    cdf = np.cumsum(pdf)
    m = np.max(cdf)
    weighted_cdf = (1 - np.exp(-lambda_value * cdf / m)) / (1 - np.exp(-lambda_value))

    # Map pixel values to their corresponding gamma correction values
    LUT = np.interp(img.flatten(), np.linspace(0, 1, 256), weighted_cdf)
    gamma_map = np.power(img.flatten(), LUT).reshape(img.shape)

    # Convert back to 8-bit
    enhanced_img = (gamma_map * 255).astype(np.uint8)

    return enhanced_img

def process_brats_mri(input_nii_path, output_nii_path):
    """ Apply AGCWD to all slices in a 3D MRI scan from the BraTS dataset """
    
    # Load the NIfTI file
    mri_nifti = nib.load(input_nii_path)
    mri_data = mri_nifti.get_fdata()  # Get 3D numpy array

    # Normalize MRI to [0, 255] for processing
    mri_data = (mri_data - np.min(mri_data)) / (np.max(mri_data) - np.min(mri_data)) * 255
    mri_data = mri_data.astype(np.uint8)

    # Process each slice with AGCWD
    enhanced_slices = np.array([agcwd(slice) for slice in mri_data], dtype=np.uint8)

    # Convert back to NIfTI format
    enhanced_nifti = nib.Nifti1Image(enhanced_slices, affine=mri_nifti.affine)

    # Save the processed MRI scan
    nib.save(enhanced_nifti, output_nii_path)
    print(f"Saved enhanced MRI scan: {output_nii_path}")

# Example usage
input_nii = r"dataset and backend\\Training1\\BraTS20\\BraTS20_Training_001\\BraTS20_Training_001_flair.nii"  # Replace with your file
output_nii = "BraTS2021_00000_flair_agcwd.nii"

process_brats_mri(input_nii, output_nii)
