import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage.restoration import denoise_nl_means, estimate_sigma

# === Step 1: Bias Correction ===
def bias_correction(image):
    sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
    mask = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(sitk_image, mask)
    corrected_array = sitk.GetArrayFromImage(corrected_image)
    return np.nan_to_num(corrected_array, nan=0.0)

# === Step 2: Non-Local Means Denoising ===
def nlm_denoising(image):
    if np.count_nonzero(image) == 0:
        return image
    sigma_est = np.mean(estimate_sigma(image, channel_axis=None))
    denoised = denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=3)
    return denoised

# === Step 3: AGCWD ===
def compute_lambda(image):
    mean_intensity = np.mean(image) / np.max(image) if np.max(image) != 0 else 0
    return 1 - mean_intensity

def agcwd(image):
    if np.max(image) == 0:
        return image.astype(np.uint8)
    
    lambda_value = compute_lambda(image)
    img = image.astype(np.float32) / np.max(image)

    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 1))
    pdf = hist / np.sum(hist)
    cdf = np.cumsum(pdf)
    m = np.max(cdf)
    
    weighted_cdf = (1 - np.exp(-lambda_value * cdf / m)) / (1 - np.exp(-lambda_value))
    LUT = np.interp(img.flatten(), np.linspace(0, 1, 256), weighted_cdf)
    
    gamma_map = np.power(img.flatten(), LUT).reshape(img.shape)
    enhanced_img = (gamma_map * 255).astype(np.uint8)
    return enhanced_img

# === Full Pipeline ===
def preprocess_mri_pipeline(input_nii_path, output_nii_path):
    mri_nifti = nib.load(input_nii_path)
    mri_data = mri_nifti.get_fdata()

    # Normalize to [0, 255]
    mri_data = (mri_data - np.min(mri_data)) / (np.max(mri_data) - np.min(mri_data)) * 255

    # Show middle slice steps
    slice_idx = mri_data.shape[2] // 2
    original = mri_data[:, :, slice_idx]
    corrected = bias_correction(original)
    denoised = nlm_denoising(corrected)
    enhanced = agcwd(denoised)

    '''plt.figure(figsize=(20, 5))
    titles = ['Original', 'Bias Corrected', 'Denoised (NLM)', 'Enhanced (AGCWD)']
    imgs = [original, corrected, denoised, enhanced]
    for i, (img, title) in enumerate(zip(imgs, titles)):
        plt.subplot(1, 4, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()'''

    # Process full volume
    processed_volume = []
    for i in range(mri_data.shape[2]):
        slice_2d = mri_data[:, :, i]

        corrected = bias_correction(slice_2d)
        denoised = nlm_denoising(corrected)
        enhanced = agcwd(denoised)

        processed_volume.append(enhanced)
        #print(f"Processed slice {i+1}/{mri_data.shape[2]}")

    processed_volume = np.stack(processed_volume, axis=2)
    processed_nifti = nib.Nifti1Image(processed_volume.astype(np.uint8), affine=mri_nifti.affine)
    nib.save(processed_nifti, output_nii_path)
    print(f"✅ Saved fully preprocessed MRI scan: {output_nii_path}")

# === Example usage ===
input_nii = r"dataset and backend\Training1\BraTS20\BraTS20_Training_001\BraTS20_Training_001_flair.nii"
output_nii = "preprocessed_images\BraTS20_Training_001_flair_preprocessed_full.nii"

preprocess_mri_pipeline(input_nii, output_nii)
