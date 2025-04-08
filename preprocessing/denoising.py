import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma

# Load a BraTS 2020 MRI scan (NIfTI format)
nii_img = nib.load('dataset and backend\Training1\BraTS20\BraTS20_Training_001\BraTS20_Training_001_flair.nii')  # Replace with actual file path
mri_scan = nii_img.get_fdata()

# Select a single slice for visualization
slice_idx = mri_scan.shape[2] // 2  # Choose the middle slice
noisy_slice = mri_scan[:, :, slice_idx]

# Estimate noise level
sigma_est = np.mean(estimate_sigma(noisy_slice, channel_axis=None))

# Apply Non-Local Means Denoising
denoised_nlm = denoise_nl_means(noisy_slice, h=1.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=3)

# Plot the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(noisy_slice, cmap="gray")
plt.title("Original Noisy Slice")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(denoised_nlm, cmap="gray")
plt.title("Denoised (NLM)")
plt.axis("off")

plt.show()
