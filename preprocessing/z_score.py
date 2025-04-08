import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def min_max_normalization(image_path, min_val=0, max_val=1):
    """Normalize an MRI image using Min-Max scaling."""
    img = nib.load(image_path).get_fdata()
    
    # Extract non-zero pixels (ignoring background)
    nonzero_pixels = img[img > 0]
    
    # Compute min and max of non-zero pixels
    min_pixel = np.min(nonzero_pixels)
    max_pixel = np.max(nonzero_pixels)
    
    # Apply Min-Max Normalization
    img_norm = (img - min_pixel) / (max_pixel - min_pixel) * (max_val - min_val) + min_val
    
    # Keep background as 0
    img_norm[img == 0] = 0
    
    return img_norm

def display_image(image_path):
    """Displays the middle slice of the normalized MRI scan."""
    img_norm = min_max_normalization(image_path)

    middle_slice = img_norm.shape[2] // 2  # Select the middle slice along the Z-axis

    plt.figure(figsize=(6, 6))
    plt.imshow(img_norm[:, :, middle_slice], cmap="gray")
    plt.title("Min-Max Normalized MRI Slice")
    plt.axis("off")
    plt.colorbar()
    plt.show()

# Example usage
image_path = r"C:\\Users\\crdik\\Downloads\\mpw\\dataset and backend\\Training1\\BraTS20\\BraTS20_Training_001\\BraTS20_Training_001_flair.nii" # Replace with your file path
display_image(image_path)
