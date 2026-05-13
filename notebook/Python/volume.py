
# Cell 1: Volume loading & visualization

import os
import numpy as np
import imageio
import matplotlib.pyplot as plt


# Paths

# Path where processed outputs (patches, figures) will be stored.
# This separates raw data from derived artifacts, which is good practice in ML pipelines.
dataset_path = "/home/hpc/b292dc/b292dc16/vesuvius_project/data"

# Path to the actual CT scan slices (TIFF images).
# This folder contains sequential cross-sectional slices of the papyrus scroll.
volume_folder = "/home/atuin/b292dc/b292dc10/Herculaneum/Scroll1/PHercParis4.volpkg/volumes/20230205180739"



# Load first 50 slices


# Sort filenames to preserve correct anatomical ordering of slices.
# This is critical because CT volumes rely on consistent slice ordering.
files = sorted(os.listdir(volume_folder))[:50]

# Load each slice as a 2D image.
# imageio is used instead of cv2 for better compatibility with TIFF formats.
volume = [imageio.v2.imread(os.path.join(volume_folder, f)) for f in files]

# Stack list of 2D images into a 3D NumPy array.
# Resulting shape: (depth, height, width)
volume = np.stack(volume).astype(np.float32)



# Normalize volume


# Normalize intensity values to [0,1].
# Purpose:
# - ensures consistent intensity scaling
# - improves numerical stability
# - prevents bias in downstream learning algorithms
volume = (volume - volume.min()) / (volume.max() - volume.min())

print("Volume shape:", volume.shape)



# Visualize first and middle slices


# Visualization serves as a sanity check:
# - verifies correct loading
# - reveals structural patterns (fibers, layers)
plt.figure(figsize=(10,5))

# First slice (surface region)
plt.subplot(1,2,1)
plt.imshow(volume[0], cmap="gray")
plt.title("First slice")
plt.axis("off")

# Middle slice (internal structure)
plt.subplot(1,2,2)
plt.imshow(volume[volume.shape[0]//2], cmap="gray")
plt.title("Middle slice")
plt.axis("off")

plt.show()



# Save example slices


# Save visualizations for documentation and reporting.
# Important for reproducibility and paper figures.
figures_path = "/home/hpc/b292dc/b292dc16/vesuvius_project/figures"
os.makedirs(figures_path, exist_ok=True)

plt.imsave(os.path.join(figures_path, "slice_first.png"), volume[0], cmap="gray")
plt.imsave(os.path.join(figures_path, "slice_middle.png"), volume[volume.shape[0]//2], cmap="gray")

print("Saved example CT slices in figures folder")


# Cell 2: Patch extraction (10k) & saving


# Import required libraries
import os                     # for file and directory operations
import numpy as np           # for numerical operations and arrays
import matplotlib.pyplot as plt  # for saving visualization images


# Parameters


# Size of each patch (128x128 pixels)
# This is a design choice balancing:
# - enough local texture information
# - manageable memory usage
patch_size = 128

# Maximum number of patches to extract
# This limits dataset size for faster experimentation
max_patches = 10000

# Path where extracted patches will be saved
patches_path = "/home/hpc/b292dc/b292dc16/vesuvius_project/data/patches_small.npy"

# Ensure the directory exists before saving
# exist_ok=True prevents error if folder already exists
os.makedirs(os.path.dirname(patches_path), exist_ok=True)


# Extract patches


# Initialize empty list to store extracted patches
patches_small = []

# Loop over depth (z-axis) → iterate through CT slices
for z in range(volume.shape[0]):
    
    # Select one 2D slice from the 3D volume
    img = volume[z]

    # Slide window vertically across the image
    # Step size = patch_size → non-overlapping patches
    for y in range(0, img.shape[0] - patch_size, patch_size):

        # Slide window horizontally across the image
        for x in range(0, img.shape[1] - patch_size, patch_size):

            # Extract patch using slicing
            # Region: [y:y+128, x:x+128]
            patch = img[y:y+patch_size, x:x+patch_size]

            # Append patch to list
            patches_small.append(patch)

            # Stop if we reached maximum number of patches
            if len(patches_small) >= max_patches:
                break

        # Break outer loop if limit reached
        if len(patches_small) >= max_patches:
            break

    # Break z-loop if limit reached
    if len(patches_small) >= max_patches:
        break

# Convert list to NumPy array for efficient computation
# dtype=float32 reduces memory usage and matches ML frameworks
patches_small = np.array(patches_small, dtype=np.float32)

# Print resulting dataset shape
# Expected: (10000, 128, 128)
print("Patches shape (10k subset):", patches_small.shape)


# Save patches


# Save patches as .npy file (fast loading later)
np.save(patches_path, patches_small)

# Confirm save location
print("Saved 10k patches to:", patches_path)


# Save example patches as PNG


# Path to store visualization images
figures_path = "/home/hpc/b292dc/b292dc16/vesuvius_project/figures"

# Select 3 example patches:
# - first patch
# - middle patch
# - last patch
# This ensures diversity for inspection
example_indices = [0, len(patches_small)//2, len(patches_small)-1]

# Loop through selected patches
for i in example_indices:

    # Construct output file path
    out_path = os.path.join(figures_path, f"patch_small_{i}.png")

    # Save patch as grayscale image
    plt.imsave(out_path, patches_small[i], cmap="gray")

    # Print confirmation
    print("Saved example patch:", out_path)