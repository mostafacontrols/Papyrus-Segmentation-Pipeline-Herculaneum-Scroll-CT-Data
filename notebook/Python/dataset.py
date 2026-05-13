
# Cell 1: Define dataset root


# Absolute path to the Herculaneum Scroll dataset on the HPC filesystem.
# This acts as the entry point to all subsequent data exploration.
# Using absolute paths ensures reproducibility across notebook runs in HPC environments.
dataset_path = "/home/atuin/b292dc/b292dc10/Herculaneum/Scroll1"



# Cell 2: Explore dataset root


import os

# List contents of dataset directory.
# Purpose: verify dataset availability and structure before deeper processing.
# This is a sanity check step to avoid silent path errors later.
os.listdir(dataset_path)



# Cell 3: Navigate into .volpkg structure


# The dataset is packaged in a ".volpkg" structure.
# This is a standardized container used in the Vesuvius Challenge
# to organize volumetric CT data, metadata, and derived outputs.
volpkg = dataset_path + "/PHercParis4.volpkg"

# Inspect internal structure of the package.
# Helps identify where raw volumes, processed data, and metadata are stored.
os.listdir(volpkg)



# Cell 4: Locate raw CT volumes


# "volumes" directory contains raw CT slices organized by scan ID.
volume_path = volpkg + "/volumes"

# List all available volume folders.
# Typically contains scan identifiers (timestamps).
files = sorted(os.listdir(volume_path))

# Print diagnostics:
# - number of entries
# - first few entries to understand naming convention
print(len(files))
print(files[:10])


# Cell 6: Locate Zarr-based volumes


# Alternative representation of the dataset using Zarr format.
# Zarr enables chunked, compressed, and efficient access to large volumetric data.
# Useful for large-scale ML pipelines where full volumes cannot fit in memory.
volumes_zarr = "/home/atuin/b292dc/b292dc10/Herculaneum/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized"



# Cell 7: Inspect Zarr folder


import os

zarr_folder = volumes_zarr

# Check contents.
# Expect compressed archives or chunked datasets.
# Important for deciding whether to use TIFF slices or Zarr pipeline.
contents = os.listdir(zarr_folder)
print(contents)



# Cell 9: Re-check volumes directory


import os

# Reconfirm available volume directories.
# This redundancy ensures no state corruption occurred.
volumes_path = "/home/atuin/b292dc/b292dc10/Herculaneum/Scroll1/PHercParis4.volpkg/volumes"
files = sorted(os.listdir(volumes_path))
print(files[:10])



# Cell 11-12: Inspect filesystem manually


# Change working directory to Zarr folder.
# This enables shell-level inspection.
cd /home/atuin/b292dc/b292dc10/Herculaneum/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized

# Recursive listing to inspect structure depth.
# Useful for understanding dataset hierarchy.
ls -R



# Cell 14: Confirm Zarr contents


import os

base_path = volumes_zarr

print("Contents of volumes_zarr_standardized folder:")
print(os.listdir(base_path))



# Cell 15-16: Identify actual CT volume folder


volume_path = volpkg + "/volumes"
files = sorted(os.listdir(volume_path))

print(len(files))
print(files[:10])

# Important observation:
# ".vckeep" is not an image, but a placeholder file.
# Actual data resides inside subfolder (timestamp-based).

print(files[:10])
print(volume_path)

# Construct path to first entry
img_path = volume_path + "/" + files[0]
print(img_path)



# Cell 17: Attempt image loading (diagnostic)


import os
import cv2

print("First files:", files[:10])

img_path = volume_path + "/" + files[0]
print("Image path:", img_path)

# Check file existence
print("File exists:", os.path.exists(img_path))

# Attempt to read file as image
# Expected failure: ".vckeep" is not a valid image
img = cv2.imread(img_path, 0)

print("Loaded image object:", img)
print("Image type:", type(img))

# Insight:
# This step confirms that directory navigation must go deeper.



# Cell 18-20: Enter actual scan directory


import os

print(os.listdir(volume_path))

# Navigate into actual CT scan folder
# This contains the real TIFF slices.
volume_path = volume_path + "/20230205180739"

# List actual slice files
files = sorted(os.listdir(volume_path))
print(files[:10])



# Cell 21: Visualize a CT slice


import cv2
import matplotlib.pyplot as plt

# Select first slice
img_path = os.path.join(volume_path, files[0])

# Load grayscale image
img = cv2.imread(img_path, 0)

# Visualization for sanity check
# Confirms correct loading and reveals texture structure.
plt.imshow(img, cmap="gray")
plt.title("CT Slice")
plt.axis("off")
plt.show()



# Cell 22: Load multiple slices into 3D volume


import numpy as np
import imageio

# Load subset of slices (first 30 for memory efficiency)
# Full dataset is too large for initial experimentation.
files = sorted(os.listdir(volume_path))[:30]

volume = []

# Build 3D volume by stacking 2D slices
for f in files:
    img = imageio.v2.imread(os.path.join(volume_path, f))
    volume.append(img)

# Convert list into 3D NumPy array
volume = np.stack(volume)

# Shape interpretation: (depth, height, width)
print("Volume shape:", volume.shape)


# Cell 23: Normalize intensity values


# Convert to float for numerical stability
volume = volume.astype("float32")

# Min-max normalization
# Purpose:
# - standardize input range
# - improve neural network convergence
volume = (volume - volume.min()) / (volume.max() - volume.min())



# Cell 24: Visualize middle slice


import matplotlib.pyplot as plt

# Display middle slice
# Helps verify volume integrity after preprocessing
plt.imshow(volume[25], cmap="gray")
plt.title("Middle slice")
plt.axis("off")


# ==============================
# Cell 25: Patch extraction
# ==============================

patch_size = 128
patches = []

# Iterate over volume depth
for z in range(volume.shape[0]):
    img = volume[z]

    # Slide window across image
    # Step size = patch size (non-overlapping patches)
    for y in range(0, img.shape[0]-patch_size, patch_size):
        for x in range(0, img.shape[1]-patch_size, patch_size):

            # Extract patch
            patch = img[y:y+patch_size, x:x+patch_size]

            # Store patch
            patches.append(patch)

# Convert to array
patches = np.array(patches)

print("Patches shape:", patches.shape)

# Concept:
# Converts large CT slices into manageable training samples.
# Enables batch-based neural network training.



# Cell 26: Save patches


import numpy as np

# Save preprocessed dataset to disk
# Avoids recomputation and speeds up experimentation
np.save("/home/hpc/b292dc/b292dc16/vesuvius_project/data/patches.npy", patches)



# Cell 27: Check working directory


import os

# Confirm execution environment path
# Useful for debugging file saving/loading issues
os.getcwd()