
# In[1]:
# ==============================
# Cell 1: Volume loading & visualization
# ==============================
import os
import numpy as np
import imageio
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
dataset_path = "/home/hpc/b292dc/b292dc16/vesuvius_project/data"  # change if needed
volume_folder = "/home/atuin/b292dc/b292dc10/Herculaneum/Scroll1/PHercParis4.volpkg/volumes/20230205180739"

# -----------------------------
# Load first 50 slices
# -----------------------------
files = sorted(os.listdir(volume_folder))[:50]
volume = [imageio.v2.imread(os.path.join(volume_folder, f)) for f in files]
volume = np.stack(volume).astype(np.float32)

# Normalize
volume = (volume - volume.min()) / (volume.max() - volume.min())
print("Volume shape:", volume.shape)

# -----------------------------
# Visualize first and middle slices
# -----------------------------
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(volume[0], cmap="gray")
plt.title("First slice")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(volume[volume.shape[0]//2], cmap="gray")
plt.title("Middle slice")
plt.axis("off")
plt.show()

# -----------------------------
# Save example slices for sanity check
# -----------------------------
figures_path = "/home/hpc/b292dc/b292dc16/vesuvius_project/figures"
os.makedirs(figures_path, exist_ok=True)
plt.imsave(os.path.join(figures_path, "slice_first.png"), volume[0], cmap="gray")
plt.imsave(os.path.join(figures_path, "slice_middle.png"), volume[volume.shape[0]//2], cmap="gray")
print("Saved example CT slices in figures folder")

# In[2]:
# ==============================
# Cell 2: Patch extraction (10k) & saving
# ==============================
import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
patch_size = 128
max_patches = 10000
patches_path = "/home/hpc/b292dc/b292dc16/vesuvius_project/data/patches_small.npy"
os.makedirs(os.path.dirname(patches_path), exist_ok=True)

# -----------------------------
# Extract patches
# -----------------------------
patches_small = []
for z in range(volume.shape[0]):
    img = volume[z]
    for y in range(0, img.shape[0] - patch_size, patch_size):
        for x in range(0, img.shape[1] - patch_size, patch_size):
            patch = img[y:y+patch_size, x:x+patch_size]
            patches_small.append(patch)
            if len(patches_small) >= max_patches:
                break
        if len(patches_small) >= max_patches:
            break
    if len(patches_small) >= max_patches:
        break

patches_small = np.array(patches_small, dtype=np.float32)
print("Patches shape (10k subset):", patches_small.shape)

# -----------------------------
# Save patches
# -----------------------------
np.save(patches_path, patches_small)
print("Saved 10k patches to:", patches_path)

# -----------------------------
# Save example patches as PNG
# -----------------------------
figures_path = "/home/hpc/b292dc/b292dc16/vesuvius_project/figures"
example_indices = [0, len(patches_small)//2, len(patches_small)-1]
for i in example_indices:
    out_path = os.path.join(figures_path, f"patch_small_{i}.png")
    plt.imsave(out_path, patches_small[i], cmap="gray")
    print("Saved example patch:", out_path)

# In[3]:


# In[4]:


# In[5]:


# In[6]:


# In[7]:
