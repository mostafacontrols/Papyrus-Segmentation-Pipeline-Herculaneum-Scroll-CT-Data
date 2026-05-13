
# In[1]:
dataset_path = "/home/atuin/b292dc/b292dc10/Herculaneum/Scroll1"

# In[2]:
import os

os.listdir(dataset_path)

# In[3]:
volpkg = dataset_path + "/PHercParis4.volpkg"
os.listdir(volpkg)

# In[4]:
volume_path = volpkg + "/volumes"
files = sorted(os.listdir(volume_path))

print(len(files))
print(files[:10])

# In[5]:
#

# In[6]:
volumes_zarr = "/home/atuin/b292dc/b292dc10/Herculaneum/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized"

# In[7]:
import os

zarr_folder = "/home/atuin/b292dc/b292dc10/Herculaneum/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized"
contents = os.listdir(zarr_folder)
print(contents)

# In[8]:
#

# In[9]:
import os

volumes_path = "/home/atuin/b292dc/b292dc10/Herculaneum/Scroll1/PHercParis4.volpkg/volumes"
files = sorted(os.listdir(volumes_path))
print(files[:10])

# In[10]:
#


# In[11]:
cd /home/atuin/b292dc/b292dc10/Herculaneum/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized

# In[12]:

ls -R

# In[13]:
#

# In[14]:
import os

base_path = "/home/atuin/b292dc/b292dc10/Herculaneum/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized"
print("Contents of volumes_zarr_standardized folder:")
print(os.listdir(base_path))

# In[15]:
volume_path = volpkg + "/volumes"
files = sorted(os.listdir(volume_path))

print(len(files))
print(files[:10])

# In[16]:
print(files[:10])
print(volume_path)

img_path = volume_path + "/" + files[0]
print(img_path)

# In[17]:
import os
import cv2

print("First files:", files[:10])

img_path = volume_path + "/" + files[0]
print("Image path:", img_path)

print("File exists:", os.path.exists(img_path))

img = cv2.imread(img_path, 0)

print("Loaded image object:", img)
print("Image type:", type(img))

# In[18]:
import os

print(os.listdir(volume_path))

# In[19]:
volume_path = volume_path + "/20230205180739"

# In[20]:
files = sorted(os.listdir(volume_path))
print(files[:10])

# In[21]:
import cv2
import matplotlib.pyplot as plt

img_path = os.path.join(volume_path, files[0])

img = cv2.imread(img_path, 0)

plt.imshow(img, cmap="gray")
plt.title("CT Slice")
plt.axis("off")
plt.show()

# In[22]:
import numpy as np
import os
import imageio

files = sorted(os.listdir(volume_path))[:30]  # first 50 slices

volume = []

for f in files:
    img = imageio.v2.imread(os.path.join(volume_path, f))
    volume.append(img)

volume = np.stack(volume)

print("Volume shape:", volume.shape)

# In[23]:
volume = volume.astype("float32")
volume = (volume - volume.min()) / (volume.max() - volume.min())

# In[24]:
import matplotlib.pyplot as plt

plt.imshow(volume[25], cmap="gray")
plt.title("Middle slice")
plt.axis("off")

# In[25]:
patch_size = 128
patches = []

for z in range(volume.shape[0]):
    img = volume[z]

    for y in range(0, img.shape[0]-patch_size, patch_size):
        for x in range(0, img.shape[1]-patch_size, patch_size):

            patch = img[y:y+patch_size, x:x+patch_size]
            patches.append(patch)

patches = np.array(patches)

print("Patches shape:", patches.shape)

# In[26]:
import numpy as np

np.save("/home/hpc/b292dc/b292dc16/vesuvius_project/data/patches.npy", patches) 


# In[27]:
import os
os.getcwd()

# In[28]:


# In[29]:


# In[30]:


# In[31]:


# In[32]:


# In[33]:


# In[34]:


# In[35]:


# In[36]:


# In[37]:


# In[38]:


# In[39]:


# In[40]:
