
# In[1]:
import numpy as np

patches = np.load("/home/hpc/b292dc/b292dc16/vesuvius_project/data/patches.npy")

print("Patches shape:", patches.shape)

# In[2]:
#
# In[3]:
#

# In[4]:
patches_small = patches[:10000]

print("Training subset:", patches_small.shape)

# In[5]:
patches_small = patches_small.astype("float32")
patches_small = patches_small / patches_small.max()

# In[6]:
X = patches_small.reshape(patches_small.shape[0], -1)

print("Feature shape:", X.shape)

# In[7]:
#

# In[8]:
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0)
labels = kmeans.fit_predict(X)

# In[9]:
from sklearn.cluster import KMeans
print("sklearn installed successfully")

# In[10]:
from sklearn.cluster import KMeans

patches_small = patches[:10000]

X = patches_small.reshape(patches_small.shape[0], -1)

kmeans = KMeans(n_clusters=2, random_state=0)
labels = kmeans.fit_predict(X)

print("Clustering finished")

# In[11]:
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(patches_small[i], cmap="gray")
    plt.title(f"Cluster {labels[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# In[12]:
plt.savefig("/home/hpc/b292dc/b292dc16/vesuvius_project/figures/clustering_result.png", dpi=300)

# In[13]:
