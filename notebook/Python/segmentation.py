
# Cell 1: Load extracted patches


import numpy as np

# Load preprocessed patches generated in the previous notebook.
# These patches represent small 128x128 regions of CT slices.
# They are the fundamental input units for all downstream learning tasks.
patches = np.load("/home/hpc/b292dc/b292dc16/vesuvius_project/data/patches.npy")

# Print shape to verify:
# Expected format: (num_patches, height, width)
print("Patches shape:", patches.shape)



# Cell 4: Reduce dataset size


# Select a subset of patches for experimentation.
# Reason:
# - Full dataset is very large
# - KMeans is computationally expensive (O(n * k * iterations))
# - This allows fast prototyping and debugging
patches_small = patches[:10000]

print("Training subset:", patches_small.shape)



# Cell 5: Normalize patches


# Convert to float for numerical stability
patches_small = patches_small.astype("float32")

# Normalize intensities to [0, 1]
# Important because clustering is distance-based (Euclidean distance),
# and unnormalized data would bias clustering toward high-intensity regions.
patches_small = patches_small / patches_small.max()



# Cell 6: Flatten patches for clustering


# Reshape each 2D patch into a 1D feature vector
# Example:
# 128x128 → 16384 features
# Required because KMeans operates on vector space, not images
X = patches_small.reshape(patches_small.shape[0], -1)

print("Feature shape:", X.shape)



# Cell 8: Apply KMeans clustering


from sklearn.cluster import KMeans

# Define clustering model
# n_clusters=2 means:
# we force the model to split patches into two groups
# Hypothesis:
# - cluster 0: one type of texture (e.g., background fibers)
# - cluster 1: another structure (possibly denser regions)
kmeans = KMeans(n_clusters=2, random_state=0)

# Fit model and assign cluster labels
labels = kmeans.fit_predict(X)



# Cell 9: Verify sklearn installation


from sklearn.cluster import KMeans

# Simple check to confirm environment setup
print("sklearn installed successfully")



# Cell 10: Re-run clustering (clean execution)


from sklearn.cluster import KMeans

# Reload subset (defensive programming)
patches_small = patches[:10000]

# Flatten again
X = patches_small.reshape(patches_small.shape[0], -1)

# Recreate clustering model
kmeans = KMeans(n_clusters=2, random_state=0)

# Fit and predict
labels = kmeans.fit_predict(X)

print("Clustering finished")



# Cell 11: Visualize clustering results


import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

# Display first 6 patches with assigned cluster labels
for i in range(6):
    plt.subplot(2,3,i+1)
    
    # Show patch
    plt.imshow(patches_small[i], cmap="gray")
    
    # Display cluster assignment
    plt.title(f"Cluster {labels[i]}")
    
    plt.axis("off")

plt.tight_layout()
plt.show()

# Purpose:
# - Qualitative evaluation of clustering
# - Check if clusters correspond to meaningful texture differences



# Cell 12: Save visualization


# Save figure for reporting and paper inclusion
plt.savefig(
    "/home/hpc/b292dc/b292dc16/vesuvius_project/figures/clustering_result.png",
    dpi=300
)