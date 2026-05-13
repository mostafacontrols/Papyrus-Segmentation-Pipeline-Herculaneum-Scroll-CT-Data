
# Cell 1: Load dataset & prepare for training


import os                     # file system operations
import numpy as np           # numerical computations
import matplotlib.pyplot as plt  # visualization

# Path to saved patch dataset
data_path = "/home/hpc/b292dc/b292dc16/vesuvius_project/data/patches.npy"

# Load full patch dataset (previously extracted from CT volume)
patches = np.load(data_path)

# Print dataset shape (e.g., ~192K patches)
print("Full dataset:", patches.shape)


# Reduce dataset size for lightweight experimentation


# Select only first 2000 patches
# Reason:
# - faster training
# - fits GPU memory constraints
patches = patches[:2000]

# Convert to float32 for compatibility with TensorFlow
patches = patches.astype("float32") 

# Add channel dimension → required for Conv2D
# Shape becomes: (N, 128, 128, 1)
patches = np.expand_dims(patches, -1)

print("Experiment dataset:", patches.shape)


# Cell 2: Train / Validation / Test split


from sklearn.model_selection import train_test_split

# Split into training (80%) and test (20%)
train_p, test_p = train_test_split(
    patches, test_size=0.2, random_state=42
)

# Split training further into training (64%) and validation (16%)
train_p, val_p = train_test_split(
    train_p, test_size=0.2, random_state=42
)

# Print dataset sizes
print("Train:", train_p.shape)
print("Validation:", val_p.shape)
print("Test:", test_p.shape)



# Cell 3: Model Definition (Autoencoder)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

# Define convolutional autoencoder
model = Sequential([

    # -------- Encoder --------
    # First convolution layer
    # Extracts low-level features (edges, textures)
    Conv2D(8, 3, activation="relu", padding="same",
           input_shape=(128,128,1)),

    # Downsample spatial resolution by factor of 2
    MaxPooling2D(2, padding="same"),

    # Deeper feature extraction
    Conv2D(16, 3, activation="relu", padding="same"),

    # Further compression
    MaxPooling2D(2, padding="same"),

    # -------- Bottleneck --------
    # Compact latent representation
    Conv2D(16, 3, activation="relu", padding="same"),

    # -------- Decoder --------
    # Upsample back to higher resolution
    UpSampling2D(2),

    # Refine reconstructed features
    Conv2D(8, 3, activation="relu", padding="same"),

    # Restore original spatial size
    UpSampling2D(2),

    # Final reconstruction layer
    # Sigmoid ensures output in [0,1]
    Conv2D(1, 3, activation="sigmoid", padding="same")
])

# Compile model
# Adam → efficient optimizer
# MSE → reconstruction loss
model.compile(optimizer="adam", loss="mse")

# Print architecture summary
model.summary()



# Cell 4: Model Training


# Train autoencoder
history = model.fit(

    # Input = output (autoencoder)
    train_p,
    train_p,

    # Validation data
    validation_data=(val_p, val_p),

    # Small number of epochs for quick experimentation
    epochs=3,

    # Small batch size for memory efficiency
    batch_size=8
)



# Cell 5: Reconstruction Visualization


import os
import matplotlib.pyplot as plt

# Path to save figures
figures_path = "/home/hpc/b292dc/b292dc16/vesuvius_project/figures"
os.makedirs(figures_path, exist_ok=True)

# Select a few test samples
samples = test_p[:3]

# Generate reconstructions
preds = model.predict(samples)

# Loop through samples
for i in range(len(samples)):

    # Create side-by-side plot
    fig, ax = plt.subplots(1,2, figsize=(6,3))

    # Original input image
    ax[0].imshow(samples[i].squeeze(), cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("Input")
    ax[0].axis("off")

    # Reconstructed output
    ax[1].imshow(preds[i].squeeze(), cmap="gray", vmin=0, vmax=1)
    ax[1].set_title("Reconstruction")
    ax[1].axis("off")

    plt.tight_layout()

    # Save figure
    path = os.path.join(figures_path, f"reconstruction_{i}.png")
    plt.savefig(path)

    plt.show()

    print("Saved:", path)



# Cell 6: Quantitative Evaluation (MSE)


# Predict on full test set
preds = model.predict(test_p)

# Compute Mean Squared Error
mse = np.mean((test_p - preds) ** 2)

print("Test Reconstruction MSE:", mse)



# Cell 7: Training Curve Visualization


# Plot training and validation loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

# Labels and title
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

# Legend
plt.legend(["Train","Validation"])

# Save plot
plt.savefig(figures_path + "/training_curve.png")