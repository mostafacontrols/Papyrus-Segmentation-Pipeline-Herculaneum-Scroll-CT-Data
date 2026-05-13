
# In[1]:
import os
import numpy as np
import matplotlib.pyplot as plt

data_path = "/home/hpc/b292dc/b292dc16/vesuvius_project/data/patches.npy"

patches = np.load(data_path)

print("Full dataset:", patches.shape)

# Use only 2000 patches for lightweight experiments
patches = patches[:2000]

patches = patches.astype("float32") 
patches = np.expand_dims(patches, -1)

print("Experiment dataset:", patches.shape)

# In[2]:
from sklearn.model_selection import train_test_split

train_p, test_p = train_test_split(patches, test_size=0.2, random_state=42)
train_p, val_p = train_test_split(train_p, test_size=0.2, random_state=42)

print("Train:", train_p.shape)
print("Validation:", val_p.shape)
print("Test:", test_p.shape)

# In[3]:


# In[4]:
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

model = Sequential([
    Conv2D(8, 3, activation="relu", padding="same", input_shape=(128,128,1)),
    MaxPooling2D(2, padding="same"),

    Conv2D(16, 3, activation="relu", padding="same"),
    MaxPooling2D(2, padding="same"),

    Conv2D(16, 3, activation="relu", padding="same"),
    UpSampling2D(2),

    Conv2D(8, 3, activation="relu", padding="same"),
    UpSampling2D(2),

    Conv2D(1, 3, activation="sigmoid", padding="same")
])

model.compile(optimizer="adam", loss="mse")

model.summary()

# In[5]:
history = model.fit(
    train_p,
    train_p,
    validation_data=(val_p, val_p),
    epochs=3,
    batch_size=8
)

# In[6]:
import os
import matplotlib.pyplot as plt

figures_path = "/home/hpc/b292dc/b292dc16/vesuvius_project/figures"
os.makedirs(figures_path, exist_ok=True)

samples = test_p[:3]
preds = model.predict(samples)

for i in range(len(samples)):

    fig, ax = plt.subplots(1,2, figsize=(6,3))

    ax[0].imshow(samples[i].squeeze(), cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("Input")
    ax[0].axis("off")

    ax[1].imshow(preds[i].squeeze(), cmap="gray", vmin=0, vmax=1)
    ax[1].set_title("Reconstruction")
    ax[1].axis("off")

    plt.tight_layout()

    path = os.path.join(figures_path, f"reconstruction_{i}.png")
    plt.savefig(path)

    plt.show()

    print("Saved:", path)

# In[7]:
preds = model.predict(test_p)

mse = np.mean((test_p - preds) ** 2)

print("Test Reconstruction MSE:", mse)

# In[8]:
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend(["Train","Validation"])

plt.savefig(figures_path + "/training_curve.png")

# In[9]:


# In[10]:
