Reverse Engineering Papyrus Segmentation Pipeline
Herculaneum Scroll CT Data
By: Mostafa Shehata
Overview
This project implements a full processing pipeline for CT scan data of Herculaneum papyrus.
The workflow includes dataset exploration, volume visualization, clustering analysis, and autoencoderbased
reconstruction using Python scripts.
The system is designed for lightweight execution and patch-based learning from high-resolution
volumetric CT data.
Pipeline
CT Volume Normalization Patch Extraction Clustering
Autoencoder Training Reconstruction Evaluation
Project Structure
project /
dataset_exploration .py
volume_visualization .py
segmentation_pipeline .py
experiment_pipeline .py
data /
figures /
clustering_result . png
reconstruction_ *. png
training_curve . png
requirements .txt
Installation
1. Create Virtual Environment
python -m venv venv
source venv / bin/ activate % Linux / Mac
venv \ Scripts \ activate % Windows
1
2. Install Dependencies
pip install -r requirements .txt
Dataset Setup
Download and extract the Herculaneum CT dataset and update file paths inside the scripts:
dataset_path = " path /to/ Scroll1 "
volume_folder = " path /to/ volumes /"
data_path = " path /to/ patches .npy"
Important: All paths are hardcoded and must be adjusted before execution.
Execution Order
Run the Python scripts in the following sequence:
1. Dataset Exploration
python dataset_exploration .py
Output:
• patches.npy
2. Volume Visualization and Patch Extraction
python volume_visualization .py
Output:
• patchessmall.npy
• slice images
3. Segmentation Pipeline (Clustering)
python segmentation_pipeline .py
Output:
• clustering result.png
4. Autoencoder Experiment
python experiment_pipeline .py
Output:
• reconstruction images
• training curve.png
2
Model Summary
• Model: Convolutional Autoencoder
• Input: 128×128 grayscale patches
• Parameters: ∼4800
• Loss Function: Mean Squared Error (MSE)
• Optimizer: Adam
• Training: 3 epochs, batch size 8
Key Design Decisions
• Patch-based processing for memory efficiency
• Lightweight architecture for HPC constraints
• Unsupervised learning (no labels required)
• K-Means clustering for data augmentation
Limitations
• No ink or text detection module
• Limited dataset subset for training
• Loss of fine details due to compression
Future Work
• Integrate ink detection models
• Use perceptual or contrastive loss
• Train on full dataset
Conclusion
This project demonstrates that lightweight convolutional autoencoders can effectively learn
structural representations from CT scans of papyrus, enabling meaningful reconstruction and
analysis under computational constraints.
