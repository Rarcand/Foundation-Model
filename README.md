# ML4SCI GSoC 2026: Specific Test IX - Foundation Models for Strong Lensing

This repository contains my implementation for **Specific Test IX (Task IX.A and IX.B)** for the Machine Learning for Science (ML4SCI) Google Summer of Code 2026 application. 

The primary objective of these tasks was to develop a **Foundation Model** approach to strong gravitational lensing analysis. Rather than training models from scratch for every specific task, I utilized a self-supervised Masked Autoencoder (MAE) to learn a robust, physics-aware latent representation of unperturbed Einstein rings. This pre-trained foundation was then successfully fine-tuned for two distinct downstream tasks: Substructure Classification and Super-Resolution.

## Task IX.A: Masked Autoencoder & Classification

### Strategy & Architecture
The goal of this task was to classify simulated lensing images into three categories: `no_sub` (perfect lens), `cdm` (Cold Dark Matter perturbation), and `axion` (Axion-like particle perturbation).

**1. Self-Supervised Pre-Training:**
I implemented a Vision Transformer (ViT) based Masked Autoencoder. The model was trained exclusively on the `no_sub` dataset. By masking 75% of the input image patches and forcing the model to reconstruct the missing pixels, the encoder developed a deep internal representation of standard macro-lensing geometry. 

**2. Data Engineering & Anomaly Handling:**
During the dataset audit, I discovered a significant structural anomaly: while `no_sub` and `cdm` were saved as standard 64x64 float arrays, the `axion` data was saved as inhomogeneous Python object arrays containing both the image matrix and simulation metadata. I engineered a custom, surgical PyTorch `Dataset` class to safely unpack these object arrays on the fly, dynamically cropping and standardizing all inputs to the required 60x60 tensor shape without losing structural integrity.

**3. Fine-Tuning:**
Once the foundation was established, I attached a classification head (Global Average Pooling + Linear Layer) and fine-tuned the model on the full, 3-class dataset using a 90:10 train/validation split.

### Results
Because the foundation model already understood the physics of a "perfect" lens, it was highly sensitive to the subtle perturbations caused by CDM and Axions. 

* **Final Macro AUC:** ~0.986
* **Class AUCs:** * `no_sub`: 0.9952
  * `axion`: 0.9892
  * `cdm`: 0.9739
---

## Task IX.B: Super-Resolution Fine-Tuning

### Strategy & Architecture
For the second task, I leveraged the exact same pre-trained MAE encoder to upscale Low-Resolution (LR) images to High-Resolution (HR) ground truths.

**Surgical Weight Loading:**
The LR inputs were 75x75, translating to a 5x5 patch grid (unlike the 4x4 grid used in pre-training). To handle this, I performed a partial state-dict load, importing the learned Transformer blocks and patch embeddings from Task IX.A while initializing new positional embeddings to accommodate the larger grid. 

**Upsampling Head:**
The classification head was replaced with an upsampling network utilizing `ConvTranspose2d` layers to mathematically expand the 5x5 latent grid into a sharp 150x150 High-Resolution output (a 2x upscale factor).

### Results
The model successfully sharpened fuzzy arcs and resolved distinct light sources without hallucinating non-physical artifacts. The validation loss tracked closely with the training loss, indicating strong generalization rather than pixel-memorization.

* **Peak Signal-to-Noise Ratio (PSNR):** 38.47 dB
* **Structural Similarity Index (SSIM):** > 0.90
* **Mean Squared Error (MSE):** 0.000142


---

## Repository Structure

```text
├── Foundation-Model/
│   ├── Task_IX_Foundational_Model.ipynb  # Pre-training and fine-tuning logic
│   └── Task_VII_Physics_Guided_ML.ipynb    # Upsampling and evaluation logic
├── Common-Test/
│   ├── Common_Test_1.ipynb
└── README.md
