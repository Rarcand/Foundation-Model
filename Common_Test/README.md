# GSoC 2026: Multi-Class Classification of Gravitational Lensing Substructures

This repository contains my solution for **Common Test I** of the Google Summer of Code application. The task involves classifying simulated gravitational lensing images into three categories: **No Substructure**, **Subhalo Substructure**, and **Vortex Substructure**.

## Project Strategy & Reasoning

Instead of applying a "black-box" model, my approach was guided by the physical nature of the data:

1. **Architecture (ResNet18):** I chose ResNet18 for its balance of depth and efficiency. Given the specific spatial features of lensing (arcs and rings), the skip-connections in ResNet help preserve fine-grained structural details that distinguish a smooth lens from one with subtle substructure.
2. **Transfer Learning:** Although the data is astrophysical (single-channel intensity maps), I leveraged pre-trained ImageNet weights. Low-level filters in these models are excellent at detecting edge curvature and gradients—exactly what is needed to identify gravitational arcs.
3. **Domain-Specific Augmentation:** In astrophysics, there is no "up" or "down" in a telescope's field of view. I applied 180° rotations and flips to ensure the model learned **rotational invariance**, effectively quadrupling the dataset's diversity without introducing "fake" physics.
4. **Data Handling:** Since the dataset consists of `.npy` files, I developed a custom PyTorch `Dataset` class to handle raw NumPy intensity maps, ensuring zero data loss during the conversion from simulation to tensor.

## Results Summary
- **Macro-Average AUC:** 0.9794
- **Validation Accuracy:** ~89.7%
- **Key Observation:** The model is most confident in identifying "No Substructure." The primary confusion occurs between faint "Spheres" and "No Substructure," which is physically consistent with the difficulty of detecting low-mass subhalos.

## Installation & Usage
1. Clone the repo: `git clone https://github.com/your-username/your-repo-name`
2. Download and place [`dataset.zip`](https://drive.google.com/file/d/1ZEyNMEO43u3qhJAwJeBZxFBEYc_pVYZQ/view) in the root directory.
3. Run the Jupyter Notebook: `jupyter notebook classification_task.ipynb`
