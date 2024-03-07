# Anonymize-Anyone
Anonymize Anyone: Bridging Ethnic Fairness in Text-to-Face Synthesis using a Diffusion Model

This repository contains the implementation of the following paper:

We propose **Anonymize-Anyone**, where users can use multiple modalities to control face generation and editing.
    *(a) Face Generation*. Given multi-modal controls, our framework synthesizes high-quality images consistent with the input conditions.
    *(b) Face Editing*. Collaborative Diffusion also supports multi-modal editing of real images with promising identity preservation capability.

<br>
<img src="./assets/fig_framework.jpg" width="100%">

We use pre-trained uni-modal diffusion models to perform multi-modal guided face generation and editing. At each step of the reverse process (i.e., from timestep t to t − 1), the **dynamic diffuser** predicts the spatial-varying and temporal-varying **influence function** to *selectively enhance or suppress the contributions of the given modality*.

## :heavy_check_mark: Updates
- [03/2024] ...


## :hammer: Installation

1. Clone repo

   ```
   
   ```

2. Create conda environment.<br>

   ```
   
   ```

3. Install dependencies

   ```
   
   ```


## :purple_heart: Acknowledgement

The codebase is maintained by [Ziqi Huang](https://ziqihuangg.github.io/).

This project is built on top of [LDM](https://github.com/CompVis/latent-diffusion). We trained on data provided by [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans), [CelebA-Dialog](https://github.com/ziqihuangg/CelebA-Dialog), [CelebAMask-HQ](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html), and [MM-CelebA-HQ-Dataset](https://github.com/IIGROUP/MM-CelebA-HQ-Dataset). We also make use of the [Imagic implementation](https://github.com/justinpinkney/stable-diffusion/blob/main/notebooks/imagic.ipynb).
