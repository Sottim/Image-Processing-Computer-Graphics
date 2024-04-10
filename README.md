## Finetuning Diffusion Model for Image Super-Resolution on Retinal Images
This project explores the use of a ResShift: diffusion model for the task of image super-resolution by residual shifting, specifically focusing on retinal images. Recent advancements in diffusion models have shown that they can outperform traditional approach that rely on GANs in various image generation and enhancement applications.

### Motivation
The motivation behind this project is to improve the quality and clarity of retinal images as they are crucial for accurate medical diagnosis. By leveraging the strengths of diffusion models and the proposed residual shifting approach, the ResShift model aims to effectively solve the image super-resolution task, particularly on retinal images, providing higher-quality and more detailed images for better medical assessment.

### Methodology

##### 1. Model Design
The core idea of this model is to construct a Markov chain that serves as a bridge between the low-resolution (LR) and high-resolution (HR) image pairs. The model transits from the HR image to the LR image by gradually shifting the residual between the two.

##### 2. Forward Process
The forward process defines the transition distribution to gradually shift the residual between the HR and LR images through a Markov chain of length T. The transition distribution is formulated as:

$q(x_t | x_{t-1}, y_0) = N(x_t; x_{t-1} + α_t * (y_0 - x_0), κ^2 * α_t * I)$

where $α_t$ is the incremental shift in the residual at each timestep, and $κ$ is a hyperparameter controlling the noise variance.

The marginal distribution at any timestep t is analytically integrable, given by:

$q(x_t | x_0, y_0) = N(x_t; x_0 + η_t * (y_0 - x_0), κ^2 * η_t * I)$

##### 3. Reverse Process
The reverse process aims to estimate the posterior distribution $p(x_0 | y_0)$ by learning the inverse transition kernel $p_θ(x_{t-1} | x_t, y_0)$ parameterized by θ. The optimization is achieved by minimizing the negative evidence lower bound.

##### 4. Noise Schedule
The proposed method employs a hyperparameter κ and a shifting sequence ${η_t}_{t=1}^T$ to determine the noise schedule in the diffusion process. The shifting sequence $√η_t$ is designed using a non-uniform geometric schedule to provide flexibility in controlling the shifting speed and the trade-off between fidelity and realism in the SR results.
Additionally, an elaborate noise schedule is developed to flexibly control the shifting speed and the noise strength during the diffusion process. 

### Dataset 

The dataset used in the finetuning process is
    [STARE](https://www.kaggle.com/datasets/vidheeshnacode/stare-dataset)  (STructured Analysis of the Retina) Dataset from Kaggle: **https://www.kaggle.com/datasets/vidheeshnacode/stare-dataset**


### Usage
    Input: High Resolution(HR) and Low Resolution(LR) Retinal Image Pair
    
    Output: Given the Low Resolution Input image, the model reconstructs the high resolution

### Requirements
* Python 3.10
* Pytorch 2.1.2
* xformers 0.0.23

### Acknowledgement
This project is based on the paper : [ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting](https://arxiv.org/abs/2307.12348)

* Authors: Zongsheng Yue, Jianyi Wang, Chen Change Loy

### Contributors
* Santosh Adhikari


