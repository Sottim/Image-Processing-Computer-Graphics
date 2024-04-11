## Image Super-resolution using CNN
The project aims to implement Image Super-Resolution using Convolutional Neural Networks. ISR is the process of enhancing the resolution of an image, turning a low-resolution (LR) image into a high-resolution (HR) one. CNNs have shown remarkable performance in image super-resolution. By leveraging CNN architecture, we can train models to learn the mapping between LR and HR images, effectively reconstructing details lost during down-sampling.

### Motivation
The motivation behind this project is to improve the quality and clarity of retinal images as they are crucial for accurate medical diagnosis. By leveraging the strengths of CNNs, the model aims to effectively solve the image super-resolution task, particularly on retinal images, providing higher-quality and more detailed images for better medical assessment.

### Methodology

##### 1. Data Preparation
The dataset used in this project is
    [STARE](https://www.kaggle.com/datasets/vidheeshnacode/stare-dataset)  (STructured Analysis of the Retina) Dataset from Kaggle:
    
**https://www.kaggle.com/datasets/vidheeshnacode/stare-dataset**

Pre-processing step will involve taking the images, including resizing LR images, normalization, and augmentation techniques to enhance the diversity of the dataset.

##### 2. Model Design
The model architecture is based on the the paper:

[Image Super-Resolution Using Deep
Convolutional Networks](https://arxiv.org/pdf/1501.00092.pdf)

* Authors: Zongsheng Yue, Jianyi Wang, Chen Change Loy

### Usage
    Input: High Resolution(HR) and Low Resolution(LR) Retinal Image Pair
    
    Output: Given the Low Resolution Input image, the model reconstructs the high resolution

### Requirements
* Python 3.10
* Pytorch 2.1.2
* xformers 0.0.23

### Acknowledgement
The dataset: [STARE](https://www.kaggle.com/datasets/vidheeshnacode/stare-dataset)  (STructured Analysis of the Retina) Dataset from Kaggle **https://www.kaggle.com/datasets/vidheeshnacode/stare-dataset**

Paper: [Image Super-Resolution Using Deep
Convolutional Networks](https://arxiv.org/pdf/1501.00092.pdf)

* Authors: Zongsheng Yue, Jianyi Wang, Chen Change Loy

### Contributors
* Santosh Adhikari


