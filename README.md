# Test-Time Adaptation with MEMO
Deep Learning, 2024 - UniTn\
Authors: [**Giovanni Valer**](https://github.com/jo-valer), [**Emanuele Poiana**](https://github.com/IlPoiana)

## Overview
This project focuses on implementing a Test-Time Adaptation technique for image classification. The goal is to improve the performance of a pre-trained model on out-of-distribution data, without any knowledge of the test-time data distribution. The method we implement is based on **MEMO** (Marginal Entropy Minimization with One test point). The model we use is ResNet-50, pre-trained on **ImageNet**; the dataset we use for testing is **ImageNet-A** (consisting of images that are misclassified by ResNet).

### MEMO, [Zhang et al. (2021)](https://arxiv.org/abs/2110.09506)
MEMO consists in applying a set of augmentations to the test image, collecting the output probability of the pre-trained model for each augmented image, and then undertaking a gradient-based optimization to minimize the entropy of the output distributions. The idea is to fine-tune all the model's parameters (on a single test image) to produce consistent predictions across different augmentations.

### Our Contributions
We further explore other techniques to improve the performance, either modifying MEMO or undertaking different approaches:
- Entropy Loss variants (**Weighted Entropy** Loss, **Cut Entropy** Loss)
- Final **Prediction on Marginal Distribution** (either standard or weighted average)
- **Batch Normalization**
- Different **composition of augmentations**

## Full report
The report (a self-contained notebook with code and text) can be read [here](https://github.com/jo-valer/tta-memo/blob/main/report.ipynb).
