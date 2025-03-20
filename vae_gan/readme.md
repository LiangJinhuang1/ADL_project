# TensorFlow to PyTorch: Uncertainty-Aware Deep Classifiers using Generative Models

## Background

This repository contains an implementation of Evidential Deep Learning, Variational Autoencoders (VAEs), and Generative Adversarial Networks (GANs) in TensorFlow that have been converted to PyTorch. The models are designed to automatically generate out-of-distribution (OOD) exemplars for training, enhancing uncertainty-aware classification.

The original code and concepts can be found at: [Original Repository](https://muratsensoy.github.io/gen.html).

## Features
- **Uncertainty-Aware Classification**: Uses evidential deep learning to estimate uncertainty in predictions.
- **Generative Models for OOD Data**: Incorporates VAEs and GANs to generate synthetic out-of-distribution samples.
- **PyTorch Implementation**: Converted from TensorFlow to PyTorch for better flexibility and performance.
- **Support for MNIST and NotMNIST Datasets**: Includes dataset handling for in-distribution and out-of-distribution scenarios.
- **Training and Evaluation**: Implements training loops for both generative and classification models, along with uncertainty evaluation metrics.


## Usage

### Training the Model

```bash
python test_gan_0603.py --batch_size 64 --vae_epochs 50 --class_epochs 50 --latent_dim 100 --K 10
```

### Key Arguments
- `--batch_size`: Batch size for training (default: 64)
- `--vae_epochs`: Number of epochs for training the VAE-GAN component (default: 50)
- `--class_epochs`: Number of epochs for training the uncertainty-aware classifier (default: 50)
- `--latent_dim`: Latent dimension size (default: 100)
- `--K`: Number of output classes (default: 10)

## Model Overview

The implemented model consists of:
- **Encoder**: Extracts latent representations from input images.
- **Decoder**: Reconstructs images from latent space.
- **Generator**: Generates synthetic OOD samples.
- **Discriminator**: Differentiates between real and fake images.
- **Latent Discriminator**: Classifies latent representations.
- **Uncertainty-Aware Classifier**: LeNet-based classifier that incorporates evidential deep learning.

