# FCN-DeepLabV3+ Semantic Segmentation

A comparative study of semantic segmentation models using FCN (Fully Convolutional Networks) with ResNet50 backbone and DeepLabV3+ architecture.

## Overview

This project implements and compares two popular semantic segmentation architectures:
- **FCN-ResNet50**: Fully Convolutional Networks with ResNet50 backbone
- **DeepLabV3+**: Advanced atrous convolution-based architecture with encoder-decoder structure

## Project Structure

```
.
├── README.md
├── .gitignore
└── fcn-deeplabv3-segmentation.ipynb    # Main segmentation notebook with model implementations
```

## Getting Started

### Requirements

- Python 3.8+
- PyTorch
- torchvision
- NumPy
- Matplotlib
- Jupyter Notebook

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fcn-deeplabv3
```

2. Install dependencies:
```bash
pip install torch torchvision numpy matplotlib jupyter
```

3. Run the Jupyter notebook:
```bash
jupyter notebook fcn-deeplabv3-segmentation.ipynb
```

## Models

### FCN-ResNet50
- Fully convolutional architecture based on VGG/ResNet backbones
- Skip connections for preserving spatial information
- Efficient upsampling using bilinear interpolation

### DeepLabV3+
- Atrous Spatial Pyramid Pooling (ASPP) module
- Encoder-decoder structure for multi-scale feature fusion
- State-of-the-art performance on semantic segmentation benchmarks

## Usage

Open the Jupyter notebook to:
- Load and preprocess datasets
- Train segmentation models
- Evaluate performance metrics
- Visualize segmentation results

## References

- Long et al., "Fully Convolutional Networks for Semantic Segmentation" (FCN)
- Chen et al., "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation" (DeepLabV3+)
