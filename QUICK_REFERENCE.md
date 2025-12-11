# Segmentation Models PyTorch - Quick Reference Guide

## Project At A Glance

**What**: PyTorch library for image semantic segmentation  
**Purpose**: Build and train deep learning models for pixel-level image classification  
**Key Feature**: Simple API + 12 architectures + 800+ pretrained encoders

## One-Minute Overview

### What This Library Does

```python
import segmentation_models_pytorch as smp

# Create a segmentation model in 2 lines
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", 
                 in_channels=3, classes=1)

# Use it like any PyTorch model
prediction = model(image_tensor)  # Output: segmentation mask
```

### Common Use Cases

- 🏥 **Medical Imaging**: Segment organs, tumors in CT/MRI scans
- 🚗 **Autonomous Driving**: Road scene segmentation (cars, pedestrians, roads)
- 🛰️ **Remote Sensing**: Building/road/vegetation detection in satellite images
- 🏭 **Industrial Inspection**: Product defect detection
- 🎨 **Background Removal**: Image matting and background replacement
- 📊 **Any Pixel-Level Classification**: If you need to classify every pixel in an image

## Directory Structure - Quick Map

```
segmentation_models.pytorch/
│
├── segmentation_models_pytorch/     ⭐ Core library code
│   ├── encoders/                    📦 Feature extractors (ResNet, EfficientNet, etc.)
│   ├── decoders/                    🏗️ Model architectures (Unet, DeepLab, etc.)
│   ├── losses/                      📉 Loss functions (Dice, Focal, etc.)
│   ├── metrics/                     📊 Evaluation metrics (IoU, F-score)
│   └── base/                        🔧 Base classes and utilities
│
├── examples/                        📚 Jupyter notebooks with tutorials
├── tests/                           🧪 Unit tests
├── docs/                            📖 Documentation source
└── README.md                        ℹ️ Main project documentation
```

## Available Models (Architectures)

| Model | Best For | Speed |
|-------|----------|-------|
| **Unet** | General purpose, medical imaging | Fast |
| **Unet++** | High accuracy, fine details | Medium |
| **FPN** | Multi-scale objects | Fast |
| **PSPNet** | Scene parsing | Medium |
| **DeepLabV3/V3+** | Large receptive field needed | Medium |
| **Linknet** | Real-time applications | Very Fast |
| **Segformer** | State-of-the-art accuracy | Medium |
| **DPT** | Vision Transformer based | Slow |

## Available Encoders (Backbones)

### CNN-based
- ResNet (18, 34, 50, 101, 152)
- EfficientNet (B0-B7)
- MobileNet (V2, V3) - For edge devices
- DenseNet (121, 169, 201)
- VGG (11, 13, 16, 19)
- And many more...

### Transformer-based
- Vision Transformer (ViT)
- Mix Transformer (MiT) - Used in Segformer
- 800+ models via timm library

## Key Code Patterns

### 1. Basic Training Setup

```python
import segmentation_models_pytorch as smp

# Model
model = smp.Unet('resnet34', encoder_weights='imagenet', classes=1)

# Loss
loss = smp.losses.DiceLoss(mode='binary')

# Training
for images, masks in dataloader:
    preds = model(images)
    loss_value = loss(preds, masks)
    # ... backward and optimize
```

### 2. Multi-class Segmentation

```python
model = smp.Unet(
    encoder_name="efficientnet-b0",
    encoder_weights="imagenet",
    classes=10,  # 10 classes to segment
    activation='softmax'
)
```

### 3. Custom Input Channels

```python
# Grayscale images
model = smp.FPN('resnet34', in_channels=1)

# Multispectral satellite images (10 bands)
model = smp.Unet('resnet34', in_channels=10)
```

### 4. Production Deployment

```python
# Export to ONNX
torch.onnx.export(model, dummy_input, "model.onnx")

# Load from HuggingFace Hub
model = smp.from_pretrained("username/model-name")
```

## Loss Functions Quick Guide

| Loss | Use When | Mode |
|------|----------|------|
| **DiceLoss** | General segmentation, small objects | binary, multiclass |
| **JaccardLoss** | IoU-based optimization | binary, multiclass |
| **FocalLoss** | Class imbalance | binary, multiclass |
| **TverskyLoss** | Control false positives/negatives | binary, multiclass |
| **LovaszLoss** | Optimize IoU directly | binary, multiclass |

## File Organization Explained

### Core Library (`segmentation_models_pytorch/`)

- **encoders/**: Pretrained feature extractors
  - Individual files for each backbone family (resnet.py, efficientnet.py, etc.)
  - `timm_*.py`: Integration with timm library
  
- **decoders/**: Segmentation architectures
  - Each architecture in its own directory (unet/, fpn/, etc.)
  - Contains model definition and decoder implementation
  
- **losses/**: Segmentation-specific loss functions
  - Each loss in separate file (dice.py, focal.py, etc.)
  
- **metrics/**: Evaluation metrics (IoU, F-score)

- **base/**: Foundation classes
  - Base model class
  - HuggingFace Hub integration

### Examples (`examples/`)

- Jupyter notebooks demonstrating:
  - Binary segmentation
  - Multi-class segmentation  
  - Model inference
  - ONNX export
  - HuggingFace Hub usage

### Tests (`tests/`)

- Unit tests for all components
- Model architecture tests
- Encoder tests
- Loss function tests

### Documentation (`docs/`)

- Source files for ReadTheDocs
- API documentation
- Model descriptions

### Auxiliary Files

- **scripts/**: Model conversion utilities
- **misc/**: Development tools (table generation, etc.)
- **requirements/**: Dependency specifications
- **HALLOFFAME.md**: Competition wins using this library

## Installation & Setup

```bash
# Install from PyPI (recommended)
pip install segmentation-models-pytorch

# Or install from GitHub (latest)
pip install git+https://github.com/qubvel/segmentation_models.pytorch

# For development
git clone https://github.com/qubvel/segmentation_models.pytorch
cd segmentation_models.pytorch
make install_dev
```

## Development Commands

```bash
make test          # Run tests
make fixup         # Format and lint code
make table         # Generate encoder tables
```

## Resources

- 📖 **Documentation**: https://smp.readthedocs.io/
- 💻 **GitHub**: https://github.com/qubvel/segmentation_models.pytorch
- 📦 **PyPI**: https://pypi.org/project/segmentation-models-pytorch/
- 🏆 **Competition Wins**: See HALLOFFAME.md

## Requirements

- Python 3.9+
- PyTorch 1.8+
- torchvision 0.9+
- timm 0.9+
- Other: numpy, pillow, tqdm, huggingface-hub

## License

MIT License (see LICENSE file for details)

---

**Pro Tip**: Start with the examples in `examples/` directory. The Jupyter notebooks provide hands-on tutorials for common tasks!
