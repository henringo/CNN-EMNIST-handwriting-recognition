# Handwriting Recognition System

A complete machine learning system for recognizing handwritten letters (A-Z) and digits (0-9) using a Convolutional Neural Network trained on EMNIST and MNIST datasets. Features a local Tkinter drawing application for real-time prediction.

## Features

- **36-Class Recognition**: Letters A-Z and digits 0-9
- **CNN Architecture**: Simple but effective convolutional neural network
- **Local-Only**: No web server or API required - runs entirely on your machine
- **Real-Time Drawing**: Interactive Tkinter canvas for drawing and prediction
- **Top-3 Predictions**: Shows the most likely characters with confidence scores
- **Data Augmentation**: Robust training with rotation, perspective, and contrast variations
- **Cross-Platform**: Works on macOS, Windows, and Linux

## Quick Start

### 1. Environment Setup
```bash


# Create virtual environment (Python 3.13+ recommended)
python3 -m venv .venv-tk
source .venv-tk/bin/activate  # On Windows: .venv-tk\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify Tkinter is available
python -c "import tkinter as tk; print('Tkinter OK - Version', tk.TkVersion)"
```

### 2. Train the Model
```bash
# Basic training (8 epochs, ~5-10 minutes)
python -m ml.train

# Custom training options
python -m ml.train --epochs 10 --batch-size 64 --lr 1e-3

# Training with specific parameters
python -m ml.train --epochs 8 --batch-size 128 --lr 2e-3 --weight-decay 1e-4 --early-stop 3
```

### 3. Run the Drawing App
```bash
# Launch the simple canvas app
python tools/canvas_app_simple.py

# Or use the advanced app with preprocessing
python tools/canvas_app.py
```

## Project Structure

```
handwriting project/
├── ml/                          # Machine Learning Code
│   ├── model.py                # CNN architecture (SimpleCNN)
│   ├── train.py                # Training script with data augmentation
├── tools/                       # Applications
│   ├── canvas_app_simple.py    # Simple drawing app (recommended)
│   ├── canvas_app.py           # Advanced app with preprocessing
│   └── show_all_emnist_characters.py  # Dataset visualization
├── models/                      # Trained Models
│   └── emnist36_balanced.pt    # Best trained model
├── data/                        # Datasets (auto-downloaded)
│   ├── EMNIST/                 # Letters A-Z dataset
│   └── MNIST/                  # Digits 0-9 dataset
├── emnist_*.png                # Character visualization images
├── requirements.txt            # Python dependencies
└── README.md
```

## Model Architecture

**SimpleCNN** - A lightweight but effective architecture:

```
Input: 28×28 grayscale image
├── Conv2d(1→32) + BatchNorm + ReLU
├── Conv2d(32→32) + BatchNorm + ReLU  
├── MaxPool2d(2) → 14×14
├── Conv2d(32→64) + BatchNorm + ReLU
├── Conv2d(64→64) + BatchNorm + ReLU
├── MaxPool2d(2) → 7×7
├── Flatten → 64×7×7 = 3136 features
├── Linear(3136→256) + ReLU + Dropout(0.3)
└── Linear(256→36) → 36 classes (A-Z, 0-9)
```

## Training Details

### Datasets
- **EMNIST Letters**: A-Z (both uppercase and lowercase, case-collapsed)
- **MNIST Digits**: 0-9
- **Total Classes**: 36
- **Training Samples**: ~697,000
- **Image Size**: 28×28 grayscale

### Data Augmentation
- **Random Rotation**: ±10 degrees for orientation robustness
- **Random Affine**: Translation, scaling, rotation
- **Random Perspective**: Distortion for handwriting variation
- **Random Autocontrast**: Contrast adjustment
- **Random Sharpness**: Sharpness variation

### Training Configuration
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 2e-3 with OneCycleLR scheduler
- **Batch Size**: 128
- **Epochs**: 8 (with early stopping)
- **Loss Function**: CrossEntropyLoss with label smoothing
- **Regularization**: Weight decay, gradient clipping, dropout

## Using the Drawing App

1. **Launch**: Run `python tools/canvas_app_simple.py`
2. **Reference**: Check the `emnist_*.png` files to see the exact writing style the model learned
3. **Draw**: Use mouse to draw letters or numbers on the canvas (try to match the training style)
4. **Predict**: Click "Predict" to see top-3 predictions
5. **Clear**: Click "Clear" to start over
6. **Visualize**: Use `python tools/show_all_emnist_characters.py` to generate fresh training examples

## Advanced Usage

### Custom Training
```bash
# Train with different parameters
python -m ml.train \
    --epochs 15 \
    --batch-size 64 \
    --lr 1e-3 \
    --weight-decay 5e-4 \
    --label-smoothing 0.1 \
    --early-stop 5 \
    --out models/my_model.pt
```

### Model Evaluation
```bash
# Test model performance
python -c "
import torch
from ml.model import SimpleCNN
model = torch.jit.load('models/emnist36_balanced.pt')
print('Model loaded successfully')
"
```

### Dataset Visualization
```bash
# Generate character samples
python tools/show_all_emnist_characters.py
```

## Performance

- **Training Time**: ~5-10 minutes on modern hardware
- **Model Size**: ~2MB (TorchScript)
- **Inference Speed**: Real-time on CPU
- **Accuracy**: ~92% on test set
- **Device Support**: CPU, CUDA, Apple MPS

## Troubleshooting

### Common Issues

**Tkinter not available:**
```bash
# Install Python with Tkinter support
# macOS: brew install python-tk
# Ubuntu: sudo apt-get install python3-tk
```

**CUDA out of memory:**
```bash
# Use smaller batch size
python -m ml.train --batch-size 64
```

**Model not recognizing letters well:**
- The model is trained on clean, printed EMNIST data
- Natural handwriting may differ significantly from training data
- **Important**: Refer to the training images (`emnist_*.png` files) to see the exact writing style the model learned
- Try drawing more like the printed letters shown in the training samples
- Use `python tools/show_all_emnist_characters.py` to generate training examples
- The model expects clean, standardized characters similar to the EMNIST dataset

## Technical Details

### Data Preprocessing
- **Normalization**: MNIST mean (0.1307) and std (0.3081)
- **Case Collapsing**: A/a → 0, B/b → 1, etc.
- **Image Format**: 28×28 grayscale, inverted strokes

### Model Export
- **Format**: TorchScript for fast inference
- **Optimization**: JIT compilation for performance
- **Compatibility**: Cross-platform model loading

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- **EMNIST Dataset**: Extended MNIST by Cohen et al.
- **MNIST Dataset**: Classic handwritten digits dataset
- **PyTorch**: Deep learning framework
- **Tkinter**: Python GUI toolkit

---

**Note**: This system is designed for educational and research purposes. For production use, consider additional validation and testing with your specific handwriting data.