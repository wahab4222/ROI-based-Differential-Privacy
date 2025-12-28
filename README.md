[![DOI](https://zenodo.org/badge/1103325742.svg)](https://doi.org/10.5281/zenodo.18077682)

This repository accompanies the manuscript ‘ROI-Guided Differential Privacy in Federated Learning for Enhanced Alzheimer’s Disease Classification’ submitted to The Visual Computer.

---

# Federated ROI-DP Training on OASIS (EfficientNet-B2 / InceptionV3)

This repository contains the **server** and **client** code used to train
federated convolutional neural networks on the OASIS 4-class MRI dataset
with **record-level differential privacy** and **region-of-interest differential privacy (ROI-DP)** guided by CBAM attention.

The implementation is built on top of the **Flower** federated learning framework
and **PyTorch** vision models (EfficientNet-B2 or InceptionV3).

---

## 1. Environment

- **Python version:** `3.8.20` (recommended)
- **Hardware:** CPU is supported but GPU (CUDA) is recommended for training.

---

## 2. Required Python Packages

The code relies on the following external packages:

- `numpy`
- `torch`            – PyTorch
- `torchvision`      – vision models and transforms
- `flwr`             – Flower federated learning framework
- `scikit-learn`     – classification metrics (precision/recall/F1)
- `Pillow`           – image IO (PIL)

Plus standard library modules (`argparse`, `os`, `typing`, `logging`, etc.)
that are included with Python.

### 2.1. Creating a virtual environment (recommended)

```bash
python3.8 -m venv venv
source venv/bin/activate  # Linux/macOS
# .\venv\Scripts\activate  # Windows PowerShell
pip install --upgrade pip
````

### 2.2. Installing dependencies

> **Note:** Install the correct PyTorch build (CPU/CUDA) from the official website:
> [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Minimal setup:

```bash
pip install numpy
pip install torch torchvision      # pick the wheel matching your OS & CUDA
pip install flwr
pip install scikit-learn
pip install pillow
```

---

## 3. Repository Structure

* `server.py`
  Federated **server** implementation:

  * Builds the backbone model (EfficientNet-B2 or InceptionV3) with optional
    CBAM attention and BatchNorm→GroupNorm replacement.
  * Uses a FedProx-style strategy for federated training.
  * Aggregates client metrics and logs validation performance and DP ε statistics.
  * Tracks the best validation accuracy across federated rounds.

* `client.py`
  Federated **client** implementation:

  * Loads a client-specific subset of the OASIS dataset.
  * Builds the same backbone as the server, with optional CBAM and BN→GN.
  * Uses heavy data augmentation and Mixup to stabilise learning under DP.
  * Implements a custom `UltraLowNoiseRoiDPOptimizer` that injects spatially
    varying Gaussian noise into gradients guided by CBAM-like importance maps.
  * Reports training metrics and DP ε triplets back to the server each round.

---

## 4. Key Features

* **Backbones:** EfficientNet-B2 and InceptionV3 with a 4-class head for
  dementia stage classification.
* **Attention:** Enhanced CBAM attached to the last convolutional block
  to emphasise task-salient brain regions.
* **Normalization:** Optional replacement of BatchNorm by GroupNorm for
  better stability under DP and small batch sizes.
* **Federated Learning:** Flower-based FedProx strategy with full client
  participation and configurable number of rounds.
* **Differential Privacy:**
  * Uniform DP-SGD baseline (isotropic noise + clipping).
  * ROI-DP optimizer with **region-of-interest aware** noise allocation.
* **Regularisation:** Strong data augmentation + Mixup + label smoothing
  to improve robustness under noisy gradients.
