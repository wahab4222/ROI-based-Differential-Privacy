[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18079547.svg)](https://doi.org/10.5281/zenodo.18079547)

This repository accompanies the manuscript ‘ROI-Guided Differential Privacy in Federated Learning for Enhanced Alzheimer’s Disease Classification’.

---

# Federated ROI-DP Training on OASIS (EfficientNet-B2 / InceptionV3)

This repository contains the **server** and **client** code used to train
federated convolutional neural networks on the OASIS 4-class MRI dataset
with **region-of-interest differential privacy (ROI-DP)** guided by CBAM attention.

The implementation is built on top of the **Flower** federated learning framework
and **PyTorch** vision models (EfficientNet-B2 or InceptionV3).

---

## 1. Environment

- **Python version:** `3.8.20` (recommended)
- **Hardware:** CPU is supported but GPU (CUDA) is recommended for training.

---

## 2. Installation

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

Then install the remaining dependencies:

pip install -r requirements.txt

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

---

## 5. Running Federated Training (Quickstart)

Experiments were executed in a Jupyter-based environment. Example dataset paths may therefore reflect user-specific mount points. Replace paths with your local setup.

### 5.1. Start the federated server (copy/paste)

    python server.py --data_root /path/to/federated_dataset --server_address 0.0.0.0:8026 --model inception_v3 --num_rounds 15 --round_timeout 600 --eval_split val --img_size 299 --batch_size 16 --pretrained --fraction_fit 1.0 --fraction_evaluate 1.0 --min_fit_clients 1 --min_evaluate_clients 1 --min_available_clients 1 --use_cbam

### 5.2. Start federated clients (example with 4 clients)

Client 1:

    python client.py --server_address 127.0.0.1:8026 --data_root /path/to/federated_dataset --client_id 1 --model inception_v3 --img_size 299 --batch_size 32 --pretrained --local_epochs 5 --use_cbam --use_roi_dp --roi_dp_noise_multiplier 0.0008 --target_epsilon 3.8 --use_mixup --mixup_alpha 0.3

Clients 2–4: repeat the above command with `--client_id 2`, `--client_id 3`, and `--client_id 4`.

### 5.3. Switching to EfficientNet-B2

Use the following parameters when launching `server.py` and all clients:

- `--model efficientnet_b2`
- `--img_size 224` (recommended)

Server example:

    python server.py --data_root /path/to/federated_dataset --server_address 0.0.0.0:8026 --model efficientnet_b2 --img_size 224 --num_rounds 15 --round_timeout 600 --eval_split val --batch_size 16 --pretrained --fraction_fit 1.0 --fraction_evaluate 1.0 --min_fit_clients 1 --min_evaluate_clients 1 --min_available_clients 1 --use_cbam

---

## 6. Dataset Organization

The data_root directory should contain a locally prepared OASIS MRI dataset organized by class labels. Each federated client must operate on a disjoint subset of subjects to prevent subject-level data leakage.

Due to licensing restrictions, the OASIS dataset is not redistributed. Users are expected to prepare client-specific splits prior to training.

---

## License
This project is released under the MIT License.
