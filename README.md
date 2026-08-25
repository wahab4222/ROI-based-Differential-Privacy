# ROI-Guided Differential Privacy in Federated Learning for Enhanced Alzheimer’s Disease Classification

[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22093150.svg)](https://doi.org/10.5281/zenodo.22093150)

This repository contains the revised implementation accompanying the JMI manuscript on privacy-oriented gradient perturbation for federated Alzheimer's disease MRI classification.

The revised codebase supports **three-class AD/CN/MCI classification** on **OASIS** and **ADNI**, using **EfficientNet-B2**, **InceptionV3**, and **Swin-Tiny**. The experimental pipeline includes non-perturbed federated learning, uniform Gaussian gradient perturbation, spatial ROI-guided perturbation, layer-adaptive perturbation, layer-aware ROI perturbation, simulated secure aggregation, and membership-inference attack (MIA) evaluation.

**Archived revised release:** https://doi.org/10.5281/zenodo.22093150

## Important privacy statement

The method names `fed_dp`, `roi_dp`, `adaptive_dp`, and `roi_layer_dp` are retained as experimental identifiers.

**The current implementation does not provide a certified `(epsilon, delta)`-differential privacy guarantee.**

- no validated RDP, Moments Accountant, Opacus accountant, or equivalent formal accountant is used;
- clipping is not formal per-example DP-SGD clipping;
- ROI importance is derived from training gradients before perturbation and is data-dependent;
- legacy epsilon values must not be interpreted as certified privacy bounds.

Privacy is evaluated empirically using membership-inference attacks.

## Repository structure

```text
ROI-based-Differential-Privacy/
├── unified_roi_dp.py
├── README.md
├── requirements.txt
├── CITATION.cff
├── LICENSE
├── visualization/
│   ├── generate_qualitative_visualization.py
│   ├── experiment_summary.json
│   ├── final_metrics.csv
│   ├── test_predictions.csv
│   └── inceptionv3_roi_layer_beta2p0_best_val_global_model.pt
└── legacy/
    ├── client.py
    ├── server.py
    └── README.md
```

The revised Zenodo release provides the immutable archival snapshot of the software and associated reproducibility materials. The GitHub repository contains the current revised code and selected supporting artifacts.

## Main script

Use:

```text
unified_roi_dp.py
```

Inspect the CLI with:

```bash
python unified_roi_dp.py --help
```

## Supported datasets

- OASIS
- ADNI

Classes:

```text
AD
CN
MCI
```

Expected prepared layout:

```text
dataset_root/
├── train/AD train/CN train/MCI
├── val/AD   val/CN   val/MCI
└── test/AD  test/CN  test/MCI
```

MRI data are not redistributed here.

## Supported backbones

| CLI name | Architecture | Typical input size |
|---|---|---:|
| `efficientnet_b2` | EfficientNet-B2 | 260 x 260 |
| `inception_v3` | InceptionV3 | 299 x 299 |
| `swin_t` | Swin-Tiny | 224 x 224 |

The spatial ROI mechanism is naturally defined for four-dimensional convolutional gradients; Swin-Tiny is used as a transformer generalization baseline.

## Experimental modes

- `fed_nondp`: no gradient perturbation
- `fed_dp`: uniform Gaussian gradient perturbation
- `roi_dp`: spatial ROI-guided perturbation
- `roi_dp_sa`: ROI-guided perturbation with simulated secure aggregation
- `adaptive_dp`: layer-/tensor-adaptive perturbation
- `roi_layer_dp`: layer-aware ROI perturbation combining layer-level scaling with spatial scaling for convolutional gradients

The revised ROI-layer experiments include spatial-strength values such as `beta = 0.5, 1.0, 2.0`, with bounded and normalized spatial scaling.

## Federated optimization

The revised unified pipeline uses synchronous simulated federated learning with **sample-weighted FedAvg**.

The revised manuscript experiments do **not** use the FedProx formulation described in the original repository version.

The revised OASIS client simulation is image-level and class-stratified; it should not be described as a subject-disjoint clinical federation.

## Example commands

```bash
python unified_roi_dp.py --dataset oasis --mode fed_nondp --backbone efficientnet_b2 --data_root /path/to/dataset
```

```bash
python unified_roi_dp.py --dataset oasis --mode fed_dp --backbone inception_v3 --data_root /path/to/dataset
```

```bash
python unified_roi_dp.py --dataset oasis --mode roi_dp --backbone inception_v3 --data_root /path/to/dataset
```

```bash
python unified_roi_dp.py --dataset oasis --mode adaptive_dp --backbone inception_v3 --data_root /path/to/dataset
```

```bash
python unified_roi_dp.py --dataset oasis --mode roi_layer_dp --backbone inception_v3 --data_root /path/to/dataset
```

Use the experiment-specific configuration reported in the manuscript when reproducing a particular result.

## Membership-inference evaluation

The pipeline evaluates membership leakage using confidence-, entropy-, loss-, and correctness-based attack scores.

The manuscript primarily reports **MIA-AUC**:

```text
MIA-AUC ~= 0.5  -> near-random membership discrimination
higher MIA-AUC -> greater empirical membership leakage
```

MIA is an empirical privacy diagnostic and is not a formal DP guarantee.

## Qualitative visualization

The `visualization/` directory contains the files used to reproduce the qualitative visualization reported in the revised manuscript.

The visualization corresponds to:

```text
Dataset: OASIS
Backbone: InceptionV3
Mode: roi_layer_dp
Spatial beta: 2.0
```

The directory contains:

| File | Purpose |
|---|---|
| `generate_qualitative_visualization.py` | Loads the trained model and selected OASIS test samples, generates the gradient-based importance maps, derives the perturbation-scale maps, and creates the final multi-panel visualization. |
| `experiment_summary.json` | Stores the configuration and summary metadata for the selected OASIS InceptionV3 ROI-layer-DP experiment. |
| `final_metrics.csv` | Contains the final evaluation metrics for the selected experiment, including the test performance used in the manuscript. |
| `test_predictions.csv` | Contains the held-out test predictions and class probabilities used by the visualization script to select representative correctly classified AD, CN, and MCI samples. |
| `inceptionv3_roi_layer_beta2p0_best_val_global_model.pt` | Best-validation InceptionV3 checkpoint used to generate the qualitative importance and perturbation-scale maps. |

The visualization generated by the helper script contains four columns:

1. input MRI;
2. gradient-based importance map;
3. derived perturbation-scale map;
4. importance overlay.

Rows correspond to representative AD, CN, and MCI test samples.

The importance visualization is Grad-CAM-style and is generated from the final Inception block (`Mixed_7c`). The perturbation-scale map is a qualitative reconstruction of the spatial scaling behavior used by the ROI-layer configuration. It should therefore not be interpreted as an exact record of every training-time perturbation tensor.

The visualization is intended to illustrate the spatial allocation behavior of the method. It is not proof of anatomical biomarker localization and does not establish certified differential privacy.

The OASIS MRI images themselves are not redistributed in this repository. Users must obtain the dataset independently and update the dataset path in the visualization script before reproducing the figure.

## Installation

Python 3.9+ is recommended.

```bash
python -m venv venv
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CPU/CUDA environment.

## Reproducibility and versioning

The revised paper contains a historical baseline package and later reviewer-driven adaptive/ROI-layer experiments. Some runs differ in client count, rounds, local epochs, trainable scope, or final evaluation mode. They should not be pooled as if produced under one identical configuration.

The immutable revised software snapshot is:

**https://doi.org/10.5281/zenodo.22093150**

## Legacy implementation

The old repository used separate Flower `client.py` and `server.py` scripts and described an earlier OASIS-only, four-class, CBAM/FedProx-oriented workflow with epsilon-oriented privacy language.

Those scripts no longer represent the revised JMI pipeline. If retained, keep them under `legacy/`.

## Data availability

OASIS and ADNI data are not redistributed in this repository. Users must obtain them through authorized sources and comply with applicable data-use agreements.

## Citation

If you use this software, cite:

**Wahab, Abdul. ROI-Guided Privacy-Noise Allocation in Federated Learning for Alzheimer's MRI Classification. Zenodo. https://doi.org/10.5281/zenodo.22093150**

See `CITATION.cff` for machine-readable metadata.

## License

MIT License. See `LICENSE`.

## Scientific scope

The revised implementation supports the following conclusions:

- perturbation-based training generally reduces empirical membership leakage relative to non-perturbed FL;
- spatial ROI-guided perturbation can recover utility lost to uniform perturbation in selected settings;
- layer-adaptive perturbation is a strong comparator;
- layer-aware ROI perturbation substantially improves the original spatial formulation and reaches comparable performance to the adaptive baseline in the evaluated CNN experiments;
- formal differential-privacy certification remains future work.
