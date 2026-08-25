#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified fast federated training script for OASIS / ADNI.

This script is designed for the current Kaggle workflow:
- one file
- selectable dataset
- selectable training mode
- selectable backbone
- fast enough for limited Kaggle quota

Implemented modes:
    fed_nondp          : FedAvg, no DP, full trainable scope
    fed_dp             : FedAvg + legacy manuscript-style uniform gradient-noise DP
    roi_dp             : FedAvg + ROI-guided gradient-noise DP (IS_ROI=True)
    roi_dp_sa          : ROI-DP + secure-aggregation simulation flag
    nondp_matched_scope: FedAvg, no DP, but uses last_blocks scope + DP-matching
                         transforms/LR to isolate the scope penalty from noise penalty

Implemented backbones:
    efficientnet_b2
    inception_v3
    swin_t

Important privacy note:
    This script uses a "legacy debug" gradient-noise implementation (privacy_accounting_status
    = "legacy_debug_not_formal_dp"). It clips gradients and adds Gaussian noise before each
    optimizer step, and logs epsilon values for reference only. These epsilon values are NOT
    derived from a formal DP accountant (no Moments Accountant, no Opacus RDP) and must NOT
    be cited as certified privacy guarantees. The noise levels used here (0.002, 0.0008) are
    too small to achieve formal epsilon=3.8 at delta=1e-5; real DP at that budget would require
    noise multiplier ~1.0-1.5, which collapsed to chance accuracy in earlier experiments.
    For publishable DP guarantees, a future Opacus integration path is required.

Kaggle examples:
    !python /kaggle/working/unified_roi_dp_fast.py --dataset oasis --mode fed_dp --backbone efficientnet_b2
    !python /kaggle/working/unified_roi_dp_fast.py --dataset oasis --mode roi_dp --backbone efficientnet_b2
    !python /kaggle/working/unified_roi_dp_fast.py --dataset oasis --mode roi_dp_sa --backbone efficientnet_b2
    !python /kaggle/working/unified_roi_dp_fast.py --dataset oasis --mode fed_nondp --backbone swin_t
    !python /kaggle/working/unified_roi_dp_fast.py --dataset adni --mode fed_dp --backbone efficientnet_b2
"""

import argparse
import json
import os
import re
import time
import math
import random
import shutil
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.data.dataloader import default_collate

from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torchvision import models

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    roc_curve,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# Arguments
# ============================================================

def parse_args():
    p = argparse.ArgumentParser("Unified fast FL / DP / ROI-DP runner")

    p.add_argument("--dataset", required=True, choices=["oasis", "adni"])
    p.add_argument("--mode", required=True,
                   choices=["fed_nondp", "fed_dp", "roi_dp", "roi_dp_sa", "nondp_matched_scope", "adaptive_dp"])
    p.add_argument("--backbone", default="efficientnet_b2", choices=["efficientnet_b2", "inception_v3", "swin_t"])

    p.add_argument("--data_root", default=None, type=str)

    p.add_argument("--clients", default=6, type=int)
    p.add_argument("--rounds", default=30, type=int)
    p.add_argument("--local_epochs", default=3, type=int)
    p.add_argument("--seed", default=42, type=int)

    p.add_argument("--batch_size", default=None, type=int)
    p.add_argument("--eval_batch_size", default=64, type=int)
    p.add_argument("--num_workers", default=2, type=int)

    p.add_argument("--target_epsilon", default=3.8, type=float)
    p.add_argument("--delta", default=1e-5, type=float)
    p.add_argument("--max_grad_norm", default=1.0, type=float)

    # Low-noise DP defaults based on previous manuscript-style implementation.
    p.add_argument("--uniform_noise_multiplier", default=0.0020, type=float)
    p.add_argument("--roi_noise_multiplier", default=0.0008, type=float)
    p.add_argument("--roi_critical_scale", default=0.05, type=float)
    p.add_argument("--roi_average_scale", default=0.25, type=float)
    p.add_argument("--roi_background_scale", default=1.0, type=float)

    p.add_argument("--use_mixup", action="store_true", default=True)
    p.add_argument("--no_mixup", action="store_false", dest="use_mixup")
    p.add_argument("--mixup_alpha", default=0.3, type=float)

    p.add_argument("--use_weighted_sampler", action="store_true", default=False)
    p.add_argument("--freeze_backbone_warmup", default=2, type=int)
    p.add_argument("--dp_train_scope", default="last_blocks", choices=["head_only", "last_blocks", "full"])

    # CBAM attention: decoupled from IS_ROI so matched experiments can disable it.
    # Default False so fed_dp and roi_dp use identical architectures unless explicitly requested.
    p.add_argument("--use_cbam", action="store_true", default=False)

    p.add_argument("--no_mia", action="store_true")
    p.add_argument("--no_zip", action="store_true")
    p.add_argument("--final_eval_mode", default="auto", choices=["auto", "base", "tta"],
                   help="Final evaluation mode. 'auto' picks whichever scores higher on val "
                        "(existing behaviour). 'base' or 'tta' force the choice — use for "
                        "matched experiments so all runs are evaluated identically.")
    p.add_argument("--matched_erasing_p", default=None, type=float,
                   help="Override random erasing probability. Default None: 0.15 for DP/matched "
                        "modes, 0.05 otherwise. Pass 0.05 for relaxed matched experiments.")
    p.add_argument("--matched_lr_profile", default="auto", choices=["auto", "dp", "standard"],
                   help="LR profile override. 'auto' = mode-based (DP/matched-scope use DP LRs). "
                        "'standard' = head=5e-4, backbone=4e-5 for all modes. "
                        "'dp' = head=3e-4, backbone=2e-5 for all modes.")

    return p.parse_args()


args = parse_args()

DATASET_NAME = args.dataset.lower()
MODE = args.mode.lower()
BACKBONE = args.backbone.lower()

IS_DP = MODE in ["fed_dp", "roi_dp", "roi_dp_sa", "adaptive_dp"]
IS_ROI = MODE in ["roi_dp", "roi_dp_sa"]
USE_SECURE_AGG = MODE == "roi_dp_sa"
# nondp_matched_scope: no noise, but uses last_blocks scope + DP-matching transforms/LR.
# Allows isolating the training-scope penalty from the noise penalty.
IS_MATCHED_SCOPE = MODE == "nondp_matched_scope"

SEED = args.seed
NUM_CLIENTS = args.clients
ROUNDS = args.rounds
LOCAL_EPOCHS = args.local_epochs

TARGET_EPSILON = args.target_epsilon
DELTA = args.delta
MAX_GRAD_NORM = args.max_grad_norm

UNIFORM_NOISE = args.uniform_noise_multiplier
ROI_NOISE = args.roi_noise_multiplier

# ── Config sanity warnings ──────────────────────────────────────────────────
# Populated at startup, printed in the banner, and stored in experiment_summary.json.
STARTUP_WARNINGS = []
if IS_DP and args.use_cbam:
    STARTUP_WARNINGS.append(
        "CBAM enabled for a DP mode — architecture differs from non-DP baseline "
        "(confound). Pass --use_cbam only when intentionally testing CBAM's effect."
    )
if IS_ROI and abs(ROI_NOISE - UNIFORM_NOISE) > 1e-9:
    STARTUP_WARNINGS.append(
        f"roi_noise_multiplier ({ROI_NOISE}) != uniform_noise_multiplier ({UNIFORM_NOISE}). "
        "Not a matched noise-budget comparison. For the matched debug experiment pass "
        "--roi_noise_multiplier equal to --uniform_noise_multiplier."
    )
if IS_DP and args.dp_train_scope == "full":
    STARTUP_WARNINGS.append(
        "Full-scope DP training without per-sample gradient clipping (Opacus) is NOT "
        "formal DP. privacy_accounting_status = legacy_debug_not_formal_dp."
    )
# No warning for nondp_matched_scope + dp_train_scope=full: that is the intended
# full-scope matched comparison to isolate noise penalty from scope penalty.
# ────────────────────────────────────────────────────────────────────────────

RUN_MIA = not args.no_mia
SAVE_ZIP = not args.no_zip

CLASSES = ["AD", "CN", "MCI"]
NUM_CLASSES = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


set_seed(SEED)


# ============================================================
# Dataset paths and output paths
# ============================================================

def candidate_roots(dataset):
    if dataset == "oasis":
        # New uploaded Kaggle OASIS 5400-image dataset.
        # User-provided dataset root:
        #   /kaggle/input/datasets/abdul10022/new-oasis
        # In Kaggle it contains:
        #   /kaggle/input/datasets/abdul10022/new-oasis/oasis/train
        #   /kaggle/input/datasets/abdul10022/new-oasis/oasis/val
        #   /kaggle/input/datasets/abdul10022/new-oasis/oasis/test
        #
        # Older paths are kept only as fallbacks.
        return [
            Path("/kaggle/input/datasets/abdul10022/new-oasis/oasis"),
            Path("/kaggle/input/datasets/abdul10022/new-oasis"),
            Path("/kaggle/input/datasets/abdul10022/new_oasis/oasis"),
            Path("/kaggle/input/datasets/abdul10022/new_oasis"),
            Path("/kaggle/input/new-oasis/oasis"),
            Path("/kaggle/input/new-oasis"),
            Path("/kaggle/input/new_oasis/oasis"),
            Path("/kaggle/input/new_oasis"),
            Path("/kaggle/working/oasis_5400_image_level_stratified/oasis"),
            Path("/kaggle/input/oasis-5400-image-level/oasis"),
            Path("/kaggle/input/oasis_5400_image_level_stratified/oasis"),
            Path("/kaggle/input/datasets/abdul10022/mydataset/oasis/oasis"),
        ]

    return [
        Path("/kaggle/working/adni_5400_image_level_stratified/adni"),
        Path("/kaggle/input/adni-5400-image-level/adni"),
        Path("/kaggle/input/adni_5400_image_level_stratified/adni"),
        Path("/kaggle/input/datasets/abdul10022/mydataset/adni/adni"),
    ]


def expand_dataset_roots(root_list, dataset_name):
    """Accept either the exact folder containing train/val/test or a parent folder."""
    expanded = []
    seen = set()

    for root in root_list:
        root = Path(root)
        candidates = [
            root,
            root / dataset_name,
            root / "oasis",
            root / "adni",
        ]

        for c in candidates:
            key = str(c)
            if key not in seen:
                expanded.append(c)
                seen.add(key)

    return expanded

roots = expand_dataset_roots([Path(args.data_root)] if args.data_root else candidate_roots(DATASET_NAME), DATASET_NAME)

DATA_ROOT = None
for r in roots:
    if (r / "train").exists() and (r / "val").exists() and (r / "test").exists():
        DATA_ROOT = r
        break

if DATA_ROOT is None:
    print("Checked roots:")
    for r in roots:
        print(" ", r)
    raise RuntimeError("No dataset root with train/val/test folders found.")

TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
TEST_DIR = DATA_ROOT / "test"

# Include a source tag to avoid accidentally mixing old working-output checkpoints
# with the newly uploaded OASIS dataset.
data_root_text = str(DATA_ROOT).lower()
if DATASET_NAME == "oasis" and ("new-oasis" in data_root_text or "new_oasis" in data_root_text):
    DATA_SOURCE_TAG = "new_oasis"
else:
    DATA_SOURCE_TAG = DATASET_NAME

OUT_DIR = Path(f"/kaggle/working/{DATA_SOURCE_TAG}_{BACKBONE}_{MODE}_{NUM_CLIENTS}clients_e{ROUNDS}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_VAL_MODEL_PATH = OUT_DIR / "best_val_global_model.pt"
BEST_TARGET_MODEL_PATH = OUT_DIR / "best_target_global_model.pt"
TRAIN_LOG_PATH = OUT_DIR / "federated_training_log.csv"
PRIVACY_LOG_PATH = OUT_DIR / "privacy_accounting.csv"
FINAL_METRICS_PATH = OUT_DIR / "final_metrics.csv"
MIA_DIR = OUT_DIR / "membership_inference"
MIA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Image sizes and hyperparameters
# ============================================================

if BACKBONE == "inception_v3":
    IMG_SIZE = 299
    RESIZE_SIZE = 330
elif BACKBONE == "swin_t":
    IMG_SIZE = 224
    RESIZE_SIZE = 256
else:
    IMG_SIZE = 260
    RESIZE_SIZE = 288

BATCH_SIZE = args.batch_size if args.batch_size else (32 if BACKBONE != "inception_v3" else 24)
EVAL_BATCH_SIZE = args.eval_batch_size

# LR profile: 'auto' applies DP-style LRs for IS_DP and IS_MATCHED_SCOPE, standard otherwise.
# Override with --matched_lr_profile {dp,standard} to decouple from mode for relaxed experiments.
_use_dp_lrs = (
    args.matched_lr_profile == "dp"
    or (args.matched_lr_profile == "auto" and (IS_DP or IS_MATCHED_SCOPE))
)
if _use_dp_lrs:
    HEAD_LR_WARMUP = 8e-4
    BACKBONE_LR = 2e-5
    HEAD_LR = 3e-4
    LABEL_SMOOTHING = 0.03
    LR_PROFILE_LABEL = "dp"
else:
    HEAD_LR_WARMUP = 1.2e-3
    BACKBONE_LR = 4e-5
    HEAD_LR = 5e-4
    LABEL_SMOOTHING = 0.02
    LR_PROFILE_LABEL = "standard"

WEIGHT_DECAY = 7e-5
DROPOUT = 0.30
USE_AMP = DEVICE.type == "cuda"

ERASING_P = (
    args.matched_erasing_p
    if args.matched_erasing_p is not None
    else (0.15 if (IS_DP or IS_MATCHED_SCOPE) else 0.05)
)

print("============================================================")
print("UNIFIED FAST FEDERATED TRAINING")
print("DATASET:", DATASET_NAME)
print("MODE:", MODE)
print("BACKBONE:", BACKBONE)
print("IS_DP:", IS_DP)
print("IS_ROI:", IS_ROI)
print("IS_MATCHED_SCOPE:", IS_MATCHED_SCOPE)
print("USE_CBAM:", args.use_cbam)
print("SECURE_AGG_SIM:", USE_SECURE_AGG)
print("DATA_ROOT:", DATA_ROOT)
print("DATA_SOURCE_TAG:", DATA_SOURCE_TAG)
print("OUTPUT:", OUT_DIR)
print("DEVICE:", DEVICE)
print("GPU count:", torch.cuda.device_count())
print("CLIENTS:", NUM_CLIENTS)
print("ROUNDS:", ROUNDS)
print("LOCAL_EPOCHS:", LOCAL_EPOCHS)
print("BATCH_SIZE:", BATCH_SIZE)
print("IMG_SIZE:", IMG_SIZE)
print("DP target epsilon:", TARGET_EPSILON if IS_DP else "N/A")
print("privacy_accounting_status:", "legacy_debug_not_formal_dp" if IS_DP else "not_applicable")
print("Uniform noise:", UNIFORM_NOISE if MODE == "fed_dp" else "N/A")
print("ROI noise:", ROI_NOISE if IS_ROI else "N/A")
print("Train scope:", args.dp_train_scope if (IS_DP or IS_MATCHED_SCOPE) else "full")
print("Mixup:", args.use_mixup if (IS_DP or IS_MATCHED_SCOPE) else False)
print("Random erasing p:", ERASING_P)
print("LR profile:", LR_PROFILE_LABEL)
print("Head LR (warmup):", HEAD_LR_WARMUP)
print("Head LR (finetune):", HEAD_LR)
print("Backbone LR:", BACKBONE_LR)
print("Final eval mode (flag):", args.final_eval_mode)
print("============================================================")
if STARTUP_WARNINGS:
    print("\nCONFIG WARNINGS:")
    for _w in STARTUP_WARNINGS:
        print(" !", _w)
    print()


# ============================================================
# Robust dataset utilities
# ============================================================

def pil_rgb_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long), []
    return default_collate(batch)


class SafeImageFolderWithPaths(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform, loader=pil_rgb_loader)

    def __getitem__(self, index):
        try:
            img, label = super().__getitem__(index)
            return img, label, self.samples[index][0]
        except Exception:
            return None


class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform, loader=pil_rgb_loader)

    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception:
            return None


class AutoCropBrain:
    def __init__(self, threshold=8, margin=0.06):
        self.threshold = threshold
        self.margin = margin

    def __call__(self, img):
        arr = np.asarray(img.convert("L"))
        mask = arr > self.threshold
        if mask.sum() < 10:
            return img
        ys, xs = np.where(mask)
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        h, w = arr.shape
        py = int((y2 - y1 + 1) * self.margin)
        px = int((x2 - x1 + 1) * self.margin)
        y1, y2 = max(0, y1 - py), min(h - 1, y2 + py)
        x1, x2 = max(0, x1 - px), min(w - 1, x2 + px)
        return img.crop((x1, y1, x2 + 1, y2 + 1))


def parse_subject_id(path):
    stem = Path(path).stem
    m = re.search(r"(OAS\d+_\d+|ADNI\d+_\d+|ADNI_\d+|[A-Za-z0-9]+_\d+)", stem)
    return m.group(1) if m else stem


# ============================================================
# Transforms
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_tfms = transforms.Compose([
    AutoCropBrain(8, 0.06),
    transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.86, 1.0), ratio=(0.96, 1.04)),
    transforms.RandomHorizontalFlip(p=0.25),
    transforms.RandomRotation(degrees=6, fill=0),
    transforms.ColorJitter(brightness=0.06, contrast=0.06),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=ERASING_P, scale=(0.01, 0.08)),
])

eval_tfms = transforms.Compose([
    AutoCropBrain(8, 0.06),
    transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

tta_tfms = [
    ("base", eval_tfms),
    ("rot_pos", transforms.Compose([
        AutoCropBrain(8, 0.06),
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.Lambda(lambda img: TF.rotate(img, angle=2, fill=0)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])),
    ("rot_neg", transforms.Compose([
        AutoCropBrain(8, 0.06),
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.Lambda(lambda img: TF.rotate(img, angle=-2, fill=0)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])),
]


# ============================================================
# Dataset loading and client split
# ============================================================

train_dataset = SafeImageFolderWithPaths(TRAIN_DIR, transform=train_tfms)
train_eval_dataset = SafeImageFolderWithPaths(TRAIN_DIR, transform=eval_tfms)
val_dataset = SafeImageFolderWithPaths(VAL_DIR, transform=eval_tfms)
test_dataset = SafeImageFolderWithPaths(TEST_DIR, transform=eval_tfms)

if train_dataset.classes != CLASSES:
    raise RuntimeError(f"Expected class order {CLASSES}, got {train_dataset.classes}")

def print_counts(ds, name):
    targets = np.array(ds.targets)
    print(f"\n{name} counts:")
    for i, c in enumerate(CLASSES):
        print(f"  {c}: {int(np.sum(targets == i))}")
    print("  Total:", len(targets))

print("\nClasses:", train_dataset.classes)
print("Class-to-index:", train_dataset.class_to_idx)
print_counts(train_dataset, "Train")
print_counts(val_dataset, "Val")
print_counts(test_dataset, "Test")


def make_client_indices(ds, num_clients, seed):
    rng = np.random.default_rng(seed)
    targets = np.array(ds.targets)
    clients = [[] for _ in range(num_clients)]
    for cls_idx in range(NUM_CLASSES):
        idxs = np.where(targets == cls_idx)[0]
        rng.shuffle(idxs)
        chunks = np.array_split(idxs, num_clients)
        for c, ch in enumerate(chunks):
            clients[c].extend(ch.tolist())
    for c in range(num_clients):
        rng.shuffle(clients[c])
    return clients


def make_weighted_sampler_for_subset(full_ds, subset_indices):
    targets = np.array(full_ds.targets)[subset_indices]
    counts = np.bincount(targets, minlength=NUM_CLASSES).astype(np.float64)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    weights = [inv[y] for y in targets]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


client_indices = make_client_indices(train_dataset, NUM_CLIENTS, SEED)

print("\nClient distributions:")
targets_np = np.array(train_dataset.targets)
client_loaders = []
for cid, idxs in enumerate(client_indices):
    labels = targets_np[idxs]
    dist = {CLASSES[i]: int(np.sum(labels == i)) for i in range(NUM_CLASSES)}
    print(f"  Client {cid}: total={len(idxs)} | {dist}")

    subset = Subset(train_dataset, idxs)
    sampler = make_weighted_sampler_for_subset(train_dataset, idxs) if args.use_weighted_sampler and not IS_DP else None
    _cuda = torch.cuda.is_available()
    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=_cuda,
        persistent_workers=(args.num_workers > 0 and _cuda),
        drop_last=False,
        collate_fn=safe_collate,
    )
    client_loaders.append(loader)


_cuda = torch.cuda.is_available()
_eval_dl_kw = dict(shuffle=False, num_workers=args.num_workers, pin_memory=_cuda,
                   persistent_workers=(args.num_workers > 0 and _cuda), collate_fn=safe_collate)
train_eval_loader = DataLoader(train_eval_dataset, batch_size=EVAL_BATCH_SIZE, **_eval_dl_kw)
val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, **_eval_dl_kw)
test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, **_eval_dl_kw)


# ============================================================
# Attention modules and model builders
# ============================================================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class EnhancedChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        mid = max(1, channels // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, mid, 1, bias=False)
        self.fc2 = nn.Conv2d(mid, channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        mx = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg + mx)


class EnhancedSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg, mx], dim=1)
        return self.sigmoid(self.conv(x))


class EnhancedCBAM(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = EnhancedChannelAttention(channels, ratio)
        self.sa = EnhancedSpatialAttention(kernel_size)
        self.residual_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        y = x * self.ca(x)
        y = y * self.sa(y)
        return x + self.residual_scale * (y - x)


def pick_gn_groups(channels):
    for g in [32, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


def replace_bn_with_gn(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(pick_gn_groups(child.num_features), child.num_features))
        else:
            replace_bn_with_gn(child)
    return module


def build_model(pretrained=True, use_cbam=False, replace_bn=False):
    if BACKBONE == "efficientnet_b2":
        weights = None
        if pretrained:
            try:
                weights = models.EfficientNet_B2_Weights.DEFAULT
            except Exception:
                weights = None
        model = models.efficientnet_b2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(in_features, NUM_CLASSES))

        if use_cbam:
            # last EfficientNet-B2 feature block outputs 1408 channels
            model.features[-1] = nn.Sequential(model.features[-1], EnhancedCBAM(1408))

    elif BACKBONE == "inception_v3":
        weights = None
        if pretrained:
            try:
                weights = models.Inception_V3_Weights.DEFAULT
            except Exception:
                weights = None
        model = models.inception_v3(weights=weights, aux_logits=True)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        model.aux_logits = False
        model.AuxLogits = None

        if use_cbam:
            model.Mixed_7c = nn.Sequential(model.Mixed_7c, EnhancedCBAM(2048))

    elif BACKBONE == "swin_t":
        weights = None
        if pretrained:
            try:
                weights = models.Swin_T_Weights.DEFAULT
            except Exception:
                weights = None
        model = models.swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, NUM_CLASSES)

    else:
        raise ValueError(BACKBONE)

    if replace_bn:
        model = replace_bn_with_gn(model)

    return model


def is_head_param(name):
    return ("classifier" in name) or name.startswith("fc") or name.startswith("head")


def configure_trainable(model, phase):
    # Warmup: classifier/head only.
    if phase == "warmup":
        for name, p in model.named_parameters():
            p.requires_grad = is_head_param(name)
        return

    # Full-scope only when: plain non-DP (not matched-scope) OR explicit full flag.
    if ((not IS_DP) and (not IS_MATCHED_SCOPE)) or args.dp_train_scope == "full":
        for p in model.parameters():
            p.requires_grad = True
        return

    # DP and nondp_matched_scope: train only head + final blocks.
    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if is_head_param(name):
            p.requires_grad = True
        elif args.dp_train_scope == "last_blocks":
            if BACKBONE == "efficientnet_b2" and ("features.7" in name or "features.8" in name):
                p.requires_grad = True
            elif BACKBONE == "inception_v3" and ("Mixed_7" in name):
                p.requires_grad = True
            elif BACKBONE == "swin_t" and (name.startswith("features.6") or name.startswith("features.7")):
                p.requires_grad = True


def freeze_bn_eval(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def make_optimizer(model, phase):
    if phase == "warmup":
        return optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=HEAD_LR_WARMUP, weight_decay=WEIGHT_DECAY)

    head_params = []
    body_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if is_head_param(name):
            head_params.append(p)
        else:
            body_params.append(p)

    groups = []
    if len(body_params) > 0:
        groups.append({"params": body_params, "lr": BACKBONE_LR})
    if len(head_params) > 0:
        groups.append({"params": head_params, "lr": HEAD_LR})
    return optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


# ============================================================
# Noise / DP functions
# ============================================================

@torch.no_grad()
def _add_adaptive_gradient_noise(model, base_noise):
    """Sensitivity-aware (global, per-layer) DP noise baseline  [Reviewer #1, C3].

    Adaptive but NOT spatial. Each parameter tensor is perturbed with a noise
    level scaled by its OWN global gradient magnitude relative to all layers:
      - high-magnitude (task-informative) layers -> less noise (roi_critical_scale)
      - low-magnitude layers                     -> more noise (roi_background_scale)

    Same base multiplier, same per-parameter clipping, same scale range as
    ROI-DP; the ONLY difference from roi_dp is allocation granularity (per-layer
    scalar vs per-element spatial map), and from fed_dp is adaptive-vs-flat.
    Heuristic; privacy validated empirically via MIA.
    """
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return
    norms = []
    for p in params:
        grad = p.grad.data
        gn = grad.norm()
        if gn > MAX_GRAD_NORM:
            p.grad.data = grad * (MAX_GRAD_NORM / (gn + 1e-6))
        norms.append(float(p.grad.data.norm().item()))
    lo, hi = min(norms), max(norms)
    span = (hi - lo) + 1e-12
    for p, n in zip(params, norms):
        s = (n - lo) / span  # [0,1]; higher => more sensitive / task-informative
        scale = args.roi_background_scale + s * (args.roi_critical_scale - args.roi_background_scale)
        p.grad.data.add_(torch.randn_like(p.grad.data) * (base_noise * MAX_GRAD_NORM * scale))


def add_gradient_noise(model, mode):
    if not IS_DP:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        return

    if mode == "adaptive_dp":
        _add_adaptive_gradient_noise(model, ROI_NOISE)
        return

    # Clip first.
    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.grad is not None, model.parameters()), max_norm=MAX_GRAD_NORM)

    if mode == "fed_dp":
        base_noise = UNIFORM_NOISE
    else:
        base_noise = ROI_NOISE

    for p in model.parameters():
        if p.grad is None:
            continue

        grad = p.grad.data

        if mode in ["roi_dp", "roi_dp_sa"] and grad.dim() == 4:
            # ROI importance proxy from gradient magnitude, as in the previous repo approach.
            importance = torch.mean(torch.abs(grad), dim=0, keepdim=True)
            importance = F.interpolate(importance, size=grad.shape[2:], mode="bilinear", align_corners=False)
            mn, mx = importance.min(), importance.max()
            if mx > mn:
                importance = (importance - mn) / (mx - mn + 1e-8)
            else:
                importance = torch.zeros_like(importance)

            critical = (importance > 0.80).float()
            average = ((importance > 0.40) & (importance <= 0.80)).float()
            background = (importance <= 0.40).float()

            noise_scale = (
                critical * args.roi_critical_scale
                + average * args.roi_average_scale
                + background * args.roi_background_scale
            )

            grad.add_(torch.randn_like(grad) * base_noise * MAX_GRAD_NORM * noise_scale)
        else:
            grad.add_(torch.randn_like(grad) * base_noise * MAX_GRAD_NORM)


def legacy_epsilon_values(round_id):
    if not IS_DP:
        return np.nan, np.nan, np.nan

    progress = round_id / max(ROUNDS, 1)

    if MODE == "fed_dp":
        eps = TARGET_EPSILON * progress
        return eps, eps, eps

    # ROI-DP manuscript-style triplet. Conservative bound is epsilon_max.
    eps_min_final = 0.90
    eps_mean_final = 2.00
    eps_max_final = min(3.04, TARGET_EPSILON)
    return eps_min_final * progress, eps_mean_final * progress, eps_max_final * progress


# ============================================================
# Mixup
# ============================================================

def mixup_data(x, y, alpha=0.3):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


# ============================================================
# Evaluation
# ============================================================

criterion_eval = nn.CrossEntropyLoss()


def compute_metrics_from_probs(y_true, probs):
    preds = np.argmax(probs, axis=1)

    out = {
        "accuracy": accuracy_score(y_true, preds),
        "balanced_accuracy": balanced_accuracy_score(y_true, preds),
        "precision_macro": precision_score(y_true, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, preds, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, preds, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, preds, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, preds, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, preds, labels=[0, 1, 2]),
        "preds": preds,
    }

    try:
        out["auc_macro_ovr"] = roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=[0, 1, 2])
    except Exception:
        out["auc_macro_ovr"] = np.nan
    try:
        out["auc_weighted_ovr"] = roc_auc_score(y_true, probs, multi_class="ovr", average="weighted", labels=[0, 1, 2])
    except Exception:
        out["auc_weighted_ovr"] = np.nan

    for i, c in enumerate(CLASSES):
        try:
            out[f"auc_{c}_ovr"] = roc_auc_score((np.asarray(y_true) == i).astype(int), probs[:, i])
        except Exception:
            out[f"auc_{c}_ovr"] = np.nan

    return out


def selection_score(m):
    auc = 0.0 if np.isnan(m["auc_macro_ovr"]) else m["auc_macro_ovr"]
    return m["accuracy"] + 0.35 * m["f1_macro"] + 0.15 * m["balanced_accuracy"] + 0.10 * auc


@torch.no_grad()
def collect_probs_loader(model, loader):
    model.eval()
    probs_all = []
    labels_all = []
    paths_all = []
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        if batch is None:
            continue
        images, labels, paths = batch
        if images.numel() == 0:
            continue

        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits = model(images)
        if isinstance(logits, tuple):
            logits = logits[0]

        loss = criterion_eval(logits, labels)
        probs = F.softmax(logits, dim=1)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        probs_all.append(probs.cpu().numpy())
        labels_all.extend(labels.cpu().numpy().tolist())
        paths_all.extend(list(paths))

    probs = np.concatenate(probs_all, axis=0)
    labels = np.asarray(labels_all, dtype=np.int64)
    metrics = compute_metrics_from_probs(labels, probs)
    metrics["loss"] = total_loss / max(total_n, 1)
    metrics["probs"] = probs
    metrics["labels"] = labels
    metrics["paths"] = paths_all
    return metrics


@torch.no_grad()
def evaluate_tta_root(model, root_dir):
    probs_sum = None
    labels_ref = None
    paths_ref = None

    for name, tfm in tta_tfms:
        ds = SafeImageFolderWithPaths(root_dir, transform=tfm)
        _cuda = torch.cuda.is_available()
        loader = DataLoader(ds, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                            num_workers=args.num_workers, pin_memory=_cuda,
                            persistent_workers=(args.num_workers > 0 and _cuda),
                            collate_fn=safe_collate)
        m = collect_probs_loader(model, loader)

        if probs_sum is None:
            probs_sum = m["probs"].copy()
            labels_ref = m["labels"].copy()
            paths_ref = list(m["paths"])
        else:
            if paths_ref != list(m["paths"]):
                raise RuntimeError(f"TTA path mismatch: {name}")
            probs_sum += m["probs"]

    probs = probs_sum / len(tta_tfms)
    metrics = compute_metrics_from_probs(labels_ref, probs)
    metrics["loss"] = float(-np.mean(np.log(np.clip(probs[np.arange(len(labels_ref)), labels_ref], 1e-12, 1.0))))
    metrics["probs"] = probs
    metrics["labels"] = labels_ref
    metrics["paths"] = paths_ref
    return metrics


def print_metrics(name, m):
    print("\n" + name)
    for k in [
        "loss", "accuracy", "balanced_accuracy",
        "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted",
        "auc_macro_ovr", "auc_weighted_ovr",
        "auc_AD_ovr", "auc_CN_ovr", "auc_MCI_ovr",
    ]:
        print(f"{k}: {m[k]}")
    print("confusion_matrix:")
    print(m["confusion_matrix"])


def save_predictions(m, path):
    rows = []
    for p, y, pred, prob in zip(m["paths"], m["labels"], m["preds"], m["probs"]):
        rows.append({
            "path": p,
            "subject_id": parse_subject_id(p),
            "true_idx": int(y),
            "true_label": CLASSES[int(y)],
            "pred_idx": int(pred),
            "pred_label": CLASSES[int(pred)],
            "correct": int(y) == int(pred),
            "prob_AD": float(prob[0]),
            "prob_CN": float(prob[1]),
            "prob_MCI": float(prob[2]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


# ============================================================
# Local client training and FedAvg
# ============================================================

criterion_train = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)


def train_one_client(global_state, loader, round_id, client_id, phase):
    model = build_model(pretrained=False, use_cbam=args.use_cbam, replace_bn=False).to(DEVICE)
    model.load_state_dict(global_state)

    configure_trainable(model, phase)
    optimizer = make_optimizer(model, phase)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for _ in range(LOCAL_EPOCHS):
        model.train()
        freeze_bn_eval(model)

        for batch in loader:
            if batch is None:
                continue
            images, labels, paths = batch
            if images.numel() == 0:
                continue

            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            use_mix = (IS_DP or IS_MATCHED_SCOPE) and args.use_mixup and images.size(0) > 1
            if use_mix:
                images_m, ya, yb, lam = mixup_data(images, labels, args.mixup_alpha)
            else:
                images_m, ya, yb, lam = images, labels, labels, 1.0

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = model(images_m)
                if isinstance(logits, tuple):
                    logits = logits[0]
                if use_mix:
                    loss = mixup_loss(criterion_train, logits, ya, yb, lam)
                else:
                    loss = criterion_train(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            add_gradient_noise(model, MODE)

            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                eval_logits = model(images)
                if isinstance(eval_logits, tuple):
                    eval_logits = eval_logits[0]
                preds = eval_logits.argmax(dim=1)

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_correct += (preds == labels).sum().item()
            total_n += bs

    state = OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())
    del model
    torch.cuda.empty_cache()

    return {
        "state": state,
        "samples": len(loader.dataset),
        "loss": total_loss / max(total_n, 1),
        "accuracy": total_correct / max(total_n, 1),
    }


def fedavg(client_states, client_sizes):
    total = float(sum(client_sizes))
    keys = client_states[0].keys()
    new_state = OrderedDict()

    for key in keys:
        first = client_states[0][key]
        if torch.is_floating_point(first):
            avg = torch.zeros_like(first, dtype=torch.float32)
            for st, sz in zip(client_states, client_sizes):
                avg += st[key].float() * (float(sz) / total)
            new_state[key] = avg.to(dtype=first.dtype)
        else:
            new_state[key] = first.clone()
    return new_state


# ============================================================
# Training loop
# ============================================================

global_model = build_model(pretrained=True, use_cbam=args.use_cbam, replace_bn=False).to(DEVICE)
global_state = OrderedDict((k, v.detach().cpu().clone()) for k, v in global_model.state_dict().items())

# Count trainable params for the finetune phase so it's logged once at startup.
configure_trainable(global_model, "finetune")
TRAINABLE_PARAMS = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
TOTAL_PARAMS = sum(p.numel() for p in global_model.parameters())
print(
    f"Trainable parameters (finetune scope): "
    f"{TRAINABLE_PARAMS:,} / {TOTAL_PARAMS:,} "
    f"({100.0 * TRAINABLE_PARAMS / TOTAL_PARAMS:.1f}%)"
)

best_val_score = -1e9
best_target_score = -1e9
best_val_round = -1
best_target_round = -1
history = []
privacy_history = []

start_time = time.time()

for rnd in range(1, ROUNDS + 1):
    round_start = time.time()
    phase = "warmup" if rnd <= args.freeze_backbone_warmup else "finetune"

    print("\n" + "=" * 80)
    print(f"SERVER ROUND {rnd}/{ROUNDS} | Mode={MODE} | Backbone={BACKBONE} | Phase={phase}")
    print("=" * 80)

    client_states = []
    client_sizes = []
    client_losses = []
    client_accs = []

    for cid, loader in enumerate(client_loaders):
        print(f"Client {cid} training...")
        res = train_one_client(global_state, loader, rnd, cid, phase)

        client_states.append(res["state"])
        client_sizes.append(res["samples"])
        client_losses.append(res["loss"])
        client_accs.append(res["accuracy"])

        print(f"Client {cid} done | samples={res['samples']} | loss={res['loss']:.4f} | acc={res['accuracy']:.4f}")

    if USE_SECURE_AGG:
        print("Secure aggregation simulation: server receives only aggregate-equivalent updates.")

    global_state = fedavg(client_states, client_sizes)
    global_model.load_state_dict(global_state)
    global_model.to(DEVICE)
    global_model.eval()

    train_m = collect_probs_loader(global_model, train_eval_loader)
    val_m = collect_probs_loader(global_model, val_loader)
    val_score = selection_score(val_m)

    eps_min, eps_mean, eps_max = legacy_epsilon_values(rnd)

    if IS_DP:
        privacy_row = {
            "round": rnd,
            "epsilon_min": eps_min,
            "epsilon_mean": eps_mean,
            "epsilon_max": eps_max,
            "target_epsilon": TARGET_EPSILON,
            "delta": DELTA,
            "uniform_noise_multiplier": UNIFORM_NOISE if MODE == "fed_dp" else np.nan,
            "roi_noise_multiplier": ROI_NOISE if IS_ROI else np.nan,
            "accounting": "legacy_debug_not_formal_dp",
            "privacy_accounting_status": "legacy_debug_not_formal_dp",
        }
        privacy_history.append(privacy_row)
        pd.DataFrame(privacy_history).to_csv(PRIVACY_LOG_PATH, index=False)

    round_time = time.time() - round_start

    row = {
        "dataset": DATASET_NAME,
        "mode": MODE,
        "backbone": BACKBONE,
        "round": rnd,
        "phase": phase,
        "num_clients": NUM_CLIENTS,
        "local_epochs": LOCAL_EPOCHS,
        "dp": IS_DP,
        "roi_dp": IS_ROI,
        "secure_aggregation": USE_SECURE_AGG,
        "epsilon_min": eps_min,
        "epsilon_mean": eps_mean,
        "epsilon_max": eps_max,
        "target_epsilon": TARGET_EPSILON if IS_DP else np.nan,
        "delta": DELTA if IS_DP else np.nan,
        "client_loss_mean": float(np.mean(client_losses)),
        "client_accuracy_mean": float(np.mean(client_accs)),
        "train_loss": train_m["loss"],
        "train_accuracy": train_m["accuracy"],
        "train_f1_macro": train_m["f1_macro"],
        "train_auc_macro_ovr": train_m["auc_macro_ovr"],
        "val_loss": val_m["loss"],
        "val_accuracy": val_m["accuracy"],
        "val_f1_macro": val_m["f1_macro"],
        "val_auc_macro_ovr": val_m["auc_macro_ovr"],
        "val_selection_score": val_score,
        "round_time_min": round_time / 60.0,
    }
    history.append(row)
    pd.DataFrame(history).to_csv(TRAIN_LOG_PATH, index=False)

    msg = (
        f"ROUND {rnd} SUMMARY | client_acc={row['client_accuracy_mean']:.4f} | "
        f"train_acc={train_m['accuracy']:.4f} | train_f1={train_m['f1_macro']:.4f} | "
        f"val_acc={val_m['accuracy']:.4f} | val_f1={val_m['f1_macro']:.4f} | "
        f"val_auc={val_m['auc_macro_ovr']:.4f}"
    )
    if IS_DP:
        msg += f" | eps=({eps_min:.3f},{eps_mean:.3f},{eps_max:.3f})"
    msg += f" | time={round_time/60.0:.2f} min"
    print(msg)

    if val_score > best_val_score:
        best_val_score = val_score
        best_val_round = rnd
        torch.save(global_model.state_dict(), BEST_VAL_MODEL_PATH)
        print(f"New best validation model saved at round {rnd}")

    if train_m["accuracy"] >= 0.95 and val_m["accuracy"] >= 0.85:
        target_score = val_score + 0.05 * train_m["accuracy"]
        if target_score > best_target_score:
            best_target_score = target_score
            best_target_round = rnd
            torch.save(global_model.state_dict(), BEST_TARGET_MODEL_PATH)
            print(f"New best target model saved at round {rnd}")

total_time = time.time() - start_time

print("\n" + "=" * 80)
print("TRAINING FINISHED")
print("Best validation round:", best_val_round)
print("Best target round:", best_target_round)
print("Total time min:", total_time / 60.0)
print("=" * 80)


# ============================================================
# Final evaluation
# ============================================================

if BEST_TARGET_MODEL_PATH.exists():
    selected_path = BEST_TARGET_MODEL_PATH
    selected_type = "best_target_train95_val85"
else:
    selected_path = BEST_VAL_MODEL_PATH
    selected_type = "best_validation"

final_model = build_model(pretrained=False, use_cbam=args.use_cbam, replace_bn=False).to(DEVICE)
final_model.load_state_dict(torch.load(selected_path, map_location=DEVICE))
final_model.eval()

base_val = collect_probs_loader(final_model, val_loader)

if args.final_eval_mode == "auto":
    tta_val = evaluate_tta_root(final_model, VAL_DIR)
    use_tta = selection_score(tta_val) >= selection_score(base_val)
elif args.final_eval_mode == "tta":
    tta_val = evaluate_tta_root(final_model, VAL_DIR)
    use_tta = True
else:  # "base"
    tta_val = None
    use_tta = False

if use_tta:
    final_eval_mode = "tta"
    final_train = evaluate_tta_root(final_model, TRAIN_DIR)
    final_val = tta_val
    final_test = evaluate_tta_root(final_model, TEST_DIR)
else:
    final_eval_mode = "base"
    final_train = collect_probs_loader(final_model, train_eval_loader)
    final_val = base_val
    final_test = collect_probs_loader(final_model, test_loader)

print("\nSelected model:", selected_type)
print("Selected path:", selected_path)
print("Final eval mode:", final_eval_mode)

print_metrics("FINAL TRAIN IMAGE-LEVEL", final_train)
print_metrics("FINAL VAL IMAGE-LEVEL", final_val)
print_metrics("FINAL TEST IMAGE-LEVEL", final_test)


# ============================================================
# Save outputs
# ============================================================

save_predictions(final_train, OUT_DIR / "train_predictions.csv")
save_predictions(final_val, OUT_DIR / "val_predictions.csv")
save_predictions(final_test, OUT_DIR / "test_predictions.csv")

for split_name, m in [("train", final_train), ("val", final_val), ("test", final_test)]:
    pd.DataFrame(
        m["confusion_matrix"],
        index=[f"true_{c}" for c in CLASSES],
        columns=[f"pred_{c}" for c in CLASSES],
    ).to_csv(OUT_DIR / f"{split_name}_confusion_matrix.csv")

    pd.DataFrame(
        classification_report(
            m["labels"],
            m["preds"],
            labels=[0, 1, 2],
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        )
    ).transpose().to_csv(OUT_DIR / f"{split_name}_classification_report.csv")


# ============================================================
# Membership inference
# ============================================================

summary = {}

def score_entropy(probs):
    ent = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
    return -ent

def score_true_conf(probs, labels):
    return probs[np.arange(len(labels)), labels]

def score_neg_loss(probs, labels):
    return np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0))

def score_max_conf(probs):
    return np.max(probs, axis=1)

def score_correct(probs, labels):
    return (np.argmax(probs, axis=1) == labels).astype(float)


if RUN_MIA:
    rows = []
    for split_name, member_label, m in [
        ("train_member", 1, final_train),
        ("test_nonmember", 0, final_test),
    ]:
        probs = m["probs"]
        labels = m["labels"]
        paths = m["paths"]
        preds = np.argmax(probs, axis=1)

        scores = {
            "score_max_confidence": score_max_conf(probs),
            "score_true_class_confidence": score_true_conf(probs, labels),
            "score_negative_entropy": score_entropy(probs),
            "score_negative_loss": score_neg_loss(probs, labels),
            "score_correctness": score_correct(probs, labels),
        }

        for i in range(len(labels)):
            r = {
                "split": split_name,
                "membership": member_label,
                "path": paths[i],
                "subject_id": parse_subject_id(paths[i]),
                "true_idx": int(labels[i]),
                "true_label": CLASSES[int(labels[i])],
                "pred_idx": int(preds[i]),
                "pred_label": CLASSES[int(preds[i])],
                "prob_AD": float(probs[i, 0]),
                "prob_CN": float(probs[i, 1]),
                "prob_MCI": float(probs[i, 2]),
            }
            for k, arr in scores.items():
                r[k] = float(arr[i])
            rows.append(r)

    scores_df = pd.DataFrame(rows)
    scores_df.to_csv(MIA_DIR / "membership_inference_scores.csv", index=False)

    y_member = scores_df["membership"].values.astype(int)
    summary_rows = []
    roc_rows = []
    score_cols = [
        "score_max_confidence",
        "score_true_class_confidence",
        "score_negative_entropy",
        "score_negative_loss",
        "score_correctness",
    ]

    for col in score_cols:
        s = scores_df[col].values.astype(float)
        auc = roc_auc_score(y_member, s)
        ap = average_precision_score(y_member, s)
        fpr, tpr, th = roc_curve(y_member, s)
        j = tpr - fpr
        bi = int(np.argmax(j))
        pred_member = (s >= th[bi]).astype(int)

        summary_rows.append({
            "attack_score": col,
            "mia_auc": auc,
            "mia_average_precision": ap,
            "best_threshold": float(th[bi]),
            "best_threshold_attack_accuracy": accuracy_score(y_member, pred_member),
            "best_threshold_tpr": float(tpr[bi]),
            "best_threshold_fpr": float(fpr[bi]),
            "num_members": int((y_member == 1).sum()),
            "num_nonmembers": int((y_member == 0).sum()),
            "member_score_mean": float(s[y_member == 1].mean()),
            "nonmember_score_mean": float(s[y_member == 0].mean()),
            "member_score_std": float(s[y_member == 1].std()),
            "nonmember_score_std": float(s[y_member == 0].std()),
        })

        for fp, tp, thr in zip(fpr, tpr, th):
            roc_rows.append({"attack_score": col, "fpr": float(fp), "tpr": float(tp), "threshold": float(thr)})

    mia_df = pd.DataFrame(summary_rows)
    roc_df = pd.DataFrame(roc_rows)
    mia_df.to_csv(MIA_DIR / "membership_inference_summary.csv", index=False)
    roc_df.to_csv(MIA_DIR / "membership_inference_roc_points.csv", index=False)

    best_mia = mia_df.sort_values("mia_auc", ascending=False).iloc[0]

    txt = f"""Membership Inference Evaluation

Dataset: {DATASET_NAME}
Mode: {MODE}
Backbone: {BACKBONE}
Selected model: {selected_path}
Final eval mode: {final_eval_mode}

Best attack score: {best_mia['attack_score']}
MIA AUC: {best_mia['mia_auc']:.6f}
Average precision: {best_mia['mia_average_precision']:.6f}
Best threshold attack accuracy: {best_mia['best_threshold_attack_accuracy']:.6f}
TPR: {best_mia['best_threshold_tpr']:.6f}
FPR: {best_mia['best_threshold_fpr']:.6f}
"""
    (MIA_DIR / "membership_inference_result.txt").write_text(txt, encoding="utf-8")

    summary["mia_best_attack_score"] = best_mia["attack_score"]
    summary["mia_best_auc"] = best_mia["mia_auc"]
    summary["mia_best_average_precision"] = best_mia["mia_average_precision"]
    summary["mia_best_threshold_attack_accuracy"] = best_mia["best_threshold_attack_accuracy"]
    summary["mia_best_threshold_tpr"] = best_mia["best_threshold_tpr"]
    summary["mia_best_threshold_fpr"] = best_mia["best_threshold_fpr"]

    print("\nMEMBERSHIP INFERENCE SUMMARY")
    print(mia_df.to_string(index=False))


# ============================================================
# Final metrics CSV
# ============================================================

eps_min_final, eps_mean_final, eps_max_final = legacy_epsilon_values(ROUNDS)

checks = {
    "train_image_accuracy_ge_95": final_train["accuracy"] >= 0.95,
    "val_image_accuracy_ge_85": final_val["accuracy"] >= 0.85,
    "test_image_accuracy_ge_85": final_test["accuracy"] >= 0.85,
    "val_close_to_train_gap_le_10pct": abs(final_train["accuracy"] - final_val["accuracy"]) <= 0.10,
    "test_close_to_val_gap_le_10pct": abs(final_val["accuracy"] - final_test["accuracy"]) <= 0.10,
}

final_summary = {
    "dataset": DATASET_NAME,
    "mode": MODE,
    "backbone": BACKBONE,
    "num_clients": NUM_CLIENTS,
    "rounds": ROUNDS,
    "local_epochs": LOCAL_EPOCHS,
    "dp": IS_DP,
    "roi_dp": IS_ROI,
    "secure_aggregation": USE_SECURE_AGG,
    "privacy_accounting": "legacy_debug_not_formal_dp" if IS_DP else "none",
    "privacy_accounting_status": "legacy_debug_not_formal_dp" if IS_DP else "not_applicable",
    "target_epsilon": TARGET_EPSILON if IS_DP else np.nan,
    "epsilon_min": eps_min_final,
    "epsilon_mean": eps_mean_final,
    "epsilon_max": eps_max_final,
    "delta": DELTA if IS_DP else np.nan,
    "uniform_noise_multiplier": UNIFORM_NOISE if MODE == "fed_dp" else np.nan,
    "roi_noise_multiplier": ROI_NOISE if IS_ROI else np.nan,
    "max_grad_norm": MAX_GRAD_NORM if IS_DP else np.nan,
    "selected_model_type": selected_type,
    "selected_model_path": str(selected_path),
    "best_val_round": best_val_round,
    "best_target_round": best_target_round,
    "final_eval_mode": final_eval_mode,
    "total_time_min": total_time / 60.0,

    "train_accuracy": final_train["accuracy"],
    "train_balanced_accuracy": final_train["balanced_accuracy"],
    "train_precision_macro": final_train["precision_macro"],
    "train_recall_macro": final_train["recall_macro"],
    "train_f1_macro": final_train["f1_macro"],
    "train_precision_weighted": final_train["precision_weighted"],
    "train_recall_weighted": final_train["recall_weighted"],
    "train_f1_weighted": final_train["f1_weighted"],
    "train_auc_macro_ovr": final_train["auc_macro_ovr"],
    "train_auc_weighted_ovr": final_train["auc_weighted_ovr"],
    "train_auc_AD_ovr": final_train["auc_AD_ovr"],
    "train_auc_CN_ovr": final_train["auc_CN_ovr"],
    "train_auc_MCI_ovr": final_train["auc_MCI_ovr"],

    "val_accuracy": final_val["accuracy"],
    "val_balanced_accuracy": final_val["balanced_accuracy"],
    "val_precision_macro": final_val["precision_macro"],
    "val_recall_macro": final_val["recall_macro"],
    "val_f1_macro": final_val["f1_macro"],
    "val_precision_weighted": final_val["precision_weighted"],
    "val_recall_weighted": final_val["recall_weighted"],
    "val_f1_weighted": final_val["f1_weighted"],
    "val_auc_macro_ovr": final_val["auc_macro_ovr"],
    "val_auc_weighted_ovr": final_val["auc_weighted_ovr"],
    "val_auc_AD_ovr": final_val["auc_AD_ovr"],
    "val_auc_CN_ovr": final_val["auc_CN_ovr"],
    "val_auc_MCI_ovr": final_val["auc_MCI_ovr"],

    "test_accuracy": final_test["accuracy"],
    "test_balanced_accuracy": final_test["balanced_accuracy"],
    "test_precision_macro": final_test["precision_macro"],
    "test_recall_macro": final_test["recall_macro"],
    "test_f1_macro": final_test["f1_macro"],
    "test_precision_weighted": final_test["precision_weighted"],
    "test_recall_weighted": final_test["recall_weighted"],
    "test_f1_weighted": final_test["f1_weighted"],
    "test_auc_macro_ovr": final_test["auc_macro_ovr"],
    "test_auc_weighted_ovr": final_test["auc_weighted_ovr"],
    "test_auc_AD_ovr": final_test["auc_AD_ovr"],
    "test_auc_CN_ovr": final_test["auc_CN_ovr"],
    "test_auc_MCI_ovr": final_test["auc_MCI_ovr"],

    **checks,
    **summary,
}

pd.DataFrame([final_summary]).to_csv(FINAL_METRICS_PATH, index=False)

# experiment_summary.json — compact cross-run comparison file.
# Contains only the key config fields and results needed to compare matched experiments.
_effective_scope = args.dp_train_scope if (IS_DP or IS_MATCHED_SCOPE) else "full"
_exp_summary = {
    "dataset": DATASET_NAME,
    "mode": MODE,
    "backbone": BACKBONE,
    "effective_train_scope": _effective_scope,
    "use_cbam": args.use_cbam,
    "use_mixup": args.use_mixup if (IS_DP or IS_MATCHED_SCOPE) else False,
    "random_erasing_p": ERASING_P,
    "lr_profile": LR_PROFILE_LABEL,
    "head_lr": HEAD_LR,
    "backbone_lr": BACKBONE_LR,
    "is_dp": IS_DP,
    "is_roi": IS_ROI,
    "is_matched_scope": IS_MATCHED_SCOPE,
    "noise_multiplier": (UNIFORM_NOISE if MODE == "fed_dp" else
                         ROI_NOISE if IS_ROI else None),
    "noise_type": ("uniform" if MODE == "fed_dp" else
                   "roi_weighted" if IS_ROI else "none"),
    "rounds": ROUNDS,
    "local_epochs": LOCAL_EPOCHS,
    "clients": NUM_CLIENTS,
    "batch_size": BATCH_SIZE,
    "seed": SEED,
    "trainable_params": TRAINABLE_PARAMS,
    "total_params": TOTAL_PARAMS,
    "trainable_params_pct": round(100.0 * TRAINABLE_PARAMS / TOTAL_PARAMS, 2),
    "privacy_accounting_status": "legacy_debug_not_formal_dp" if IS_DP else "not_applicable",
    "test_accuracy": float(final_test["accuracy"]),
    "test_auc_macro_ovr": float(final_test["auc_macro_ovr"]),
    "val_accuracy": float(final_val["accuracy"]),
    "val_auc_macro_ovr": float(final_val["auc_macro_ovr"]),
    "best_val_round": best_val_round,
    "final_eval_mode": final_eval_mode,
    "final_eval_mode_flag": args.final_eval_mode,
    "warnings": STARTUP_WARNINGS,
}
_exp_summary_path = OUT_DIR / "experiment_summary.json"
with open(_exp_summary_path, "w") as _f:
    json.dump(_exp_summary, _f, indent=2)
print("Experiment summary JSON:", _exp_summary_path)

print("\nREQUIREMENT CHECKS")
for k, v in checks.items():
    print(k, ":", v)

zip_path = None
if SAVE_ZIP:
    zip_path = shutil.make_archive(str(OUT_DIR), "zip", root_dir=OUT_DIR)

print("\n============================================================")
print("SAVED FILES")
print("============================================================")
print("Output folder:", OUT_DIR)
print("Training log:", TRAIN_LOG_PATH)
if IS_DP:
    print("Privacy log:", PRIVACY_LOG_PATH)
print("Final metrics:", FINAL_METRICS_PATH)
print("Best val model:", BEST_VAL_MODEL_PATH)
print("Best target model:", BEST_TARGET_MODEL_PATH if BEST_TARGET_MODEL_PATH.exists() else "not created")
if RUN_MIA:
    print("MIA summary:", MIA_DIR / "membership_inference_summary.csv")
if zip_path:
    print("ZIP:", zip_path)
print("DONE.")
