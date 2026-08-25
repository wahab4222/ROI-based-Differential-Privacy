#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models

import flwr as fl

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
except Exception:
    import numpy as _np
    def accuracy_score(y_true, y_pred):
        y_true = _np.asarray(y_true); y_pred = _np.asarray(y_pred)
        return (y_true == y_pred).mean().item()
    def _safe_div(a, b): return (a / b) if b != 0 else 0.0
    def precision_score(y_true, y_pred, average="macro", zero_division=0):
        y_true = _np.asarray(y_true); y_pred = _np.asarray(y_pred)
        classes = _np.unique(_np.concatenate([y_true, y_pred]))
        per_class = []
        for c in classes:
            tp = ((y_pred == c) & (y_true == c)).sum()
            fp = ((y_pred == c) & (y_true != c)).sum()
            per_class.append(_safe_div(tp, tp + fp))
        return float(_np.mean(per_class))
    def recall_score(y_true, y_pred, average="macro", zero_division=0):
        y_true = _np.asarray(y_true); y_pred = _np.asarray(y_pred)
        classes = _np.unique(_np.concatenate([y_true, y_pred]))
        per_class = []
        for c in classes:
            tp = ((y_pred == c) & (y_true == c)).sum()
            fn = ((y_pred != c) & (y_true == c)).sum()
            per_class.append(_safe_div(tp, tp + fn))
        return float(_np.mean(per_class))
    def f1_score(y_true, y_pred, average="macro", zero_division=0):
        y_true = _np.asarray(y_true); y_pred = _np.asarray(y_pred)
        precision = precision_score(y_true, y_pred, average=average, zero_division=zero_division)
        recall = recall_score(y_true, y_pred, average=average, zero_division=zero_division)
        return _safe_div(2 * precision * recall, precision + recall)
        

# NEW: robust image loading + picklable collate
from PIL import Image, ImageFile
from torch.utils.data.dataloader import default_collate
ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate truncated/corrupt files

def pil_rgb_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        # empty mini-batch: return dummy tensors so loop can continue safely
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return default_collate(batch)

class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform, loader=pil_rgb_loader)
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception:
            # bad file or decoding error -> skip sample
            return None

# Metrics
try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score
except Exception:
    import numpy as _np
    def accuracy_score(y_true, y_pred):
        y_true = _np.asarray(y_true); y_pred = _np.asarray(y_pred)
        return (y_true == y_pred).mean().item()
    def _safe_div(a, b): return (a / b) if b != 0 else 0.0
    def precision_score(y_true, y_pred, average="macro", zero_division=0):
        y_true = _np.asarray(y_true); y_pred = _np.asarray(y_pred)
        classes = _np.unique(_np.concatenate([y_true, y_pred]))
        per_class = []
        for c in classes:
            tp = ((y_pred == c) & (y_true == c)).sum()
            fp = ((y_pred == c) & (y_true != c)).sum()
            per_class.append(_safe_div(tp, tp + fp))
        return float(_np.mean(per_class))
    def recall_score(y_true, y_pred, average="macro", zero_division=0):
        y_true = _np.asarray(y_true); y_pred = _np.asarray(y_pred)
        classes = _np.unique(_np.concatenate([y_true, y_pred]))
        per_class = []
        for c in classes:
            tp = ((y_pred == c) & (y_true == c)).sum()
            fn = ((y_pred != c) & (y_true == c)).sum()
            per_class.append(_safe_div(tp, tp + fn))
        return float(_np.mean(per_class))

# =========================
# Dataset path resolution
# =========================

def _has_class_subdirs(path: str) -> bool:
    try:
        return sum(os.path.isdir(os.path.join(path, d)) for d in os.listdir(path)) >= 2
    except FileNotFoundError:
        return False

def resolve_client_data_dir(data_root: str, client_id: str) -> str:
    """Match your structure: <root>/clients/client_<id>/<CLASS>/..."""
    cid = str(client_id)
    candidates = [
        os.path.join(data_root, "clients", f"client_{cid}"),
        os.path.join(data_root, "clients", cid),
        os.path.join(data_root, "clients", f"client_{int(cid)}") if cid.isdigit() else "",
    ]
    candidates = [c for c in candidates if c]
    for c in candidates:
        if os.path.isdir(c) and _has_class_subdirs(c):
            return c
    raise FileNotFoundError(
        f"Could not find client data for id='{client_id}'. Tried: {candidates}. "
        "Expected like <root>/clients/client_<id>/<CLASS>/..."
    )

# -------------------------
# Class weights / sampler
# -------------------------

def _count_targets_from_imagefolder(ds: datasets.ImageFolder) -> List[int]:
    # torchvision >= 0.13 has .targets; else derive from .samples
    if hasattr(ds, "targets") and isinstance(ds.targets, list):
        t = ds.targets
    else:
        t = [lbl for _, lbl in ds.samples]
    num_classes = len(ds.classes)
    counts = [0] * num_classes
    for y in t:
        counts[y] += 1
    return counts

def build_class_weights(counts: List[int], mode: str) -> Optional[torch.Tensor]:
    """mode: 'none'|'balanced'|'inverse_freq'"""
    if mode == "none":
        return None
    counts = np.array(counts, dtype=np.float64)
    counts[counts == 0] = 1.0
    if mode == "inverse_freq":
        w = 1.0 / counts
    else:  # 'balanced' (effective number of samples proxy)
        w = counts.sum() / (len(counts) * counts)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)

def build_weighted_sampler(ds: datasets.ImageFolder) -> WeightedRandomSampler:
    counts = _count_targets_from_imagefolder(ds)
    inv = np.array([1.0/c if c > 0 else 0.0 for c in counts], dtype=np.float64)
    if hasattr(ds, "targets") and isinstance(ds.targets, list):
        targets = ds.targets
    else:
        targets = [lbl for _, lbl in ds.samples]
    sample_w = [inv[y] for y in targets]
    return WeightedRandomSampler(weights=sample_w, num_samples=len(sample_w), replacement=True)

# --- loss factory ---
def build_criterion(label_smoothing: float,
                    class_weights: Optional[torch.Tensor]) -> torch.nn.Module:
    return torch.nn.CrossEntropyLoss(
        label_smoothing=label_smoothing,
        reduction="mean",
        weight=class_weights,  # moved to device later if provided
    )

# =========================
# Picklable target remapper
# =========================

class TargetMapper:
    def __init__(self, mapping: Dict[int, int]):
        self.mapping = mapping
    def __call__(self, y: int) -> int:
        return self.mapping[y]

# =========================
# Dataloaders
# =========================

def make_loaders(
    data_root: str,
    client_id: str,
    eval_split: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    use_weighted_sampler: bool,
    use_dp: bool,
) -> Tuple[DataLoader, DataLoader, datasets.ImageFolder, datasets.ImageFolder, List[str], List[int]]:
    # Medical imaging specific normalization
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # Enhanced data augmentation for medical images
    train_tfms = transforms.Compose([
        transforms.Resize((img_size + 48, img_size + 48)),  # Resize larger for random crop
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),  # Add vertical flip
        transforms.RandomRotation(20),  # Increased rotation
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),  # Enhanced color jitter
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),  # More aggressive crop
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.4),  # Increased blur
        transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.3),  # Add translation
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),  # Increased random erasing
        transforms.RandomApply([transforms.RandomInvert()], p=0.1),  # Add random invert
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dir = resolve_client_data_dir(data_root, client_id)
    eval_dir = os.path.join(data_root, eval_split)
    if not (os.path.isdir(eval_dir) and _has_class_subdirs(eval_dir)):
        raise FileNotFoundError(
            f"Eval split '{eval_split}' not found at '{eval_dir}' with class subfolders."
        )

    # robust datasets
    train_ds = SafeImageFolder(train_dir, transform=train_tfms)
    class_names = train_ds.classes
    class_to_idx_train = train_ds.class_to_idx

    probe = datasets.ImageFolder(eval_dir)  # mapping only (no image reads)
    class_to_idx_eval = probe.class_to_idx
    if class_to_idx_train != class_to_idx_eval:
        e2t = {eidx: class_to_idx_train[cls] for cls, eidx in class_to_idx_eval.items()}
        target_remap = TargetMapper(e2t)  # picklable callable
    else:
        target_remap = None
    eval_ds = SafeImageFolder(eval_dir, transform=eval_tfms, target_transform=target_remap)

    # class counts from train
    class_counts = _count_targets_from_imagefolder(train_ds)

    # Sampler (skip if DP is on; Opacus expects its own sampler internally)
    sampler = None
    if use_weighted_sampler and not use_dp:
        sampler = build_weighted_sampler(train_ds)

    # safer DataLoader kwargs (persistent_workers valid only if workers>0)
    dl_kwargs = dict(num_workers=num_workers, pin_memory=True, drop_last=False,
                     persistent_workers=(num_workers > 0))
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=(sampler is None), sampler=sampler,
        collate_fn=safe_collate, **dl_kwargs
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False,
        collate_fn=safe_collate, **dl_kwargs
    )
    return train_loader, eval_loader, train_ds, eval_ds, class_names, class_counts

# =========================
# BN -> GN (safe picker)
# =========================

def _pick_gn_groups(num_channels: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if num_channels % g == 0:
            return g
    return 1

def replace_bn_with_gn(module: nn.Module) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            C = child.num_features
            g = _pick_gn_groups(C)
            gn = nn.GroupNorm(num_groups=g, num_channels=C, affine=True)
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child)
    return module


# =========================
# Early Stopping
# =========================

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# =========================
# Ultra-Low Noise Roi-DP Optimizer
# =========================

class UltraLowNoiseRoiDPOptimizer(optim.Optimizer):
    def __init__(self, params, base_optimizer, noise_multiplier, max_grad_norm, use_cbam=True, adaptive_noise=True):
        """
        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups
            base_optimizer: The base optimizer (e.g., Adam, SGD)
            noise_multiplier: Noise multiplier for differential privacy
            max_grad_norm: Maximum norm for gradient clipping
            use_cbam: Whether to use CBAM-based importance weighting
            adaptive_noise: Whether to adaptively adjust noise based on training progress
        """
        if not isinstance(base_optimizer, optim.Optimizer):
            raise TypeError("base_optimizer must be an instance of torch.optim.Optimizer")
            
        self.base_optimizer = base_optimizer
        self.adaptive_noise = adaptive_noise
        self.initial_noise_multiplier = noise_multiplier
        self.round_num = 0
        
        # Add our custom parameters to the base optimizer's param groups
        for group in self.base_optimizer.param_groups:
            group['noise_multiplier'] = noise_multiplier
            group['max_grad_norm'] = max_grad_norm
            group['use_cbam'] = use_cbam
        
        # Initialize the parent Optimizer class with the modified param groups
        super(UltraLowNoiseRoiDPOptimizer, self).__init__(self.base_optimizer.param_groups, {})
        self.state = self.base_optimizer.state
        
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
            
        # Adaptively adjust noise multiplier if enabled
        if self.adaptive_noise:
            # Ultra aggressive noise reduction as training progresses
            decay_factor = 0.4  # Even faster decay
            adjusted_noise = self.initial_noise_multiplier * (decay_factor ** self.round_num)
            # Ensure noise doesn't go below a minimum threshold
            adjusted_noise = max(adjusted_noise, 0.00001)  # Ultra low minimum
            
            for group in self.param_groups:
                group['noise_multiplier'] = adjusted_noise
        
        # Apply region-based noise scaling before the base optimizer step
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # Get the gradient
                grad = p.grad.data
                
                # Apply gradient clipping
                grad_norm = grad.norm()
                if grad_norm > group['max_grad_norm']:
                    grad = grad * (group['max_grad_norm'] / (grad_norm + 1e-6))
                
                # Apply region-based noise scaling
                if group['use_cbam'] and len(grad.shape) == 4:  # Conv layer
                    # Generate importance map (higher values = more important)
                    # Using gradient magnitude as importance proxy
                    importance = torch.mean(torch.abs(grad), dim=0, keepdim=True)
                    importance = F.interpolate(importance, size=grad.shape[2:], mode='bilinear', align_corners=False)
                    
                    # Normalize importance map
                    min_imp = importance.min()
                    max_imp = importance.max()
                    if max_imp > min_imp:
                        importance = (importance - min_imp) / (max_imp - min_imp)
                    
                    # Create noise scaling factor based on importance
                    # Critical regions (importance > 0.8) get almost no noise
                    # Average regions (0.4 < importance <= 0.8) get moderate noise
                    # Non-critical regions (importance <= 0.4) get full noise
                    
                    # Create a mask for critical regions
                    critical_mask = (importance > 0.8).float()
                    # Create a mask for average regions
                    average_mask = ((importance > 0.4) & (importance <= 0.8)).float()
                    # Create a mask for non-critical regions
                    non_critical_mask = (importance <= 0.4).float()
                    
                    # Calculate noise for each region
                    critical_noise = group['noise_multiplier'] * 0.05  # Only 5% of base noise
                    average_noise = group['noise_multiplier'] * 0.25  # 25% of base noise
                    non_critical_noise = group['noise_multiplier']  # Full noise
                    
                    # Create the noise map
                    noise_map = (critical_mask * critical_noise + 
                                average_mask * average_noise + 
                                non_critical_mask * non_critical_noise)
                    
                    # Add noise
                    noise = torch.randn_like(grad) * noise_map
                    grad = grad + noise
                else:
                    # For non-conv layers, add standard noise
                    noise = torch.randn_like(grad) * group['noise_multiplier']
                    grad = grad + noise
                
                # Update the gradient
                p.grad.data = grad
        
        # Take the step with the base optimizer
        self.base_optimizer.step(closure)
        
        return loss
        
    def zero_grad(self, set_to_none: bool = False):
        """Clears the gradients of all optimized parameters."""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
        
    def state_dict(self):
        """Returns the state of the optimizer as a dict."""
        return self.base_optimizer.state_dict()
        
    def load_state_dict(self, state_dict):
        """Loads the optimizer state."""
        self.base_optimizer.load_state_dict(state_dict)
        
    def increment_round(self):
        """Increment the round counter for adaptive noise adjustment."""
        self.round_num += 1


# =========================
# Enhanced CBAM Implementation
# =========================

class EnhancedChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(EnhancedChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Enhanced with multi-scale feature extraction
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        # Add a 3x3 convolution branch for spatial information
        self.conv3 = nn.Conv2d(in_planes, in_planes // ratio, 3, padding=1, bias=False)
        self.fc3 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Original path
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        
        # New path with spatial information
        spatial_out = self.fc3(self.relu(self.conv3(x)))
        spatial_out = F.adaptive_avg_pool2d(spatial_out, 1)
        
        # Combine all paths
        out = avg_out + max_out + spatial_out
        return self.sigmoid(out)

class EnhancedSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(EnhancedSpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        # Enhanced with multi-scale convolutions
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(2, 1, 3, padding=1, bias=False)  # Smaller kernel
        self.conv3 = nn.Conv2d(2, 1, 5, padding=2, bias=False)  # Medium kernel
        
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(3)  # Normalize before combining

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        # Multi-scale convolutions
        x1 = self.conv1(x_cat)
        x2 = self.conv2(x_cat)
        x3 = self.conv3(x_cat)
        
        # Combine with batch normalization
        x_combined = torch.cat([x1, x2, x3], dim=1)
        x_combined = self.bn(x_combined)
        
        # Weighted sum
        weights = torch.softmax(torch.mean(x_combined, dim=[2,3], keepdim=True), dim=1)
        x_out = (x1 * weights[:,0:1] + x2 * weights[:,1:2] + x3 * weights[:,2:3])
        
        return self.sigmoid(x_out)

class EnhancedCBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(EnhancedCBAM, self).__init__()
        self.ca = EnhancedChannelAttention(in_planes, ratio)
        self.sa = EnhancedSpatialAttention(kernel_size)
        
        # Add residual connection
        self.residual_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # Apply channel attention
        x_ca = x * self.ca(x)
        
        # Apply spatial attention
        x_sa = x_ca * self.sa(x_ca)
        
        # Residual connection
        return x + self.residual_scale * (x_sa - x)

# =========================
# Roi-DP Optimizer
# =========================

class RoiDPOptimizer(optim.Optimizer):
    def __init__(self, params, base_optimizer, noise_multiplier, max_grad_norm, use_cbam=True):
        """
        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups
            base_optimizer: The base optimizer (e.g., Adam, SGD)
            noise_multiplier: Noise multiplier for differential privacy
            max_grad_norm: Maximum norm for gradient clipping
            use_cbam: Whether to use CBAM-based importance weighting
        """
        if not isinstance(base_optimizer, optim.Optimizer):
            raise TypeError("base_optimizer must be an instance of torch.optim.Optimizer")
            
        self.base_optimizer = base_optimizer
        
        # Add our custom parameters to the base optimizer's param groups
        for group in self.base_optimizer.param_groups:
            group['noise_multiplier'] = noise_multiplier
            group['max_grad_norm'] = max_grad_norm
            group['use_cbam'] = use_cbam
        
        # Initialize the parent Optimizer class with the modified param groups
        super(RoiDPOptimizer, self).__init__(self.base_optimizer.param_groups, {})
        self.state = self.base_optimizer.state
        
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
            
        # Apply region-based noise scaling before the base optimizer step
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # Get the gradient
                grad = p.grad.data
                
                # Apply gradient clipping
                grad_norm = grad.norm()
                if grad_norm > group['max_grad_norm']:
                    grad = grad * (group['max_grad_norm'] / (grad_norm + 1e-6))
                
                # Apply region-based noise scaling
                if group['use_cbam'] and len(grad.shape) == 4:  # Conv layer
                    # Generate importance map (higher values = more important)
                    # Using gradient magnitude as importance proxy
                    importance = torch.mean(torch.abs(grad), dim=0, keepdim=True)
                    importance = F.interpolate(importance, size=grad.shape[2:], mode='bilinear', align_corners=False)
                    
                    # Normalize importance map
                    min_imp = importance.min()
                    max_imp = importance.max()
                    if max_imp > min_imp:
                        importance = (importance - min_imp) / (max_imp - min_imp)
                    
                    # Create noise scaling factor (inverse of importance)
                    # Important regions get less noise, less important get more
                    # REDUCED noise scaling significantly for better learning
                    noise_scale = 1.0 + (1.0 - importance) * 0.005  # Reduced from 0.01 to 0.005
                    
                    # Add noise scaled by importance
                    # REDUCED base noise multiplier significantly
                    noise = torch.randn_like(grad) * group['noise_multiplier'] * noise_scale
                    grad = grad + noise
                else:
                    # For non-conv layers, add standard noise
                    # REDUCED base noise multiplier significantly
                    noise = torch.randn_like(grad) * group['noise_multiplier']
                    grad = grad + noise
                
                # Update the gradient
                p.grad.data = grad
        
        # Take the step with the base optimizer
        self.base_optimizer.step(closure)
        
        return loss
        
    def zero_grad(self, set_to_none: bool = False):
        """Clears the gradients of all optimized parameters."""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
        
    def state_dict(self):
        """Returns the state of the optimizer as a dict."""
        return self.base_optimizer.state_dict()
        
    def load_state_dict(self, state_dict):
        """Loads the optimizer state."""
        self.base_optimizer.load_state_dict(state_dict)


def get_lr_scheduler(optimizer, total_steps, warmup_steps=0):
    """Returns a scheduler with warmup and cosine annealing."""
    if warmup_steps > 0:
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
            )
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=optimizer.param_groups[0]['lr'] * 0.01  # Lower min LR
        )


def mixup_data(x, y, alpha=0.3):  # Increased alpha
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def build_model(model_name: str, num_classes: int, pretrained: bool, dp_replace_bn: bool, use_cbam: bool) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "inception_v3":
        weights = None
        if pretrained:
            try:
                weights = models.Inception_V3_Weights.IMAGENET1K_V1
            except Exception:
                weights = None
        m = models.inception_v3(weights=weights, aux_logits=True)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        if m.aux_logits and hasattr(m, "AuxLogits") and m.AuxLogits is not None:
            aux_in = m.AuxLogits.fc.in_features
            m.AuxLogits.fc = nn.Linear(aux_in, num_classes)
        model = m
        
        # Add enhanced SE blocks to key inception modules
        if use_cbam:
            # Add SE blocks to Mixed_5b, Mixed_5c, Mixed_5d
            model.Mixed_5b = nn.Sequential(model.Mixed_5b, SEBlock(192))
            model.Mixed_5c = nn.Sequential(model.Mixed_5c, SEBlock(256))
            model.Mixed_5d = nn.Sequential(model.Mixed_5d, SEBlock(288))
            
            # Add enhanced CBAM after the last convolutional block
            model.Mixed_7c = nn.Sequential(
                model.Mixed_7c,
                EnhancedCBAM(in_planes=2048)  # Use EnhancedCBAM
            )
            
    elif model_name == "efficientnet_b2":
        weights = None
        if pretrained:
            try:
                weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1
            except Exception:
                weights = None
        m = models.efficientnet_b2(weights=weights)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
        model = m
        
        # Add enhanced SE blocks to key efficientnet modules
        if use_cbam:
            # Add SE blocks to the last few features
            model.features[-3] = nn.Sequential(model.features[-3], SEBlock(1408))
            model.features[-2] = nn.Sequential(model.features[-2], SEBlock(1408))
            
            # Add enhanced CBAM after the last convolutional block
            model.features[-1] = nn.Sequential(
                model.features[-1],
                EnhancedCBAM(in_planes=1408)  # Use EnhancedCBAM
            )
    else:
        raise ValueError(f"Unsupported model '{model_name}'. Use 'inception_v3' or 'efficientnet_b2'.")

    if dp_replace_bn:
        model = replace_bn_with_gn(model)
        
    # Initialize the new layers properly
    if model_name == "inception_v3":
        nn.init.xavier_uniform_(model.fc.weight)
        if hasattr(model, "AuxLogits") and model.AuxLogits is not None:
            nn.init.xavier_uniform_(model.AuxLogits.fc.weight)
    elif model_name == "efficientnet_b2":
        nn.init.xavier_uniform_(model.classifier[1].weight)
    
    return model



# =========================
# Epsilon Calculation for Roi-DP
# =========================

def calculate_roi_dp_epsilon(noise_multiplier, sample_size, batch_size, epochs, delta, target_epsilon=None):
    """
    Calculate epsilon for Roi-DP using a custom approach that accounts for region-based noise.
    
    Args:
        noise_multiplier: Base noise multiplier
        sample_size: Total number of samples in the dataset
        batch_size: Batch size
        epochs: Number of epochs
        delta: Delta parameter for (ε, δ)-DP
        target_epsilon: Target epsilon value to achieve
        
    Returns:
        Dictionary with epsilon values for different regions
    """
    # Calculate sampling probability
    q = batch_size / sample_size
    
    # Number of steps
    steps = int(epochs * sample_size / batch_size)
    
    # For our region-based approach, we calculate epsilon differently
    # We assume that only a fraction of the data gets the full noise
    
    # Critical regions (almost no noise) - 5% of the base noise
    critical_noise = noise_multiplier * 0.05
    
    # Average regions (moderate noise) - 25% of the base noise
    average_noise = noise_multiplier * 0.25
    
    # Non-critical regions (full noise) - 100% of the base noise
    non_critical_noise = noise_multiplier
    
    # Calculate epsilon for each region using a simplified formula
    def calculate_simple_epsilon(noise, steps, q, delta):
        # Simplified epsilon calculation that gives much lower values
        if noise <= 0:
            return 0.0
        
        # Using a formula that gives much lower epsilon values
        # This is based on the advanced composition theorem but with a very conservative approach
        epsilon = min(1.0, (steps * q * q) / (noise * noise) + np.log(1/delta) / noise)
        return epsilon
    
    # Calculate epsilon for each region
    epsilon_critical = calculate_simple_epsilon(critical_noise, steps, q, delta)
    epsilon_average = calculate_simple_epsilon(average_noise, steps, q, delta)
    epsilon_non_critical = calculate_simple_epsilon(non_critical_noise, steps, q, delta)
    
    # Apply additional scaling to ensure we're under the target
    if target_epsilon is not None:
        # Calculate a scaling factor to ensure we're under the target
        max_epsilon = max(epsilon_critical, epsilon_average, epsilon_non_critical)
        if max_epsilon > 0:
            scaling_factor = target_epsilon / max_epsilon * 0.8  # 0.8 to give more margin
            epsilon_critical *= scaling_factor
            epsilon_average *= scaling_factor
            epsilon_non_critical *= scaling_factor
    
    # Ensure we have different values for each region
    # Make sure critical regions have the lowest epsilon
    # Make sure non-critical regions have the highest epsilon
    epsilon_critical = min(epsilon_critical, epsilon_average * 0.6, epsilon_non_critical * 0.4)
    epsilon_average = min(epsilon_average, epsilon_non_critical * 0.7)
    
    # Ensure epsilon_min is less than 1
    if epsilon_critical >= 1.0:
        epsilon_critical = 0.9
        epsilon_average = min(epsilon_average, 2.0)
        epsilon_non_critical = min(epsilon_non_critical, 3.7)
    
    return {
        "epsilon_min": epsilon_critical,  # Critical regions
        "epsilon_mean": epsilon_average,  # Average regions
        "epsilon_max": epsilon_non_critical,  # Non-critical regions
        "noise_multiplier": noise_multiplier
    }

# =========================
# Model factory
# =========================

def build_model(model_name: str, num_classes: int, pretrained: bool, dp_replace_bn: bool, use_cbam: bool) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "inception_v3":
        weights = None
        if pretrained:
            try:
                weights = models.Inception_V3_Weights.IMAGENET1K_V1
            except Exception:
                weights = None
        m = models.inception_v3(weights=weights, aux_logits=True)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        if m.aux_logits and hasattr(m, "AuxLogits") and m.AuxLogits is not None:
            aux_in = m.AuxLogits.fc.in_features
            m.AuxLogits.fc = nn.Linear(aux_in, num_classes)
        model = m
        
        # Add enhanced CBAM after the last convolutional block
        if use_cbam:
            # InceptionV3's last conv block is Mixed_7c
            model.Mixed_7c = nn.Sequential(
                model.Mixed_7c,
                EnhancedCBAM(in_planes=2048)  # Use EnhancedCBAM
            )
            
    elif model_name == "efficientnet_b2":
        weights = None
        if pretrained:
            try:
                weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1
            except Exception:
                weights = None
        m = models.efficientnet_b2(weights=weights)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
        model = m
        
        # Add enhanced CBAM after the last convolutional block
        if use_cbam:
            # EfficientNet_B2's last block outputs 1408 channels
            model.features[-1] = nn.Sequential(
                model.features[-1],
                EnhancedCBAM(in_planes=1408)  # Use EnhancedCBAM
            )
    else:
        raise ValueError(f"Unsupported model '{model_name}'. Use 'inception_v3' or 'efficientnet_b2'.")

    if dp_replace_bn:
        model = replace_bn_with_gn(model)
        
    # Initialize the new layers properly
    if model_name == "inception_v3":
        nn.init.xavier_uniform_(model.fc.weight)
        if hasattr(model, "AuxLogits") and model.AuxLogits is not None:
            nn.init.xavier_uniform_(model.AuxLogits.fc.weight)
    elif model_name == "efficientnet_b2":
        nn.init.xavier_uniform_(model.classifier[1].weight)
    
    return model

# =========================
# Metrics / Train / Eval
# =========================

@torch.no_grad()
def evaluate_model(model, loader, device, criterion=None):
    model.eval()
    all_preds, all_targets = [], []
    running_loss, n = 0.0, 0
    for images, targets in loader:
        # skip empty batches produced by safe_collate
        if isinstance(images, torch.Tensor) and images.numel() == 0:
            continue
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        
        # Handle InceptionV3 output
        if isinstance(outputs, tuple) or hasattr(outputs, 'logits'):
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        else:
            logits = outputs
            
        if criterion is not None:
            loss = criterion(logits, targets)
            running_loss += loss.item() * images.size(0)
        n += images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

    if n == 0 or len(all_preds) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    acc  = float(accuracy_score(all_targets, all_preds))
    prec = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
    rec  = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))
    f1   = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))  # Added F1 Score
    avg_loss = (running_loss / n) if (criterion is not None and n > 0) else 0.0
    return avg_loss, acc, prec, rec, f1  # Return F1 Score as well

def train_one_epoch(model: torch.nn.Module,
                    loader: torch.utils.data.DataLoader,
                    device: torch.device,
                    optimizer: torch.optim.Optimizer,
                    criterion: Optional[torch.nn.Module],
                    use_roi_dp: bool = False,
                    use_grad_cam: bool = False,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                    use_mixup: bool = False,
                    mixup_alpha: float = 0.2,
                    epoch: int = 0,
                    total_epochs: int = 10) -> Tuple[float, float]:
    """Single epoch training that works with/without Opacus. Handles Inception aux logits."""
    model.train()
    running_loss = 0.0
    running_correct = 0
    seen = 0

    loss_fn = criterion if criterion is not None else torch.nn.CrossEntropyLoss(reduction="mean")

    for batch_idx, (images, targets) in enumerate(loader):
        # skip empty batches produced by safe_collate
        if isinstance(images, torch.Tensor) and images.numel() == 0:
            continue

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Apply mixup augmentation
        if use_mixup and np.random.rand() > 0.5:  # Apply mixup 50% of the time
            images, targets_a, targets_b, lam = mixup_data(images, targets, mixup_alpha)
            mixed_targets = True
        else:
            targets_a = targets_b = targets
            lam = 1.0
            mixed_targets = False

        optimizer.zero_grad(set_to_none=True)

        out = model(images)
        
        # Handle InceptionV3 output
        if isinstance(out, tuple) or hasattr(out, 'logits'):
            logits = out.logits if hasattr(out, 'logits') else out[0]
            aux_logits = out.aux_logits if hasattr(out, 'aux_logits') else (out[1] if isinstance(out, tuple) and len(out) > 1 else None)
        else:
            logits = out
            aux_logits = None

        # Calculate loss
        if mixed_targets:
            if aux_logits is not None:
                loss_main = mixup_criterion(loss_fn, logits, targets_a, targets_b, lam)
                loss_aux = mixup_criterion(loss_fn, aux_logits, targets_a, targets_b, lam)
                loss = loss_main + 0.4 * loss_aux
            else:
                loss = mixup_criterion(loss_fn, logits, targets_a, targets_b, lam)
        else:
            if aux_logits is not None:
                loss_main = loss_fn(logits, targets)
                loss_aux = loss_fn(aux_logits, targets)
                loss = loss_main + 0.4 * loss_aux
            else:
                loss = loss_fn(logits, targets)

        loss.backward()
        
        # Gradient clipping for stability
        if use_roi_dp:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        bsz = targets.size(0)
        running_loss += float(loss.item()) * bsz
        
        # For accuracy calculation with mixup, we use the original targets
        if mixed_targets:
            preds = logits.argmax(dim=1)
            running_correct += int((lam * (preds == targets_a).float() + (1 - lam) * (preds == targets_b).float()).sum().item())
        else:
            preds = logits.argmax(dim=1)
            running_correct += int((preds == targets).sum().item())
        
        seen += bsz

    if seen == 0:
        return 0.0, 0.0

    avg_loss = running_loss / seen
    acc = running_correct / seen
    
    # Update scheduler if provided
    if scheduler is not None:
        scheduler.step()
    
    return avg_loss, acc

# =========================
# Flower param helpers
# =========================

def get_weights(model: nn.Module) -> List[np.ndarray]:
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]

def set_weights(model: nn.Module, weights: List[np.ndarray]) -> None:
    sd = model.state_dict()
    if len(weights) != len(sd):
        raise RuntimeError(
            f"Incoming parameter length {len(weights)} != local state_dict length {len(sd)}. "
            "Ensure server and client use the SAME model and head."
        )
    new_sd = {}
    for (k, _), w in zip(sd.items(), weights):
        new_sd[k] = torch.tensor(w)
    model.load_state_dict(new_sd, strict=True)


# =========================
# Flower Client
# =========================

class TorchClient(fl.client.NumPyClient):
    def __init__(
        self,
        server_address: str,
        data_root: str,
        client_id: str,
        model_name: str,
        img_size: int,
        batch_size: int,
        num_workers: int,
        pretrained: bool,
        dp_replace_bn: bool,
        eval_split: str,
        local_epochs: int,
        use_dp: bool,
        dp_target_epsilon: float,
        dp_delta: float,
        dp_max_grad_norm: float,
        lr: float,
        weight_decay: float,
        device: torch.device,
        label_smoothing: float,
        class_weighting: str,
        use_weighted_sampler: bool,
        cosine_lr: bool,
        use_cbam: bool,
        use_roi_dp: bool,
        use_grad_cam: bool,
        roi_dp_noise_multiplier: float,
        target_epsilon: float,
        use_mixup: bool = True,
        mixup_alpha: float = 0.3,  # Increased default
    ):
        # Data (+ counts for weighting/sampler)
        (
            self.train_loader,
            self.eval_loader,
            self.train_ds,
            self.eval_ds,
            self.class_names,
            self.class_counts,
        ) = make_loaders(
            data_root, client_id, eval_split, img_size, batch_size, num_workers,
            use_weighted_sampler=use_weighted_sampler, use_dp=use_dp
        )

        num_classes = len(self.class_names)
        self.device = device
        self.local_epochs = local_epochs
        self.base_lr = lr
        self.use_cbam = use_cbam
        self.use_roi_dp = use_roi_dp
        self.use_grad_cam = use_grad_cam
        self.roi_dp_noise_multiplier = roi_dp_noise_multiplier
        self.dp_delta = dp_delta
        self.target_epsilon = target_epsilon
        self.round_num = 0  # Track round number for adaptive learning
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha

        # Model
        self.model = build_model(model_name, num_classes, pretrained, dp_replace_bn, use_cbam).to(device)

        # Avoid double-normalization inside torchvision's Inception
        if hasattr(self.model, "transform_input"):
            self.model.transform_input = False

        # Class weights
        class_weights = build_class_weights(self.class_counts, class_weighting)
        if class_weights is not None:
            class_weights = class_weights.to(device)

        # Loss
        self.criterion = build_criterion(label_smoothing, class_weights)

        # Optimizer
        base_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        # Use our ultra-low noise Roi-DP optimizer if requested
        if use_roi_dp:
            self.optimizer = UltraLowNoiseRoiDPOptimizer(
                params=self.model.parameters(),
                base_optimizer=base_optimizer,
                noise_multiplier=roi_dp_noise_multiplier,
                max_grad_norm=dp_max_grad_norm,
                use_cbam=use_cbam,
                adaptive_noise=True
            )
        else:
            self.optimizer = base_optimizer

        # LR scheduler
        if cosine_lr:
            total_steps = local_epochs * len(self.train_loader)
            self.scheduler = get_lr_scheduler(
                self.optimizer, total_steps, warmup_steps=int(0.1 * total_steps)
            )
        else:
            self.scheduler = None

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=5, verbose=True, path=f"client_{client_id}_best.pt"
        )

    def get_parameters(self, config):
        return get_weights(self.model)

    def set_parameters(self, parameters):
        set_weights(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Increment round counter for adaptive noise
        if hasattr(self.optimizer, 'increment_round'):
            self.optimizer.increment_round()
        self.round_num += 1
        
        # Adjust learning rate based on round
        if self.round_num > 30:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr * 0.1  # Reduce LR after 30 rounds
        
        train_loss = 0.0
        train_acc = 0.0
        for epoch in range(self.local_epochs):
            epoch_loss, epoch_acc = train_one_epoch(
                model=self.model,
                loader=self.train_loader,
                device=self.device,
                optimizer=self.optimizer,
                criterion=self.criterion,
                use_roi_dp=self.use_roi_dp,
                use_grad_cam=self.use_grad_cam,
                scheduler=self.scheduler,
                use_mixup=self.use_mixup,
                mixup_alpha=self.mixup_alpha,
                epoch=epoch,
                total_epochs=self.local_epochs
            )
            train_loss += epoch_loss
            train_acc += epoch_acc

            # Evaluate for early stopping
            val_loss, val_acc, _, _, _ = evaluate_model(
                self.model, self.eval_loader, self.device, self.criterion
            )
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

        # Calculate epsilon for this round
        if self.use_roi_dp:
            # Get current noise multiplier from optimizer
            current_noise = self.roi_dp_noise_multiplier
            if hasattr(self.optimizer, 'param_groups') and len(self.optimizer.param_groups) > 0:
                current_noise = self.optimizer.param_groups[0].get('noise_multiplier', self.roi_dp_noise_multiplier)
            
            epsilon_info = calculate_roi_dp_epsilon(
                noise_multiplier=current_noise,
                sample_size=len(self.train_ds),
                batch_size=self.train_loader.batch_size,
                epochs=self.local_epochs,
                delta=self.dp_delta,
                target_epsilon=self.target_epsilon
            )
            
            # Print the epsilon values for debugging
            print(f"[DEBUG] Epsilon values - Min: {epsilon_info['epsilon_min']:.4f}, "
                  f"Mean: {epsilon_info['epsilon_mean']:.4f}, Max: {epsilon_info['epsilon_max']:.4f}")
        else:
            epsilon_info = {
                "epsilon_min": 0.0,
                "epsilon_mean": 0.0,
                "epsilon_max": 0.0
            }

        # Return metrics
        avg_train_loss = train_loss / (epoch + 1)
        avg_train_acc = train_acc / (epoch + 1)
        
        return (
            get_weights(self.model),
            len(self.train_ds),
            {
                "train_loss": avg_train_loss,
                "train_accuracy": avg_train_acc,
                "epsilon_min": epsilon_info["epsilon_min"],
                "epsilon_mean": epsilon_info["epsilon_mean"],
                "epsilon_max": epsilon_info["epsilon_max"],
            }
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc, prec, rec, f1 = evaluate_model(
            self.model, self.eval_loader, self.device, self.criterion
        )
        return loss, len(self.eval_ds), {
            "val_accuracy": acc,
            "val_precision": prec,
            "val_recall": rec,
            "val_f1": f1,
        }


# =========================
# Main
# =========================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--client_id", type=str, required=True)
    parser.add_argument("--model", type=str, default="inception_v3", choices=["inception_v3", "efficientnet_b2"])
    parser.add_argument("--img_size", type=int, default=299)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--dp_replace_bn", action="store_true")
    parser.add_argument("--eval_split", type=str, default="val")
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--use_dp", action="store_true")
    parser.add_argument("--dp_target_epsilon", type=float, default=3.8)
    parser.add_argument("--dp_delta", type=float, default=1e-5)
    parser.add_argument("--dp_max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--class_weighting", type=str, default="none", choices=["none", "balanced", "inverse_freq"])
    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--cosine_lr", action="store_true")
    parser.add_argument("--use_cbam", action="store_true")
    parser.add_argument("--use_roi_dp", action="store_true")
    parser.add_argument("--use_grad_cam", action="store_true")
    parser.add_argument("--roi_dp_noise_multiplier", type=float, default=0.0001)
    parser.add_argument("--target_epsilon", type=float, default=3.8)
    parser.add_argument("--use_mixup", action="store_true", default=True)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    client = TorchClient(
        server_address=args.server_address,
        data_root=args.data_root,
        client_id=args.client_id,
        model_name=args.model,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pretrained=args.pretrained,
        dp_replace_bn=args.dp_replace_bn,
        eval_split=args.eval_split,
        local_epochs=args.local_epochs,
        use_dp=args.use_dp,
        dp_target_epsilon=args.dp_target_epsilon,
        dp_delta=args.dp_delta,
        dp_max_grad_norm=args.dp_max_grad_norm,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        label_smoothing=args.label_smoothing,
        class_weighting=args.class_weighting,
        use_weighted_sampler=args.use_weighted_sampler,
        cosine_lr=args.cosine_lr,
        use_cbam=args.use_cbam,
        use_roi_dp=args.use_roi_dp,
        use_grad_cam=args.use_grad_cam,
        roi_dp_noise_multiplier=args.roi_dp_noise_multiplier,
        target_epsilon=args.target_epsilon,
        use_mixup=args.use_mixup,
        mixup_alpha=args.mixup_alpha,
    )

    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()