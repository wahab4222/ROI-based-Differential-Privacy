#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models

import flwr as fl
from flwr.common import ndarrays_to_parameters, Scalar, FitRes, Status, Code

# -----------------------------
# Utilities: weights <-> numpy
# -----------------------------
def get_weights(model: torch.nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

# -----------------------------------
# BN -> GN replacement (safe divisors)
# -----------------------------------
def _best_num_groups(num_channels: int) -> int:
    for g in [32, 16, 8, 4, 2, 1]:
        if num_channels % g == 0:
            return g
    return 1

def replace_bn_with_gn(module: nn.Module) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            num_channels = child.num_features
            g = _best_num_groups(num_channels)
            setattr(module, name, nn.GroupNorm(num_groups=g, num_channels=num_channels, eps=1e-5, affine=True))
        else:
            replace_bn_with_gn(child)
    return module


# =========================
# Enhanced Server Strategy
# =========================

class AdaptiveFedProxWithLogging(fl.server.strategy.FedProx):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_num = 0
        self.best_val_accuracy = 0.0
        self.patience = 5
        self.patience_counter = 0
        self.client_accuracies = {}  # Track client performance
        self.client_weights = {}  # Dynamic client weights based on performance
    
    def aggregate_fit(self, server_round, results, failures):
        self.round_num += 1
        
        # Update client performance tracking
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            if client_id not in self.client_accuracies:
                self.client_accuracies[client_id] = []
            
            # Extract accuracy from metrics if available
            if fit_res.metrics and "train_accuracy" in fit_res.metrics:
                self.client_accuracies[client_id].append(fit_res.metrics["train_accuracy"])
                
                # Calculate dynamic weight based on recent performance
                recent_acc = self.client_accuracies[client_id][-3:] if len(self.client_accuracies[client_id]) >= 3 else self.client_accuracies[client_id]
                avg_acc = sum(recent_acc) / len(recent_acc)
                # Higher accuracy gets higher weight (normalized)
                self.client_weights[client_id] = avg_acc
        
        # Normalize weights
        if self.client_weights:
            total_weight = sum(self.client_weights.values())
            for client_id in self.client_weights:
                self.client_weights[client_id] /= total_weight
        
        # Use original results for aggregation (avoiding FitRes creation issues)
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_metrics:
            print(f"\nINFO :      [ROUND {self.round_num} METRICS]")
            print(f"INFO :      Train Loss: {aggregated_metrics.get('train_loss', 'N/A'):.4f}")
            print(f"INFO :      Train Accuracy: {aggregated_metrics.get('train_accuracy', 'N/A'):.4f}")
            
            if 'epsilon_min' in aggregated_metrics:
                print(f"INFO :      Epsilon Min: {aggregated_metrics['epsilon_min']:.4f}")
                print(f"INFO :      Epsilon Mean: {aggregated_metrics['epsilon_mean']:.4f}")
                print(f"INFO :      Epsilon Max: {aggregated_metrics['epsilon_max']:.4f}")
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
    
        if aggregated_metrics:
            val_acc = aggregated_metrics.get('val_accuracy', 0.0)
            print(f"INFO :      Validation Accuracy: {val_acc:.4f}")
            print(f"INFO :      Validation Precision: {aggregated_metrics.get('val_precision', 'N/A'):.4f}")
            print(f"INFO :      Validation Recall: {aggregated_metrics.get('val_recall', 'N/A'):.4f}")
            print(f"INFO :      Validation F1 Score: {aggregated_metrics.get('val_f1', 'N/A'):.4f}")  # Added F1 Score
        
            # Track best validation accuracy
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.patience_counter = 0
                print(f"INFO :      New best validation accuracy: {val_acc:.4f}")
            else:
                self.patience_counter += 1
                print(f"INFO :      Patience counter: {self.patience_counter}/{self.patience}")
            
                # Early stopping if no improvement
                if self.patience_counter >= self.patience:
                    print(f"INFO :      Early stopping triggered after {self.patience} rounds without improvement")
    
        return aggregated_loss, aggregated_metrics


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

# -----------------------------
# Model factory (server side)
# -----------------------------
def build_model(model_name: str, num_classes: int, pretrained: bool, dp_replace_bn: bool, use_cbam: bool) -> nn.Module:
    if model_name == "inception_v3":
        weights = None
        if pretrained:
            try:
                weights = models.Inception_V3_Weights.IMAGENET1K_V1
            except Exception:
                weights = None
        model = models.inception_v3(weights=weights, aux_logits=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        if getattr(model, "aux_logits", False) and hasattr(model, "AuxLogits") and model.AuxLogits is not None:
            aux_in = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(aux_in, num_classes)
            
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
        m = torchvision.models.efficientnet_b2(weights=weights)
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
        raise ValueError(f"Unknown model: {model_name}")

    if dp_replace_bn:
        model = replace_bn_with_gn(model)

    return model

# ------------------------------------
# Eval/Fit metrics aggregation (Flower)
# NOTE: your Flower version passes ONLY one argument to these fns.
# ------------------------------------
def weighted_average_eval(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    if not metrics:
        return {}
    total = 0.0
    sums = {"val_accuracy": 0.0, "val_precision": 0.0, "val_recall": 0.0, "val_f1": 0.0}  # Added F1 Score
    for num_examples, m in metrics:
        n = float(num_examples)
        if n <= 0:
            continue
        total += n
        for k in ("val_accuracy", "val_precision", "val_recall", "val_f1"):  # Added F1 Score
            if k in m and m[k] is not None:
                sums[k] += n * float(m[k])
    out: Dict[str, Scalar] = {}
    if total > 0:
        for k in ("val_accuracy", "val_precision", "val_recall", "val_f1"):  # Added F1 Score
            out[k] = sums[k] / total
        out["num_examples"] = total
    return out

def weighted_average_fit(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    if not metrics:
        return {}
    total = 0.0
    sum_loss = 0.0
    sum_acc = 0.0
    eps_min_pool: List[float] = []
    eps_mean_pool: List[float] = []
    eps_max_pool: List[float] = []
    for num_examples, m in metrics:
        n = float(num_examples)
        if n <= 0:
            continue
        total += n
        if "train_loss" in m and m["train_loss"] is not None:
            sum_loss += n * float(m["train_loss"])
        if "train_accuracy" in m and m["train_accuracy"] is not None:
            sum_acc += n * float(m["train_accuracy"])
        for k in ("epsilon_min", "epsilon_mean", "epsilon_max"):
            if k in m and m[k] is not None:
                try:
                    if k == "epsilon_min":
                        eps_min_pool.append(float(m[k]))
                    elif k == "epsilon_mean":
                        eps_mean_pool.append(float(m[k]))
                    else:
                        eps_max_pool.append(float(m[k]))
                except Exception:
                    pass
    out: Dict[str, Scalar] = {}
    if total > 0:
        out["train_loss"] = sum_loss / total
        out["train_accuracy"] = sum_acc / total
    if eps_min_pool:
        out["epsilon_min"] = float(min(eps_min_pool))
    if eps_mean_pool:
        out["epsilon_mean"] = float(sum(eps_mean_pool) / len(eps_mean_pool))
    if eps_max_pool:
        out["epsilon_max"] = float(max(eps_max_pool))
    return out

# -----------------------------------------
# Infer #classes from eval split on server
# -----------------------------------------
def infer_num_classes(data_root: str, split: str = "val") -> int:
    split_dir = os.path.join(data_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    class_names = sorted(
        [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    )
    if not class_names:
        raise RuntimeError(f"No class folders found in {split_dir}")
    return len(class_names)

# -------------
# Arg parsing
# -------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--server_address", type=str, default="0.0.0.0:8005")
    p.add_argument("--model", type=str, default="inception_v3", choices=["inception_v3", "efficientnet_b2"])
    p.add_argument("--num_rounds", type=int, default=15) 
    p.add_argument("--round_timeout", type=float, default=300.0)
    p.add_argument("--eval_split", type=str, default="val")
    p.add_argument("--img_size", type=int, default=299)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--dp_replace_bn", action="store_true")
    p.add_argument("--fraction_fit", type=float, default=1.0)
    p.add_argument("--fraction_evaluate", type=float, default=1.0)
    p.add_argument("--min_fit_clients", type=int, default=1)
    p.add_argument("--min_evaluate_clients", type=int, default=1)
    p.add_argument("--min_available_clients", type=int, default=1)
    p.add_argument("--use_cbam", action="store_true", help="Use CBAM attention module")
    return p.parse_args()

# -----
# Main
# -----
def main():
    args = parse_args()

    num_classes = infer_num_classes(args.data_root, args.eval_split)

    model = build_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=args.pretrained,
        dp_replace_bn=args.dp_replace_bn,
        use_cbam=args.use_cbam,
    )
    initial_parameters = ndarrays_to_parameters(get_weights(model))

    print(
        f"INFO :      [Server] Starting at {args.server_address} for {args.num_rounds} rounds "
        f"(model={args.model}, img={args.img_size}, eval_split={args.eval_split}, classes={num_classes}, "
        f"cbam={args.use_cbam})..."
    )

    strategy = AdaptiveFedProxWithLogging(
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=args.min_fit_clients,
        min_evaluate_clients=args.min_evaluate_clients,
        min_available_clients=args.min_available_clients,
        initial_parameters=initial_parameters,
        proximal_mu=0.01,          # try 0.001–0.1
        fit_metrics_aggregation_fn=weighted_average_fit,
        evaluate_metrics_aggregation_fn=weighted_average_eval,
    )

    history = fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(
            num_rounds=args.num_rounds,
            round_timeout=args.round_timeout,
        ),
        strategy=strategy,
    )

    print("\nINFO :      [SUMMARY]")
    print(f"INFO :      Run finished {args.num_rounds} rounds")
    print(f"INFO :      Best validation accuracy: {strategy.best_val_accuracy:.4f}")

if __name__ == "__main__":
    main()