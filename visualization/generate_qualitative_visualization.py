import os, glob, json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# =========================================================
# PATHS
# =========================================================
VIS_ROOT = "/kaggle/input/datasets/albabfahad/visualization/visualization"
CKPT_PATH = os.path.join(VIS_ROOT, "inceptionv3_roi_layer_beta2p0_best_val_global_model.pt")
PRED_PATH = os.path.join(VIS_ROOT, "test_predictions.csv")
SUMMARY_PATH = os.path.join(VIS_ROOT, "experiment_summary.json")

# Change this only if your OASIS dataset path is different
OASIS_ROOT = "/kaggle/input/datasets/albabfahad/oasis-dataset/new_oasis"

OUT_PNG = "/kaggle/working/qualitative_oasis_inceptionv3_roi_layer_beta2p0.png"

# =========================================================
# CONFIG
# =========================================================
CLASS_NAMES = ["AD", "CN", "MCI"]
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 299
RESIZE_SIZE = 330

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

ROI_LAYER_BETA = 2.0
ROI_LAYER_MIN = 0.25
ROI_LAYER_MAX = 2.5
ROI_LAYER_NORM = True

# =========================================================
# HELPERS
# =========================================================
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

def normalize_map(x):
    x = x - x.min()
    xmax = x.max()
    if xmax > 0:
        x = x / xmax
    return x

def make_noise_map(importance):
    raw = 1.0 + ROI_LAYER_BETA * (1.0 - importance)
    raw = np.clip(raw, ROI_LAYER_MIN, ROI_LAYER_MAX)
    if ROI_LAYER_NORM:
        raw = raw / (raw.mean() + 1e-8)
    return raw

def denorm_to_gray(img_tensor):
    x = img_tensor.detach().cpu().clone()
    for c in range(3):
        x[c] = x[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    x = x.clamp(0, 1).permute(1, 2, 0).numpy()
    gray = x.mean(axis=2)
    return gray

def find_image_by_basename(root, basename):
    matches = glob.glob(os.path.join(root, "test", "*", basename))
    if len(matches) == 0:
        return None
    return matches[0]

# =========================================================
# TRANSFORM
# =========================================================
eval_tfms = transforms.Compose([
    AutoCropBrain(8, 0.06),
    transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# =========================================================
# MODEL
# =========================================================
def build_model():
    model = models.inception_v3(weights=None, aux_logits=True)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.aux_logits = False
    model.AuxLogits = None
    return model

model = build_model().to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt)
model.eval()

# target layer for Grad-CAM
target_layer = model.Mixed_7c
activations = []
gradients = []

def fwd_hook(module, inp, out):
    activations.append(out)

def bwd_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

h1 = target_layer.register_forward_hook(fwd_hook)
h2 = target_layer.register_full_backward_hook(bwd_hook)

# =========================================================
# PICK ONE CORRECTLY PREDICTED TEST SAMPLE PER CLASS
# =========================================================
pred_df = pd.read_csv(PRED_PATH)

selected = []
for cls in CLASS_NAMES:
    sub = pred_df[(pred_df["true_label"] == cls) & (pred_df["pred_label"] == cls)].copy()
    if len(sub) == 0:
        sub = pred_df[pred_df["true_label"] == cls].copy()
    if len(sub) == 0:
        raise RuntimeError(f"No sample found for class {cls}")

    # prefer higher confidence in the true class
    prob_col = f"prob_{cls}"
    if prob_col in sub.columns:
        sub = sub.sort_values(prob_col, ascending=False)

    row = sub.iloc[0]
    base = os.path.basename(row["path"])
    current_path = find_image_by_basename(OASIS_ROOT, base)

    if current_path is None:
        # fallback: try original path if accessible
        current_path = row["path"]

    selected.append({
        "class_name": cls,
        "img_path": current_path
    })

print("Selected samples:")
for s in selected:
    print(s["class_name"], "->", s["img_path"])

# =========================================================
# GENERATE MAPS
# =========================================================
rows = []

for s in selected:
    img_path = s["img_path"]
    pil_img = Image.open(img_path).convert("RGB")
    x = eval_tfms(pil_img).unsqueeze(0).to(DEVICE)

    activations.clear()
    gradients.clear()

    logits = model(x)
    if isinstance(logits, tuple):
        logits = logits[0]

    pred = int(torch.argmax(logits, dim=1).item())

    model.zero_grad(set_to_none=True)
    logits[0, pred].backward()

    acts = activations[-1]      # [1,C,H,W]
    grads = gradients[-1]       # [1,C,H,W]

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam = normalize_map(cam)

    noise_map = make_noise_map(cam)
    gray_img = denorm_to_gray(x[0])

    rows.append({
        "true_class": s["class_name"],
        "pred_class": CLASS_NAMES[pred],
        "gray_img": gray_img,
        "importance_map": cam,
        "noise_map": noise_map
    })

h1.remove()
h2.remove()

# =========================================================
# PLOT
# =========================================================
fig, axes = plt.subplots(len(rows), 4, figsize=(12, 9))

col_titles = ["Input MRI", "Importance map", "Perturbation-scale map", "Overlay"]
for j, title in enumerate(col_titles):
    axes[0, j].set_title(title, fontsize=12)

for i, row in enumerate(rows):
    img = row["gray_img"]
    cam = row["importance_map"]
    noise_map = row["noise_map"]

    axes[i, 0].imshow(img, cmap="gray")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(cam, cmap="jet")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(noise_map, cmap="viridis")
    axes[i, 2].axis("off")

    axes[i, 3].imshow(img, cmap="gray")
    axes[i, 3].imshow(cam, cmap="jet", alpha=0.45)
    axes[i, 3].axis("off")

# =========================================================
# ADD ROW LABELS HERE: AD / CN / MCI
# =========================================================
row_labels = ["AD", "CN", "MCI"]
row_y_positions = [0.80, 0.50, 0.20]   # top, middle, bottom

for label, ypos in zip(row_labels, row_y_positions):
    fig.text(
        0.01, ypos, label,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="center"
    )

plt.tight_layout(rect=[0.04, 0.0, 1.0, 1.0])
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
plt.show()

print("Saved figure to:", OUT_PNG)

# Optional: print metadata
if os.path.exists(SUMMARY_PATH):
    with open(SUMMARY_PATH, "r") as f:
        summary = json.load(f)
    print("\nExperiment summary:")
    for k in ["dataset", "mode", "backbone", "test_accuracy", "test_auc_macro_ovr", "roi_layer_spatial_beta"]:
        if k in summary:
            print(f"{k}: {summary[k]}")