import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import random_split
from transformers import SegformerForSemanticSegmentation

from dataset import KITTIRoadDataset


# =========================================================
# Configuration
# =========================================================

MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"
MODEL_PATH = "segformer_b2_kitti_best.pth"

DATASET_ROOT = "data_road/training"

RESULTS_DIR = "results_b2"

NUM_IMAGES = 10
SEED = 42


# =========================================================
# Device
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =========================================================
# Dataset
# =========================================================

dataset = KITTIRoadDataset(
    root=DATASET_ROOT
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(SEED)

_, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

print("Validation samples:", len(val_dataset))


# =========================================================
# Results directory
# =========================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# =========================================================
# Load SegFormer-B2
# =========================================================

print("Loading SegFormer-B2...")

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME
)

model.decode_head.classifier = nn.Conv2d(
    768,
    2,
    kernel_size=1
)

model.config.num_labels = 2

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()


# =========================================================
# ImageNet normalization
# =========================================================

mean = torch.tensor(
    [0.485, 0.456, 0.406]
).view(3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225]
).view(3, 1, 1)


# =========================================================
# Select same validation samples
# =========================================================

random.seed(SEED)

indices = random.sample(
    range(len(val_dataset)),
    NUM_IMAGES
)


# =========================================================
# Generate visualizations
# =========================================================

for number, index in enumerate(
    indices,
    start=1
):

    image, mask = val_dataset[index]

    image_input = image.unsqueeze(0).to(device)

    mask = mask.squeeze(0)

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    with torch.no_grad():

        outputs = model(
            pixel_values=image_input
        )

        logits = outputs.logits

        logits = F.interpolate(
            logits,
            size=mask.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        prediction = torch.argmax(
            logits,
            dim=1
        ).squeeze(0)

        # Road = class 1
        prediction = (
            prediction == 1
        ).float().cpu()

    # -----------------------------------------------------
    # Ground truth
    # -----------------------------------------------------

    ground_truth = (
        mask == 1
    ).float().cpu()

    # -----------------------------------------------------
    # IoU
    # -----------------------------------------------------

    intersection = (
        prediction * ground_truth
    ).sum().item()

    union = (
        (prediction + ground_truth) > 0
    ).float().sum().item()

    if union > 0:
        iou = intersection / union
    else:
        iou = 1.0

    # -----------------------------------------------------
    # Prepare original image
    # -----------------------------------------------------

    image_display = (
        image.cpu() * std + mean
    ).clamp(0, 1)

    image_display = (
        image_display
        .permute(1, 2, 0)
        .numpy()
    )

    ground_truth_display = (
        ground_truth.numpy()
    )

    prediction_display = (
        prediction.numpy()
    )

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(
        image_display
    )

    axes[0].set_title(
        "Original Image"
    )

    axes[0].axis("off")


    axes[1].imshow(
        ground_truth_display,
        cmap="gray"
    )

    axes[1].set_title(
        "Ground Truth"
    )

    axes[1].axis("off")


    axes[2].imshow(
        prediction_display,
        cmap="gray"
    )

    axes[2].set_title(
        f"SegFormer-B2 Prediction\nIoU: {iou:.4f}"
    )

    axes[2].axis("off")


    fig.suptitle(
        f"SegFormer-B2 KITTI Road Segmentation - Sample {number}",
        fontsize=14
    )

    plt.tight_layout()


    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    output_path = os.path.join(
        RESULTS_DIR,
        f"sample_{number:02d}.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved: {output_path} | IoU: {iou:.4f}"
    )


print()
print("Visualization complete.")