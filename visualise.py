import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split

from transformers import SegformerForSemanticSegmentation

from dataset import KITTIRoadDataset


# =========================================================
# Settings
# =========================================================

NUM_IMAGES = 10

MODEL_PATH = "segformer_kitti_best.pth"

DATASET_ROOT = "../road_segmentation/data/data_road/training"

RESULTS_DIR = "results"


# =========================================================
# Device
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)


# =========================================================
# Create results folder
# =========================================================

os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================================================
# Dataset
# =========================================================

dataset = KITTIRoadDataset(
    root=DATASET_ROOT
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

print("Validation samples:", len(val_dataset))


# =========================================================
# Model
# =========================================================

model_name = "nvidia/segformer-b0-finetuned-ade-512-512"

model = SegformerForSemanticSegmentation.from_pretrained(
    model_name
)

# Replace 150-class classifier with 2-class classifier
model.decode_head.classifier = nn.Conv2d(
    256,
    2,
    kernel_size=1
)

model.config.num_labels = 2

# Load best KITTI model
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()


# =========================================================
# ImageNet normalization values
# =========================================================

mean = torch.tensor(
    [0.485, 0.456, 0.406]
).view(3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225]
).view(3, 1, 1)


# =========================================================
# Random validation samples
# =========================================================

random.seed(42)

indices = random.sample(
    range(len(val_dataset)),
    NUM_IMAGES
)


# =========================================================
# Visualization
# =========================================================

for number, index in enumerate(indices, start=1):

    image, mask = val_dataset[index]

    # Add batch dimension
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

        # Resize to original mask size
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

        prediction = (
            prediction == 1
        ).float()


    # -----------------------------------------------------
    # Ground truth
    # -----------------------------------------------------

    ground_truth = (
        mask == 1
    ).float().cpu()


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    prediction = (
        prediction == 1
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
    # Denormalize image
    # -----------------------------------------------------

    image_display = (
        image.cpu() * std + mean
    )

    image_display = image_display.clamp(
        0, 1
    )

    image_display = image_display.permute(
        1, 2, 0
    ).numpy()


    # -----------------------------------------------------
    # Convert masks to numpy
    # -----------------------------------------------------

    ground_truth = ground_truth.cpu().numpy()
    prediction = prediction.cpu().numpy()


    # -----------------------------------------------------
    # Create figure
    # -----------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )


    # Original
    axes[0].imshow(image_display)
    axes[0].set_title("Original Image")
    axes[0].axis("off")


    # Ground Truth
    axes[1].imshow(ground_truth, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")


    # Prediction
    axes[2].imshow(prediction, cmap="gray")
    axes[2].set_title(
        f"SegFormer Prediction\nIoU: {iou:.4f}"
    )
    axes[2].axis("off")


    # Overall title
    fig.suptitle(
        f"SegFormer KITTI Road Segmentation - Sample {number}",
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
print(f"{NUM_IMAGES} images saved to '{RESULTS_DIR}/'")