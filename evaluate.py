import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split

from transformers import SegformerForSemanticSegmentation

from dataset import KITTIRoadDataset


# =========================================================
# Device
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)


# =========================================================
# Dataset
# =========================================================

dataset = KITTIRoadDataset(
    root="../road_segmentation/data/data_road/training"
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False
)

print("Validation samples:", len(val_dataset))


# =========================================================
# Model
# =========================================================

model_name = "nvidia/segformer-b0-finetuned-ade-512-512"

model = SegformerForSemanticSegmentation.from_pretrained(
    model_name
)

# 150 classes → 2 classes
model.decode_head.classifier = nn.Conv2d(
    256,
    2,
    kernel_size=1
)

model.config.num_labels = 2


# =========================================================
# Load BEST checkpoint
# =========================================================

model.load_state_dict(
    torch.load(
        "segformer_kitti_best.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


# =========================================================
# Metrics
# =========================================================

intersection_total = 0.0
union_total = 0.0

prediction_total = 0.0
mask_total = 0.0


# =========================================================
# Evaluation
# =========================================================

with torch.no_grad():

    for images, masks in val_loader:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(
            pixel_values=images
        )

        logits = outputs.logits

        # Resize predictions to mask resolution
        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # Choose class with highest probability
        predictions = torch.argmax(
            logits,
            dim=1
        )

        # Remove mask channel
        masks = masks.squeeze(1).long()

        # Keep only road class
        predictions = (
            predictions == 1
        ).float()

        masks = (
            masks == 1
        ).float()

        # -------------------------------------------------
        # Intersection
        # -------------------------------------------------

        intersection = (
            predictions * masks
        ).sum()

        # -------------------------------------------------
        # Union
        # -------------------------------------------------

        union = (
            (predictions + masks) > 0
        ).float().sum()

        intersection_total += (
            intersection.item()
        )

        union_total += (
            union.item()
        )

        # -------------------------------------------------
        # Dice components
        # -------------------------------------------------

        prediction_total += (
            predictions.sum().item()
        )

        mask_total += (
            masks.sum().item()
        )


# =========================================================
# Final Metrics
# =========================================================

iou = (
    intersection_total /
    union_total
)

dice = (
    2 * intersection_total /
    (prediction_total + mask_total)
)


# =========================================================
# Results
# =========================================================

print()
print("==============================")
print("SegFormer Evaluation")
print("==============================")

print(f"IoU:  {iou:.4f}")
print(f"Dice: {dice:.4f}")

print()
print(f"IoU:  {iou * 100:.2f}%")
print(f"Dice: {dice * 100:.2f}%")