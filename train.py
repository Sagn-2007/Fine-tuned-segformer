import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from transformers import (
    SegformerForSemanticSegmentation
)

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

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False
)

print("Train samples:", len(train_dataset))
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

model = model.to(device)


# =========================================================
# Fine-Tuning Setup
# =========================================================

# Freeze entire SegFormer backbone first
for param in model.segformer.parameters():
    param.requires_grad = False

# Fine-tune Stage 2
for param in model.segformer.stages[2].parameters():
    param.requires_grad = True

# Fine-tune Stage 3
for param in model.segformer.stages[3].parameters():
    param.requires_grad = True


# =========================================================
# Loss + Optimizer
# =========================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam([
    {
        "params": model.segformer.stages[2].parameters(),
        "lr": 5e-6
    },
    {
        "params": model.segformer.stages[3].parameters(),
        "lr": 1e-5
    },
    {
        "params": model.decode_head.parameters(),
        "lr": 5e-4
    }
])


# =========================================================
# Training
# =========================================================

epochs = 20

best_iou = 0.0

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # SegFormer expects pixel_values
        outputs = model(
            pixel_values=images
        )

        logits = outputs.logits

        # Resize logits to mask resolution
        logits = torch.nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # [B, 1, H, W] → [B, H, W]
        masks = masks.squeeze(1).long()

        loss = criterion(
            logits,
            masks
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    train_loss = (
        running_loss / len(train_loader)
    )


    # =====================================================
    # Validation
    # =====================================================

    model.eval()

    intersection_total = 0.0
    union_total = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(
                pixel_values=images
            )

            logits = outputs.logits

            logits = torch.nn.functional.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            masks = masks.squeeze(1).long()

            # Road = class 1
            predictions = (
                predictions == 1
            ).float()

            masks = (
                masks == 1
            ).float()

            intersection = (
                predictions * masks
            ).sum()

            union = (
                (predictions + masks) > 0
            ).float().sum()

            intersection_total += (
                intersection.item()
            )

            union_total += (
                union.item()
            )

    val_iou = (
        intersection_total /
        union_total
    )


    # =====================================================
    # Print Results
    # =====================================================

    print(
        f"Epoch {epoch+1:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val IoU: {val_iou:.4f}"
    )


    # =====================================================
    # Save Best Model
    # =====================================================

    if val_iou > best_iou:

        best_iou = val_iou

        torch.save(
            model.state_dict(),
            "segformer_kitti_best.pth"
        )

        print(
            f"  New best model saved! "
            f"IoU: {best_iou:.4f}"
        )


# =========================================================
# Training Complete
# =========================================================

print()
print("==============================")
print("Training Complete")
print("==============================")
print(f"Best Val IoU: {best_iou:.4f}")
print("Best model: segformer_kitti_best.pth")