import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split

from transformers import SegformerForSemanticSegmentation

from dataset import KITTIRoadDataset


# =========================================================
# Configuration
# =========================================================

MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"

DATASET_ROOT = "data_road/training"

BATCH_SIZE = 4
EPOCHS = 20

STAGE2_LR = 5e-6
STAGE3_LR = 1e-5
HEAD_LR = 5e-4

SEED = 42

MODEL_OUTPUT = "segformer_b2_kitti_best.pth"


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
# Reproducibility
# =========================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# Dataset
# =========================================================

dataset = KITTIRoadDataset(
    root=DATASET_ROOT
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(SEED)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

print()
print("Dataset:")
print("Total:", len(dataset))
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))


# =========================================================
# DataLoaders
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# =========================================================
# Load pretrained SegFormer-B2
# =========================================================

print()
print("Loading pretrained SegFormer-B2...")

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME
)


# =========================================================
# Replace segmentation head
# =========================================================

model.decode_head.classifier = nn.Conv2d(
    768,
    2,
    kernel_size=1
)

model.config.num_labels = 2


# =========================================================
# Freeze backbone
# =========================================================

for param in model.segformer.parameters():
    param.requires_grad = False


# =========================================================
# Fine-tune Stage 2
# =========================================================

for param in model.segformer.stages[2].parameters():
    param.requires_grad = True


# =========================================================
# Fine-tune Stage 3
# =========================================================

for param in model.segformer.stages[3].parameters():
    param.requires_grad = True


# Move model to GPU
model = model.to(device)


# =========================================================
# Loss
# =========================================================

criterion = nn.CrossEntropyLoss()


# =========================================================
# Optimizer
# =========================================================

optimizer = torch.optim.Adam([
    {
        "params": model.segformer.stages[2].parameters(),
        "lr": STAGE2_LR
    },
    {
        "params": model.segformer.stages[3].parameters(),
        "lr": STAGE3_LR
    },
    {
        "params": model.decode_head.parameters(),
        "lr": HEAD_LR
    }
])


# =========================================================
# Print trainable parameters
# =========================================================

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print()
print("Model:", MODEL_NAME)
print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params)

print()
print("Fine-tuning:")
print("Stage 0 → Frozen")
print("Stage 1 → Frozen")
print("Stage 2 → Trainable")
print("Stage 3 → Trainable")
print("Decode Head → Trainable")

print()
print("Learning rates:")
print("Stage 2:", STAGE2_LR)
print("Stage 3:", STAGE3_LR)
print("Decode Head:", HEAD_LR)


# =========================================================
# Training
# =========================================================

best_iou = 0.0

print()
print("==============================")
print("Starting Training")
print("==============================")


for epoch in range(EPOCHS):

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    model.train()

    running_loss = 0.0
    total_batches = 0

    for images, masks in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        masks = masks.to(
            device,
            non_blocking=True
        )

        # [B, 1, H, W] → [B, H, W]
        masks = masks.squeeze(1).long()

        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            pixel_values=images
        )

        logits = outputs.logits

        # Resize model output to mask resolution
        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # Cross entropy loss
        loss = criterion(
            logits,
            masks
        )

        # Backpropagation
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        total_batches += 1

    train_loss = running_loss / total_batches


    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    model.eval()

    intersection_total = 0.0
    union_total = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            masks = masks.to(
                device,
                non_blocking=True
            )

            masks = masks.squeeze(1).long()

            outputs = model(
                pixel_values=images
            )

            logits = outputs.logits

            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            # Road = class 1
            predictions = predictions == 1
            ground_truth = masks == 1

            intersection = (
                predictions & ground_truth
            ).sum().item()

            union = (
                predictions | ground_truth
            ).sum().item()

            intersection_total += intersection
            union_total += union

    if union_total > 0:
        val_iou = (
            intersection_total /
            union_total
        )
    else:
        val_iou = 1.0


    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val IoU: {val_iou:.4f}"
    )


    # -----------------------------------------------------
    # Save best model
    # -----------------------------------------------------

    if val_iou > best_iou:

        best_iou = val_iou

        torch.save(
            model.state_dict(),
            MODEL_OUTPUT
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

print(
    f"Best Val IoU: {best_iou:.4f}"
)

print(
    f"Best model: {MODEL_OUTPUT}"
)
