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
MODEL_PATH = "segformer_b2_kitti_best.pth"
DATASET_ROOT = "data_road/training"

BATCH_SIZE = 4
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

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print()
print("Validation samples:", len(val_dataset))


# =========================================================
# Load Model
# =========================================================

print()
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
# Evaluation
# =========================================================

intersection_total = 0.0
union_total = 0.0

predicted_pixels_total = 0.0
ground_truth_pixels_total = 0.0


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

        predictions = predictions == 1
        ground_truth = masks == 1

        intersection = (
            predictions & ground_truth
        ).sum().item()

        union = (
            predictions | ground_truth
        ).sum().item()

        predicted_pixels = predictions.sum().item()
        ground_truth_pixels = ground_truth.sum().item()

        intersection_total += intersection
        union_total += union

        predicted_pixels_total += predicted_pixels
        ground_truth_pixels_total += ground_truth_pixels


# =========================================================
# Metrics
# =========================================================

iou = (
    intersection_total /
    union_total
)

dice = (
    2 * intersection_total /
    (
        predicted_pixels_total +
        ground_truth_pixels_total
    )
)


# =========================================================
# Results
# =========================================================

print()
print("==============================")
print("SegFormer-B2 Evaluation")
print("==============================")

print(
    f"Validation IoU:  {iou:.4f} "
    f"({iou * 100:.2f}%)"
)

print(
    f"Validation Dice: {dice:.4f} "
    f"({dice * 100:.2f}%)"
)

print("==============================")