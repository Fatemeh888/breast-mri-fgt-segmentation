"""
Breast MRI FGT Segmentation (2D U-Net)

Notes:
- This script loads Breast MRI DICOM images and DICOM-SEG masks, extracts FGT segments,
  and trains a 2D U-Net for binary segmentation.
- Dataset is NOT included (size). You must set `root` to your local dataset path.
"""

import os

# Optional workaround for some Windows OpenMP conflicts (use only if you get OMP Error #15)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Optional: set a Matplotlib backend for PyCharm display issues 
import matplotlib
matplotlib.use("TkAgg")

from glob import glob
from PIL import Image

import numpy as np
import pydicom
import SimpleITK as sitk

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader  # random_split is currently not used
import torchvision.transforms as T

import matplotlib.pyplot as plt


# -------------------------
# Dataset: FGT Segmentation
# -------------------------
class BreastFGTDataset(Dataset):
    """
    For each patient folder inside `root`:
      1) Find a DICOM-SEG file (e.g., series folder like 300.*Segmentation*)
      2) Find a regular DICOM image file (any DICOM path not containing 'Segmentation')
      3) Identify the segment indices whose label corresponds to "Mammary Fibroglandular Tissue"
      4) Build a binary FGT mask (FGT=1, background=0)
    """

    def __init__(self, root: str, transform=None):
        self.root = root
        self.samples = []
        self.transform = transform

        # Patient folders convention used in this dataset
        patient_ids = sorted([d for d in os.listdir(root) if d.startswith("Breast_MRI_")])

        for pid in patient_ids:
            pdir = os.path.join(root, pid)
            if not os.path.isdir(pdir):
                continue

            # 1) Find a DICOM-SEG file (pick the first match)
            seg_candidates = glob(
                os.path.join(pdir, "**", "300.*Segmentation*", "*.dcm"),
                recursive=True,
            )
            if not seg_candidates:
                continue
            seg_path = seg_candidates[0]

            # 2) Find a regular DICOM image file (exclude paths that contain 'Segmentation')
            all_dicoms = glob(os.path.join(pdir, "**", "*.dcm"), recursive=True)
            img_candidates = [p for p in all_dicoms if "Segmentation" not in p]
            if not img_candidates:
                continue
            img_path = img_candidates[0]

            # 3) Parse DICOM-SEG to find FGT segment indices
            ds = pydicom.dcmread(seg_path)
            fgt_indices = []
            if hasattr(ds, "SegmentSequence"):
                for i, seg in enumerate(ds.SegmentSequence):
                    label = (getattr(seg, "SegmentLabel", "") or "").lower()
                    if "mammary fibroglandular tissue" in label:
                        fgt_indices.append(i)

            if not fgt_indices:
                continue

            self.samples.append(
                {
                    "img_path": img_path,
                    "seg_path": seg_path,
                    "fgt_indices": fgt_indices,
                    "patient_id": pid,
                }
            )

        print("FGT Samples found:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img_path = sample["img_path"]
        seg_path = sample["seg_path"]
        fgt_indices = sample["fgt_indices"]

        # ---- Load MRI image (DICOM) ----
        img_itk = sitk.ReadImage(img_path)
        img_arr = sitk.GetArrayFromImage(img_itk)  # often shape: (1, H, W) for single-slice

        # Convert to a 2D slice
        if img_arr.ndim == 3:
            img2d = img_arr[0]
        else:
            img2d = img_arr

        # Normalize to [0,1] then convert to uint8 for PIL
        img2d = img2d.astype(np.float32)
        img2d -= img2d.min()
        if img2d.max() > 0:
            img2d /= img2d.max()
        img_u8 = (img2d * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_u8).convert("L")

        # ---- Load segmentation (DICOM-SEG) and extract FGT mask ----
        seg_itk = sitk.ReadImage(seg_path)
        seg_arr = sitk.GetArrayFromImage(seg_itk)  # shape: (N, H, W) where N is number of frames

        # NOTE: This code assumes the indices from SegmentSequence align with seg_arr frame indexing.
        # Depending on the DICOM-SEG encoding, a more robust mapping may be needed.
        seg_fgt = seg_arr[fgt_indices]  # shape: (k, H, W)
        fgt_mask = (seg_fgt.max(axis=0) > 0).astype(np.uint8)  # shape: (H, W)

        mask_pil = Image.fromarray(fgt_mask * 255).convert("L")

        # Apply transform(s)
        if self.transform:
            img_tensor = self.transform(img_pil)   # [1, 256, 256]
            mask_tensor = self.transform(mask_pil) # [1, 256, 256]
        else:
            img_tensor = torch.from_numpy(img2d[None, ...]).float()
            mask_tensor = torch.from_numpy(fgt_mask[None, ...]).float()

        # Ensure mask is binary {0,1}
        mask_tensor = (mask_tensor > 0.5).float()
        return img_tensor, mask_tensor


# -------------------------
# U-Net building blocks
# -------------------------
class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Standard 2D U-Net for binary segmentation."""

    def __init__(self, in_ch: int = 1, out_ch: int = 1):
        super().__init__()

        # Encoder
        self.down1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottom = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        # Output layer (logits)
        self.out = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.down1(x)
        p1 = self.pool1(c1)

        c2 = self.down2(p1)
        p2 = self.pool2(c2)

        c3 = self.down3(p2)
        p3 = self.pool3(c3)

        c4 = self.down4(p3)
        p4 = self.pool4(c4)

        # Bottleneck
        c5 = self.bottom(p4)

        # Decoder with skip connections
        u4 = self.up4(c5)
        u4 = torch.cat([u4, c4], dim=1)
        c6 = self.conv4(u4)

        u3 = self.up3(c6)
        u3 = torch.cat([u3, c3], dim=1)
        c7 = self.conv3(u3)

        u2 = self.up2(c7)
        u2 = torch.cat([u2, c2], dim=1)
        c8 = self.conv2(u2)

        u1 = self.up1(c8)
        u1 = torch.cat([u1, c1], dim=1)
        c9 = self.conv1(u1)

        return self.out(c9)


# -------------------------
# Metrics & Losses
# -------------------------
def compute_dice_iou(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7):
    """
    Compute Dice and IoU for binary masks.

    Args:
        preds: predicted binary mask tensor [B, 1, H, W], values in {0,1}
        targets: ground-truth binary mask tensor [B, 1, H, W], values in {0,1}
    Returns:
        (dice, iou) as Python floats
    """
    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum().float()

    dice_den = preds.sum() + targets.sum()
    dice = (2 * intersection + eps) / (dice_den + eps)

    union_iou = preds.sum() + targets.sum() - intersection
    iou = (intersection + eps) / (union_iou + eps)

    return float(dice), float(iou)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7):
    """
    Soft Dice loss computed from logits.

    Args:
        logits: raw model outputs [B, 1, H, W]
        targets: ground truth [B, 1, H, W] in {0,1}
    """
    probs = torch.sigmoid(logits).view(-1)
    targets = targets.view(-1)

    intersection = (probs * targets).sum()
    return 1.0 - (2 * intersection + eps) / (probs.sum() + targets.sum() + eps)


# -------------------------
# Training Loop
# -------------------------
def main():
    # TODO: Set this path to your local dataset directory
    root = r"..."

    # Basic preprocessing: resize + convert to tensor
    transform = T.Compose(
        [
            T.Resize((256, 256)),
            T.ToTensor(),
        ]
    )

    dataset = BreastFGTDataset(root=root, transform=transform)
    print("Dataset length:", len(dataset))

    # For small-subset testing: use all data for training to verify the pipeline.
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_ch=1, out_ch=1).to(device)

    # BCEWithLogitsLoss with positive class weighting to reduce class-imbalance collapse
    pos_weight = torch.tensor([5.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        # ---- Train ----
        model.train()
        epoch_loss = 0.0

        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)

            # If you want combined loss, uncomment the Dice loss term:
            # bce = criterion(logits, masks)
            # d_loss = dice_loss(logits, masks)
            # loss = bce + d_loss
            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}: train loss = {epoch_loss/len(train_loader):.4f}")

        # ---- Train metrics (Dice/IoU) ----
        model.eval()
        train_dice = 0.0
        train_iou = 0.0
        n_batches = 0

        with torch.no_grad():
            for imgs, masks in train_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                logits = model(imgs)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                # Debug: check collapse (all-zero) vs over-segmentation (too many positives)
                print(
                    "preds positive:", preds.sum().item(),
                    "masks positive:", masks.sum().item()
                )

                dice, iou = compute_dice_iou(preds, masks)
                train_dice += dice
                train_iou += iou
                n_batches += 1

        print(f"           train dice = {train_dice / n_batches:.4f}")
        print(f"           train IoU  = {train_iou / n_batches:.4f}")

        model.train()

    # -----------------------------
    # Optional blocks (disabled)
    # -----------------------------
    """
    # ---- Example: Train/Val/Test split (enable if you have enough data-read README) ----
    from torch.utils.data import random_split

    train_size = int(0.7 * len(dataset))
    val_size   = int(0.15 * len(dataset))
    test_size  = len(dataset) - train_size - val_size

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=4, shuffle=False)
    """

    """
    # ---- Example: Validation with visualization (enable if desired) ----
    if len(val_loader) > 0:
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        n_batches = 0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                preds = (probs > 0.3).float()

                # Visualize one sample
                img_np = imgs[0].cpu().squeeze().numpy()
                mask_np = masks[0].cpu().squeeze().numpy()
                pred_np = preds[0].cpu().squeeze().numpy()

                plt.figure(figsize=(12, 4))

                plt.subplot(1, 3, 1)
                plt.imshow(img_np, cmap="gray")
                plt.title("Input Image")
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.imshow(mask_np, cmap="gray")
                plt.title("GT Mask")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.imshow(pred_np, cmap="gray")
                plt.title("Predicted Mask")
                plt.axis("off")

                plt.tight_layout()
                plt.show()

                dice, iou = compute_dice_iou(preds, masks)
                val_dice += dice
                val_iou += iou
                n_batches += 1

        print(f"           val loss   = {val_loss / len(val_loader):.4f}")
        print(f"           val dice   = {val_dice / n_batches:.4f}")
        print(f"           val IoU    = {val_iou / n_batches:.4f}")
    """

    """
    # ---- Example: Test evaluation (enable if desired) ----
    if len(test_loader) > 0:
        model.eval()
        test_loss = 0.0
        test_dice = 0.0
        test_iou = 0.0
        n_batches = 0

        with torch.no_grad():
            for imgs, masks in test_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                logits = model(imgs)
                loss = criterion(logits, masks)
                test_loss += loss.item()

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                dice, iou = compute_dice_iou(preds, masks)
                test_dice += dice
                test_iou += iou
                n_batches += 1

        print(f"Final Test Loss  = {test_loss / len(test_loader):.4f}")
        print(f"Final Test Dice  = {test_dice / n_batches:.4f}")
        print(f"Final Test IoU   = {test_iou / n_batches:.4f}")
    """


if __name__ == "__main__":
    main()
