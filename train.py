"""Week 3: first corrosion segmentation model (U-Net + pretrained ResNet34).

Run:  python train.py
Outputs in runs/unet_resnet34/: best_model.pt, training_log.csv, run_config.json
"""
import csv
import json
import random
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

import dataset_utils as du

ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
IMG_SIZE = 512          # images are 2048x2048; we train on 512x512 to fit memory
BATCH_SIZE = 4          # lower to 2 if you run out of GPU memory
EPOCHS = 40
LR = 3e-4
THRESH = 0.5
SEED = 42
OUT_DIR = Path("runs/unet_resnet34")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CorrosionDataset(Dataset):
    def __init__(self, pairs, augment):
        # Resize once and keep in memory (fast epochs). Masks use NEAREST so
        # labels are never blended.
        self.images, self.masks = [], []
        for p in pairs:
            img = cv2.imread(str(p["image"]))
            if img is None:
                raise IOError(f"Could not read image: {p['image']}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            mask = du.load_binary_mask(p["mask"])
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            self.images.append(img)
            self.masks.append(mask)

        steps = []
        if augment:  # training set only; image and mask are transformed together
            steps += [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05),
                         rotate=(-15, 15), p=0.5),
            ]
        steps += [A.Normalize(), ToTensorV2()]
        self.transform = A.Compose(steps)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        out = self.transform(image=self.images[i], mask=self.masks[i])
        return out["image"], out["mask"].float().unsqueeze(0)


def build_model():
    try:
        return smp.Unet(ENCODER, encoder_weights=ENCODER_WEIGHTS, in_channels=3, classes=1)
    except Exception as e:  # no internet for the pretrained weights
        print(f"Could not load pretrained weights ({e}). Training from scratch.")
        return smp.Unet(ENCODER, encoder_weights=None, in_channels=3, classes=1)


dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
bce_loss = torch.nn.BCEWithLogitsLoss()


def loss_fn(logits, target):
    return dice_loss(logits, target) + bce_loss(logits, target)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = 0
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * x.size(0)
        n += x.size(0)
        pred = torch.sigmoid(logits) > THRESH
        true = y > 0.5
        tp += (pred & true).sum().item()
        fp += (pred & ~true).sum().item()
        fn += (~pred & true).sum().item()
    iou = tp / (tp + fp + fn + 1e-9)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-9)
    return total_loss / max(n, 1), iou, dice


def main():
    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("No GPU found: training will be slow. Consider Google Colab or a lab GPU.")

    splits, official = du.get_splits()
    counts = {k: len(v) for k, v in splits.items()}
    print(f"Official split: {official}  counts: {counts}")
    if not splits["train"] or not splits["val"]:
        raise SystemExit("Train or validation set is empty. Run check_dataset.py first.")

    train_ds = CorrosionDataset(splits["train"], augment=True)
    val_ds = CorrosionDataset(splits["val"], augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=len(train_ds) > BATCH_SIZE)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "encoder": ENCODER, "encoder_weights": ENCODER_WEIGHTS, "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE, "epochs": EPOCHS, "lr": LR, "threshold": THRESH,
            "seed": SEED, "device": device, "official_split": official, "split_counts": counts,
            "versions": {
                "torch": torch.__version__,
                "segmentation_models_pytorch": smp.__version__,
                "albumentations": A.__version__,
                "opencv": cv2.__version__,
                "numpy": np.__version__,
            },
        }, f, indent=2)

    best_iou = -1.0
    with open(OUT_DIR / "training_log.csv", "w", newline="", encoding="utf-8") as logf:
        log = csv.writer(logf)
        log.writerow(["epoch", "train_loss", "val_loss", "val_iou", "val_dice", "seconds"])
        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            model.train()
            running, seen = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()
                running += loss.item() * x.size(0)
                seen += x.size(0)
            scheduler.step()

            val_loss, val_iou, val_dice = evaluate(model, val_loader, device)
            secs = time.time() - t0
            log.writerow([epoch, running / seen, val_loss, val_iou, val_dice, round(secs, 1)])
            logf.flush()
            mark = ""
            if val_iou > best_iou:  # keep the best model by validation IoU
                best_iou = val_iou
                torch.save({
                    "model_state": model.state_dict(), "encoder": ENCODER,
                    "img_size": IMG_SIZE, "threshold": THRESH,
                    "val_iou": val_iou, "epoch": epoch,
                }, OUT_DIR / "best_model.pt")
                mark = "  <- best, saved"
            print(f"Epoch {epoch:3d}/{EPOCHS}  train_loss {running / seen:.4f}  "
                  f"val_loss {val_loss:.4f}  IoU {val_iou:.4f}  Dice {val_dice:.4f}  "
                  f"{secs:.0f}s{mark}")

    print(f"Done. Best validation IoU: {best_iou:.4f}. Files are in {OUT_DIR}")


if __name__ == "__main__":
    main()