"""Week 3 (and later, the interface): run the trained model on one image.

Usage:  python predict_one.py path/to/image.png
        (with no argument it uses the first test image)

Saves predictions/<name>_mask.png (0 / 255) and predictions/<name>_overlay.png.
Other code can call analyze_image(path) and get a dict back.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

import dataset_utils as du

CKPT = Path("runs/unet_resnet34/best_model.pt")
OUT_DIR = Path("predictions")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_cache = {}


def load_model(ckpt_path=CKPT):
    if "model" in _cache:
        return _cache["model"], _cache["ckpt"], _cache["device"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    model = smp.Unet(ckpt["encoder"], encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    _cache.update(model=model, ckpt=ckpt, device=device)
    return model, ckpt, device


@torch.no_grad()
def analyze_image(image_path):
    """Return {'mask': uint8 HxW (0/255), 'overlay': BGR image, 'corrosion_ratio': float}."""
    model, ckpt, device = load_model()
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise IOError(f"Could not read image: {image_path}")
    h, w = bgr.shape[:2]
    size = ckpt["img_size"]

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)

    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    binary = (prob > ckpt["threshold"]).astype(np.uint8)

    color = np.zeros_like(bgr)
    color[binary > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(bgr, 1, color, 0.4, 0)
    return {"mask": binary * 255, "overlay": overlay,
            "corrosion_ratio": float(binary.mean())}


def main():
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    else:
        splits, _ = du.get_splits()
        if not splits["test"]:
            raise SystemExit("No test image found. Pass an image path.")
        image_path = splits["test"][0]["image"]
    print(f"Image: {image_path}")

    result = analyze_image(image_path)
    OUT_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT_DIR / f"{image_path.stem}_mask.png"), result["mask"])
    cv2.imwrite(str(OUT_DIR / f"{image_path.stem}_overlay.png"), result["overlay"])
    print(f"Corrosion area: {100 * result['corrosion_ratio']:.1f}% of the image")
    print(f"Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()