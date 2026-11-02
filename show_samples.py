"""Task 2: load the corrosion dataset, show 5 images with their masks.

Needs dataset_utils.py in the same folder. Pairs images and masks inside each
split (train / validation / test) so files with the same name in different
splits are never mixed up.
"""
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt

import dataset_utils as du

N_SAMPLES = 5
SEED = 42


def load_pair(img_path, mask_path, max_side=1000):
    img = cv2.imread(str(img_path))
    raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if img is None or raw is None:
        raise IOError(f"Could not read {img_path} or {mask_path}")
    mask = du.to_binary(raw) * 255
    # Shrink big images for display only. Masks use NEAREST so labels stay intact.
    scale = min(1.0, max_side / max(img.shape[:2]))
    new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST)
    return img, mask


def overlay(img, mask, alpha=0.4):
    color = np.zeros_like(img)
    color[mask > 0] = (0, 0, 255)  # red in BGR
    return cv2.addWeighted(img, 1, color, alpha, 0)


def main():
    pairs, unmatched_images, unmatched_masks = du.find_pairs()
    print(f"Matched pairs: {len(pairs)}  images without mask: {len(unmatched_images)}  "
          f"masks without image: {len(unmatched_masks)}")
    counts = {}
    for p in pairs:
        counts[p["split"]] = counts.get(p["split"], 0) + 1
    print(f"Pairs per split: {counts}")
    if not pairs:
        print("No pairs found. Set IMAGES_DIR and MASKS_DIR in dataset_utils.py.")
        return

    first = pairs[0]
    raw_img = cv2.imread(str(first["image"]))
    raw_mask = cv2.imread(str(first["mask"]), cv2.IMREAD_UNCHANGED)
    print(f"First image shape: {raw_img.shape}  mask shape: {raw_mask.shape}  "
          f"mask dtype: {raw_mask.dtype}  mask values: {np.unique(raw_mask)}")

    random.seed(SEED)
    chosen = random.sample(pairs, min(N_SAMPLES, len(pairs)))

    fig, axes = plt.subplots(len(chosen), 3, figsize=(12, 4 * len(chosen)))
    axes = np.atleast_2d(axes)
    for row, p in zip(axes, chosen):
        img, mask = load_pair(p["image"], p["mask"])
        coverage = 100 * np.count_nonzero(mask) / mask.size
        row[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        row[0].set_title(f"{p['split']}: {p['image'].name}", fontsize=9)
        row[1].imshow(mask, cmap="gray")
        row[1].set_title(f"mask ({coverage:.1f}% corrosion)", fontsize=9)
        row[2].imshow(cv2.cvtColor(overlay(img, mask), cv2.COLOR_BGR2RGB))
        row[2].set_title("overlay", fontsize=9)
        for ax in row:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("samples.png", dpi=120)
    print("Saved samples.png")
    plt.show()


if __name__ == "__main__":
    main()