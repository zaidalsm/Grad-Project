"""Shared helpers: find image/mask pairs, detect the official split, load masks.

Put the extracted dataset under ./data. If automatic detection picks wrong
folders, set IMAGES_DIR and MASKS_DIR below (both must be set).
"""
import random
from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path("data")
IMAGES_DIR = None   # root folder of the images (split folders may be inside)
MASKS_DIR = None    # root folder of the masks (split folders may be inside)
SEED = 42

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_WORDS = ("mask", "label", "annot", "segm")
SUFFIXES = ("_mask", "-mask", "_label", "_labels", "_gt", "_annotation", "_seg")
SPLIT_NAMES = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val",
    "test": "test", "testing": "test",
}


def list_images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTS)


def is_mask(path):
    return any(w in str(path.relative_to(DATA_DIR)).lower() for w in MASK_WORDS)


def key(path):
    stem = path.stem.lower()
    for suf in SUFFIXES:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    return stem


def split_of(path, root):
    """Return 'train', 'val', 'test' if a parent folder is named like one, else None."""
    for part in path.relative_to(root).parts[:-1]:
        name = SPLIT_NAMES.get(part.lower())
        if name:
            return name
    return None


def find_pairs():
    """Return (pairs, unmatched_images, unmatched_masks).

    Each pair is a dict: image, mask, split ('train'/'val'/'test'/None), key.
    """
    if IMAGES_DIR and MASKS_DIR:
        img_root, mask_root = Path(IMAGES_DIR), Path(MASKS_DIR)
        images, masks = list_images(img_root), list_images(mask_root)
    else:
        img_root = mask_root = DATA_DIR
        files = list_images(DATA_DIR)
        masks = [f for f in files if is_mask(f)]
        images = [f for f in files if not is_mask(f)]

    mask_by_key = {(split_of(m, mask_root), key(m)): m for m in masks}
    pairs, unmatched_images = [], []
    for img in images:
        k = (split_of(img, img_root), key(img))
        if k in mask_by_key:
            pairs.append({"image": img, "mask": mask_by_key[k],
                          "split": k[0], "key": k[1]})
        else:
            unmatched_images.append(img)
    used = {p["mask"] for p in pairs}
    unmatched_masks = [m for m in masks if m not in used]
    return pairs, unmatched_images, unmatched_masks


def get_splits(seed=SEED):
    """Return ({'train': [...], 'val': [...], 'test': [...]}, official_split_found)."""
    pairs, _, _ = find_pairs()
    if pairs and all(p["split"] for p in pairs):
        splits = {"train": [], "val": [], "test": []}
        for p in pairs:
            splits[p["split"]].append(p)
        return splits, True

    # No official split in the folders: make our own 75:15:15, same every run.
    pairs = sorted(pairs, key=lambda p: p["key"])
    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    n_train, n_val = round(0.75 * n), round(0.15 * n)
    return {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }, False


def to_binary(m):
    """Raw mask array -> uint8 array with 1 = corrosion, 0 = background."""
    if m.ndim == 3:
        if m.shape[2] == 4:      # drop alpha channel
            m = m[..., :3]
        m = m.max(axis=2)
    thr = 127 if m.max() > 1 else 0
    return (m > thr).astype(np.uint8)


def load_binary_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise IOError(f"Could not read mask: {path}")
    return to_binary(m)