"""Week 2: check the dataset and fix the split.

Checks: every image has a mask, sizes match, mask values, empty masks,
corrosion share per split, duplicates / leakage between splits.
Writes dataset_report.txt and splits.json.
"""
import hashlib
import json
from collections import defaultdict

import cv2
import numpy as np

import dataset_utils as du

lines = []


def out(text=""):
    print(text)
    lines.append(text)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    pairs, unmatched_images, unmatched_masks = du.find_pairs()
    out(f"Matched pairs: {len(pairs)}")
    out(f"Images without a mask: {len(unmatched_images)}")
    out(f"Masks without an image: {len(unmatched_masks)}")
    for p in unmatched_images[:10]:
        out(f"  image without mask: {p}")
    for p in unmatched_masks[:10]:
        out(f"  mask without image: {p}")
    if not pairs:
        out("No pairs found. Set IMAGES_DIR and MASKS_DIR in dataset_utils.py.")
        save()
        return

    splits, official = du.get_splits()
    out("")
    out(f"Official split found in folders: {official}")
    total = sum(len(v) for v in splits.values())
    for name in ("train", "val", "test"):
        n = len(splits[name])
        out(f"  {name:5s}: {n:4d} images ({100 * n / total:.1f}%)")
    out("  expected: about 75% / 15% / 15%")

    # Per-pair checks
    size_mismatch, unreadable = [], []
    image_sizes, raw_values = set(), set()
    empty_masks = defaultdict(int)
    ratios = defaultdict(list)
    for name, items in splits.items():
        for p in items:
            img = cv2.imread(str(p["image"]))
            raw = cv2.imread(str(p["mask"]), cv2.IMREAD_UNCHANGED)
            if img is None or raw is None:
                unreadable.append(p["image"])
                continue
            image_sizes.add(img.shape[:2])
            if img.shape[:2] != raw.shape[:2]:
                size_mismatch.append((p["image"].name, img.shape[:2], raw.shape[:2]))
            if len(raw_values) < 20:
                raw_values.update(np.unique(raw).tolist())
            binary = du.to_binary(raw)
            if binary.sum() == 0:
                empty_masks[name] += 1
            ratios[name].append(binary.mean())

    out("")
    out(f"Unreadable files: {len(unreadable)}")
    out(f"Image sizes found (h, w): {sorted(image_sizes)}")
    out(f"Image/mask size mismatches: {len(size_mismatch)}")
    for item in size_mismatch[:10]:
        out(f"  {item}")
    shown = sorted(raw_values)[:20]
    out(f"Raw mask values seen (first masks): {shown}")
    if len(raw_values) > 3:
        out("  WARNING: more than 2-3 distinct values. Masks may be JPEG-compressed "
            "or multi-class. Check a few by eye.")
    out("")
    out("Corrosion share of pixels per split (mean) and empty masks:")
    for name in ("train", "val", "test"):
        if ratios[name]:
            out(f"  {name:5s}: mean {100 * np.mean(ratios[name]):.1f}%  "
                f"min {100 * np.min(ratios[name]):.1f}%  max {100 * np.max(ratios[name]):.1f}%  "
                f"empty masks: {empty_masks[name]}")

    # Leakage: same file content or same name in more than one split
    out("")
    by_hash = defaultdict(list)
    by_key = defaultdict(set)
    for name, items in splits.items():
        for p in items:
            by_hash[md5(p["image"])].append((name, p["image"].name))
            by_key[p["key"]].add(name)
    dup_hash = {h: v for h, v in by_hash.items() if len(v) > 1}
    dup_key = {k: v for k, v in by_key.items() if len(v) > 1}
    out(f"Identical image files appearing more than once: {len(dup_hash)}")
    for v in list(dup_hash.values())[:10]:
        out(f"  {v}")
    out(f"File names used in more than one split: {len(dup_key)}")
    for k, v in list(dup_key.items())[:10]:
        out(f"  {k}: {sorted(v)}")
    leaks = [v for v in dup_hash.values() if len({s for s, _ in v}) > 1]
    out("LEAKAGE: none found." if not leaks and not dup_key
        else "LEAKAGE WARNING: review the lines above before training.")

    # Save the split so everyone uses the same one
    with open("splits.json", "w", encoding="utf-8") as f:
        json.dump({
            "official_split": official,
            "seed": du.SEED,
            **{name: [str(p["image"]) for p in items] for name, items in splits.items()},
        }, f, indent=2)
    out("")
    out("Saved splits.json and dataset_report.txt")
    save()


def save():
    with open("dataset_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()