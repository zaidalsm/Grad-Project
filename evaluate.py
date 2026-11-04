import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

import dataset_utils as du
from predict_one import analyze_image, load_model

OUT_DIR = Path("results")
EX_DIR = OUT_DIR / "examples"
N_EXAMPLES = 3     # best N and worst N images per split
PANEL = 512        # size of each picture in an example panel


def scores(tp, fp, fn):
    """IoU, Dice, precision, recall. If nothing is expected or predicted, the score is 1."""
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return iou, dice, precision, recall


def predict_and_truth(pair):
    t0 = time.perf_counter()
    result = analyze_image(pair["image"])
    seconds = time.perf_counter() - t0
    pred = result["mask"] > 0
    gt = du.load_binary_mask(pair["mask"]).astype(bool)
    if gt.shape != pred.shape:
        gt = cv2.resize(gt.astype(np.uint8), (pred.shape[1], pred.shape[0]),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
    return pred, gt, seconds


def tint(img, mask, color):
    layer = np.zeros_like(img)
    layer[mask] = color
    return cv2.addWeighted(img, 1, layer, 0.45, 0)


def save_example(pair, label, iou):
    pred, gt, _ = predict_and_truth(pair)
    img = cv2.imread(str(pair["image"]))
    pictures = [(img, "image"), (tint(img, gt, (0, 255, 0)), "ground truth"),
                (tint(img, pred, (0, 0, 255)), f"prediction IoU {iou:.2f}")]
    scale = PANEL / max(img.shape[:2])
    size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    tiles = []
    for pic, text in pictures:
        tile = cv2.resize(pic, size, interpolation=cv2.INTER_AREA)
        cv2.putText(tile, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
        cv2.putText(tile, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        tiles.append(tile)
    EX_DIR.mkdir(parents=True, exist_ok=True)
    out = EX_DIR / f"{pair['split']}_{label}_{pair['image'].stem}.png"
    cv2.imwrite(str(out), cv2.hconcat(tiles))


def main():
    splits, official = du.get_splits()
    _, ckpt, device = load_model()
    print(f"Device: {device}  best epoch in checkpoint: {ckpt.get('epoch')}  "
          f"threshold: {ckpt['threshold']}")

    # Warm-up call so the first timing is not unfairly slow
    first = next((p for s in ("val", "test") for p in splits[s]), None)
    if first is None:
        raise SystemExit("No validation or test images found.")
    analyze_image(first["image"])

    OUT_DIR.mkdir(exist_ok=True)
    summary = {"device": device, "official_split": official,
               "threshold": ckpt["threshold"], "best_epoch": ckpt.get("epoch")}

    for name in ("val", "test"):
        items = splits[name]
        if not items:
            continue
        rows, times = [], []
        TP = FP = FN = 0
        for p in items:
            pred, gt, seconds = predict_and_truth(p)
            tp = int((pred & gt).sum())
            fp = int((pred & ~gt).sum())
            fn = int((~pred & gt).sum())
            TP, FP, FN = TP + tp, FP + fp, FN + fn
            iou, dice, prec, rec = scores(tp, fp, fn)
            times.append(seconds)
            rows.append({"image": p["image"].name, "iou": iou, "dice": dice,
                         "precision": prec, "recall": rec,
                         "true_corrosion_%": 100 * gt.mean(),
                         "predicted_corrosion_%": 100 * pred.mean(),
                         "seconds": seconds, "_pair": p})

        with open(OUT_DIR / f"metrics_{name}.csv", "w", newline="", encoding="utf-8") as f:
            cols = [c for c in rows[0] if not c.startswith("_")]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: (round(r[c], 4) if isinstance(r[c], float) else r[c])
                            for c in cols})

        iou, dice, prec, rec = scores(TP, FP, FN)
        summary[name] = {
            "images": len(items),
            "iou": round(iou, 4), "dice": round(dice, 4),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "mean_image_iou": round(float(np.mean([r["iou"] for r in rows])), 4),
            "mean_seconds_per_image": round(float(np.mean(times)), 3),
            "max_seconds_per_image": round(float(np.max(times)), 3),
        }
        s = summary[name]
        print(f"\n{name}: {s['images']} images")
        print(f"  IoU {s['iou']}  Dice {s['dice']}  Precision {s['precision']}  "
              f"Recall {s['recall']}")
        print(f"  mean IoU per image {s['mean_image_iou']}")
        print(f"  time per image: mean {s['mean_seconds_per_image']} s, "
              f"max {s['max_seconds_per_image']} s")

        # Best and worst cases for error analysis
        ranked = sorted(rows, key=lambda r: r["iou"])
        for rank, r in enumerate(ranked[:N_EXAMPLES], 1):
            save_example(r["_pair"], f"worst_{rank}", r["iou"])
        for rank, r in enumerate(ranked[-N_EXAMPLES:], 1):
            save_example(r["_pair"], f"best_{rank}", r["iou"])

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved outputs to {OUT_DIR}/")


if __name__ == "__main__":
    main()