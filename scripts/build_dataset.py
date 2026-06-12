"""Consolidate the 4-annotator labelImg data into a clean YOLO dataset.

- Maps every label by NAME through its annotator's classes.txt to a canonical
  scheme: transformer=0, wire=1 (front_transformer/transform -> transformer).
- Drops degenerate (zero-area) boxes.
- Applies the CLAHE preprocessing (thermal.preprocess) to every image so the
  detector trains on the same representation it sees at inference.
- Splits into train/valid (default 80/20, seeded) with no image in both.
- Writes YOLO layout + data.yaml, then re-validates the output.

Usage:
    python scripts/build_dataset.py [--out DIR] [--val-frac 0.2] [--seed 42] [--no-preprocess]
"""
import argparse
import os
import glob
import random
import shutil

import cv2

from thermal.preprocess import preprocess

ROOT = "/Volumes/dronisight/thermal"
IMG_EXTS = (".jpg", ".jpeg", ".JPG", ".JPEG")

ANNOTATORS = {
    "MEM1": f"{ROOT}/JUNE 11 MEM 1/_T_collected",
    "MEM2": f"{ROOT}/Jun 11 Mem 2 ",
    "MEM3": f"{ROOT}/JUNE 11 MEM 3/_T_collected",
    "MEM4": f"{ROOT}/JUNE 11 MEM 4",
}

# Canonical classes (order = id). data.yaml names must match this.
CANONICAL = ["transformer", "wire"]
CANON_ID = {name: i for i, name in enumerate(CANONICAL)}
# Map every raw annotator label name -> canonical name.
NAME_REMAP = {
    "transformer": "transformer",
    "wire": "wire",
    "front_transformer": "transformer",  # stray view-specific label -> transformer
    "transform": "transformer",           # typo of transformer
}


def load_classes(label_dir):
    with open(os.path.join(label_dir, "classes.txt")) as f:
        return [ln.strip() for ln in f if ln.strip()]


def index_images(top_dir):
    idx = {}
    for dirpath, _, files in os.walk(top_dir):
        for fn in files:
            if fn.startswith("._"):
                continue
            stem, ext = os.path.splitext(fn)
            if ext in IMG_EXTS and stem not in idx:
                idx[stem] = os.path.join(dirpath, fn)
    return idx


def collect_samples():
    """Return list of (stem, image_path, [canonical_yolo_rows]) and stats."""
    samples = []
    stats = {"dropped_zero_area": 0, "remapped_extra": 0, "skipped_no_image": 0}
    seen_stems = {}
    for ann, label_dir in ANNOTATORS.items():
        classes = load_classes(label_dir)
        top = os.path.dirname(label_dir) if os.path.basename(label_dir) == "_T_collected" else label_dir
        images = index_images(top)
        for txt in glob.glob(os.path.join(label_dir, "*.txt")):
            if os.path.basename(txt) == "classes.txt":
                continue
            orig_stem = os.path.splitext(os.path.basename(txt))[0]
            if orig_stem not in images:
                stats["skipped_no_image"] += 1
                continue
            img_path = images[orig_stem]
            rows_out = []
            with open(txt) as f:
                for ln in f:
                    parts = ln.split()
                    if len(parts) != 5:
                        continue
                    cid = int(parts[0])
                    cx, cy, w, h = (float(x) for x in parts[1:])
                    if w <= 0 or h <= 0:
                        stats["dropped_zero_area"] += 1
                        continue
                    raw_name = classes[cid]
                    canon = NAME_REMAP[raw_name]
                    if raw_name not in ("transformer", "wire"):
                        stats["remapped_extra"] += 1
                    rows_out.append(f"{CANON_ID[canon]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            if not rows_out:
                continue
            out_stem = orig_stem
            if out_stem in seen_stems:
                # Overlap analysis showed 0; guard anyway by disambiguating.
                out_stem = f"{orig_stem}__{ann}"
            seen_stems[out_stem] = ann
            samples.append((out_stem, img_path, rows_out))
    return samples, stats


def write_split(samples, out, val_frac, seed, do_preprocess):
    rng = random.Random(seed)
    rng.shuffle(samples)
    n_val = int(len(samples) * val_frac)
    splits = {"valid": samples[:n_val], "train": samples[n_val:]}

    for split in ("train", "valid"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(out, split, sub), exist_ok=True)

    freq = {0: 0, 1: 0}
    for split, items in splits.items():
        for i, (stem, img_path, rows) in enumerate(items):
            dst_img = os.path.join(out, split, "images", f"{stem}.jpg")
            dst_lbl = os.path.join(out, split, "labels", f"{stem}.txt")
            if do_preprocess:
                bgr = cv2.imread(img_path)
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                proc = cv2.cvtColor(preprocess(rgb), cv2.COLOR_RGB2BGR)
                cv2.imwrite(dst_img, proc)
            else:
                shutil.copy(img_path, dst_img)
            with open(dst_lbl, "w") as f:
                f.write("\n".join(rows) + "\n")
            for r in rows:
                freq[int(r.split()[0])] += 1
            if (i + 1) % 200 == 0:
                print(f"  {split}: {i + 1}/{len(items)}")
        print(f"  {split}: {len(items)} images written")

    with open(os.path.join(out, "data.yaml"), "w") as f:
        f.write(
            f"path: {out}\n"
            "train: train/images\n"
            "val: valid/images\n"
            f"nc: {len(CANONICAL)}\n"
            f"names: {CANONICAL}\n"
        )
    return splits, freq


def validate(out):
    """Re-parse the written labels to confirm only canonical ids and valid coords."""
    bad = 0
    ids = {0: 0, 1: 0}
    for split in ("train", "valid"):
        for lbl in glob.glob(os.path.join(out, split, "labels", "*.txt")):
            img = lbl.replace("/labels/", "/images/").replace(".txt", ".jpg")
            if not os.path.exists(img):
                bad += 1
            with open(lbl) as f:
                for ln in f:
                    p = ln.split()
                    if len(p) != 5:
                        bad += 1
                        continue
                    cid = int(p[0])
                    if cid not in (0, 1):
                        bad += 1
                    else:
                        ids[cid] += 1
                    if not all(0 <= float(x) <= 1 for x in p[1:]):
                        bad += 1
    return bad, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/YOLO_thermal")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-preprocess", action="store_true")
    args = ap.parse_args()

    print("Collecting + remapping labels ...")
    samples, stats = collect_samples()
    print(f"  usable labeled images: {len(samples)}")
    print(f"  remapped extra-class boxes -> transformer: {stats['remapped_extra']}")
    print(f"  dropped zero-area boxes: {stats['dropped_zero_area']}")
    print(f"  labels skipped (no image): {stats['skipped_no_image']}")

    print(f"Writing dataset to {args.out} (preprocess={not args.no_preprocess}) ...")
    splits, freq = write_split(samples, args.out, args.val_frac, args.seed,
                               not args.no_preprocess)

    bad, ids = validate(args.out)
    print("\n=== RESULT ===")
    print(f"  train images: {len(splits['train'])}   valid images: {len(splits['valid'])}")
    print(f"  class boxes (written): transformer(0)={ids[0]}  wire(1)={ids[1]}")
    print(f"  validation problems: {bad}")
    print(f"  data.yaml -> {os.path.join(args.out, 'data.yaml')}")


if __name__ == "__main__":
    main()
