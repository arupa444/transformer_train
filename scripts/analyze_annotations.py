"""Forensic analysis of the 4-annotator labelImg dataset on /Volumes/dronisight/thermal.

Maps every YOLO .txt label through its OWN annotator's classes.txt (so class ids
are interpreted by name, never blindly), then reports class frequency, label
health, and cross-annotator overlap. Read-only — touches nothing.
"""
import os
import glob
from collections import defaultdict, Counter

ROOT = "/Volumes/dronisight/thermal"
IMG_EXTS = (".jpg", ".jpeg", ".JPG", ".JPEG")

# Annotator label dir = the directory that contains its classes.txt.
ANNOTATORS = {
    "MEM1": f"{ROOT}/JUNE 11 MEM 1/_T_collected",
    "MEM2": f"{ROOT}/Jun 11 Mem 2 ",
    "MEM3": f"{ROOT}/JUNE 11 MEM 3/_T_collected",
    "MEM4": f"{ROOT}/JUNE 11 MEM 4",
}


def load_classes(label_dir):
    path = os.path.join(label_dir, "classes.txt")
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def index_images(top_dir):
    """basename(without ext) -> True for every image under an annotator folder."""
    found = set()
    for dirpath, _, files in os.walk(top_dir):
        for fn in files:
            if fn.startswith("._"):
                continue
            stem, ext = os.path.splitext(fn)
            if ext in IMG_EXTS:
                found.add(stem)
    return found


def analyze():
    raw_freq = Counter()                 # class NAME -> box count (all annotators)
    per_ann_freq = defaultdict(Counter)  # annotator -> Counter(name -> boxes)
    labeled_by = defaultdict(set)        # image stem -> {annotators with >=1 box}
    issues = defaultdict(list)           # issue type -> [details]
    n_label_files = 0
    n_empty = 0
    n_boxes = 0

    for ann, label_dir in ANNOTATORS.items():
        classes = load_classes(label_dir)
        # image stems live under the annotator's TOP folder (parent of _T_collected)
        top = label_dir
        if os.path.basename(label_dir) == "_T_collected":
            top = os.path.dirname(label_dir)
        images = index_images(top)

        txts = [p for p in glob.glob(os.path.join(label_dir, "*.txt"))
                if os.path.basename(p) != "classes.txt"]
        for txt in txts:
            n_label_files += 1
            stem = os.path.splitext(os.path.basename(txt))[0]
            if stem not in images:
                issues["label_without_image"].append(f"{ann}: {stem}")
            with open(txt) as f:
                rows = [ln.strip() for ln in f if ln.strip()]
            if not rows:
                n_empty += 1
                issues["empty_label"].append(f"{ann}: {stem}")
                continue
            has_box = False
            for i, row in enumerate(rows):
                parts = row.split()
                if len(parts) != 5:
                    issues["malformed_row"].append(f"{ann}/{stem}:{i} -> '{row}'")
                    continue
                try:
                    cid = int(parts[0])
                    cx, cy, w, h = (float(x) for x in parts[1:])
                except ValueError:
                    issues["unparseable_row"].append(f"{ann}/{stem}:{i} -> '{row}'")
                    continue
                if cid < 0 or cid >= len(classes):
                    issues["class_id_out_of_range"].append(
                        f"{ann}/{stem}:{i} -> id {cid} (only {len(classes)} classes)")
                    continue
                name = classes[cid]
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    issues["coords_out_of_range"].append(f"{ann}/{stem}:{i} -> {parts[1:]}")
                if w <= 0 or h <= 0:
                    issues["zero_area_box"].append(f"{ann}/{stem}:{i} -> w={w} h={h}")
                raw_freq[name] += 1
                per_ann_freq[ann][name] += 1
                n_boxes += 1
                has_box = True
            if has_box:
                labeled_by[stem].add(ann)

    # Cross-annotator overlap: same image labeled by >1 annotator.
    overlap = {stem: anns for stem, anns in labeled_by.items() if len(anns) > 1}

    # ---- Report ----
    print("=" * 64)
    print("CLASS FREQUENCY (raw, by name as written by each annotator)")
    print("=" * 64)
    for name, c in raw_freq.most_common():
        print(f"  {name:<18} {c:>6} boxes")
    print(f"  {'TOTAL':<18} {n_boxes:>6} boxes across {n_label_files} label files")

    print("\nPER-ANNOTATOR")
    for ann in ANNOTATORS:
        cls = load_classes(ANNOTATORS[ann])
        imgs = len([s for s in labeled_by if ann in labeled_by[s]])
        print(f"  {ann}: classes={cls}")
        print(f"        labeled images={imgs}  " +
              "  ".join(f"{k}={v}" for k, v in per_ann_freq[ann].most_common()))

    print("\nUNIQUE LABELED IMAGES:", len(labeled_by))
    print("CROSS-ANNOTATOR OVERLAP (same image labeled by >1):", len(overlap))
    for stem, anns in list(overlap.items())[:10]:
        print(f"   {stem}: {sorted(anns)}")
    if len(overlap) > 10:
        print(f"   ... and {len(overlap) - 10} more")

    print("\nLABEL HEALTH ISSUES")
    if not any(issues.values()):
        print("  none")
    for k, v in issues.items():
        print(f"  {k}: {len(v)}")
        for d in v[:8]:
            print(f"     - {d}")
        if len(v) > 8:
            print(f"     ... and {len(v) - 8} more")
    print(f"  empty_label_files: {n_empty}")


if __name__ == "__main__":
    analyze()
