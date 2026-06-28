import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Directory containing images/ and labels/")
    parser.add_argument("output", type=Path, help="Destination dataset directory")
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.2, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=701)
    return parser.parse_args()


def split_dataset(source, output, ratios, seed):
    if any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("Split ratios must be non-negative and sum to 1")

    image_dir, label_dir = source / "images", source / "labels"
    images = sorted(image_dir.glob("*.tif"))
    if not images:
        raise ValueError(f"No .tif images found in {image_dir}")
    missing = [path.name for path in images if not (label_dir / path.name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} labels; first: {missing[0]}")

    # Keep all augmented variants of one source image in the same split. A
    # file-level shuffle would leak near-identical samples into validation/test.
    variants = defaultdict(list)
    for path in images:
        source_id = path.stem.rsplit("_", 1)[0]
        variants[source_id].append(path)

    source_ids = sorted(variants)
    random.Random(seed).shuffle(source_ids)
    train_end = int(len(source_ids) * ratios[0])
    val_end = train_end + int(len(source_ids) * ratios[1])
    split_ids = {
        "train": source_ids[:train_end],
        "val": source_ids[train_end:val_end],
        "test": source_ids[val_end:],
    }
    groups = {
        split: [path for source_id in ids for path in variants[source_id]]
        for split, ids in split_ids.items()
    }

    for split, paths in groups.items():
        for kind in ("images", "labels"):
            (output / split / kind).mkdir(parents=True, exist_ok=True)
        for image_path in paths:
            shutil.copy2(image_path, output / split / "images" / image_path.name)
            shutil.copy2(label_dir / image_path.name, output / split / "labels" / image_path.name)
        print(f"{split}: {len(paths)}")


if __name__ == "__main__":
    args = parse_args()
    split_dataset(args.source, args.output, args.ratios, args.seed)
