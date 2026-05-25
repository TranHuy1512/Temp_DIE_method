import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create DocNLC input|ground-truth file lists from split directories."
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Dataset root containing train/val/test directories.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("./filelists"),
        type=Path,
        help="Directory where split .txt files are written.",
    )
    parser.add_argument(
        "--splits",
        default=("train", "val", "test"),
        nargs="+",
        help="Splits to process.",
    )
    return parser.parse_args()


def collect_images(directory):
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def relative_stem(path, parent):
    return path.relative_to(parent).with_suffix("").as_posix()


def create_pairs(root, split):
    input_dir = root / split / "imgs"
    target_dir = root / split / "gt_imgs"
    if not input_dir.is_dir() or not target_dir.is_dir():
        raise FileNotFoundError(
            "Expected both '{}' and '{}'.".format(input_dir, target_dir)
        )

    inputs = collect_images(input_dir)
    targets = collect_images(target_dir)
    target_by_stem = {}
    for target in targets:
        key = relative_stem(target, target_dir)
        if key in target_by_stem:
            raise ValueError("Duplicate ground-truth stem in {}: {}".format(split, key))
        target_by_stem[key] = target

    pairs = []
    missing = []
    for image in inputs:
        key = relative_stem(image, input_dir)
        target = target_by_stem.pop(key, None)
        if target is None:
            missing.append(image)
        else:
            pairs.append((image.resolve(), target.resolve()))

    if missing or target_by_stem:
        message = ["Could not pair every image in split '{}' by relative filename stem.".format(split)]
        if missing:
            message.append("Missing GT for: {}".format(", ".join(str(path) for path in missing[:5])))
        if target_by_stem:
            message.append("Missing input for: {}".format(", ".join(target_by_stem.keys())))
        raise ValueError("\n".join(message))
    if not pairs:
        raise ValueError("No image pairs found in split '{}'.".format(split))
    return pairs


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        pairs = create_pairs(args.root, split)
        filelist = args.output_dir / "{}.txt".format(split)
        with filelist.open("w") as output:
            for image, target in pairs:
                output.write("{}|{}\n".format(image, target))
        print("Wrote {} pairs to {}.".format(len(pairs), filelist))


if __name__ == "__main__":
    main()
