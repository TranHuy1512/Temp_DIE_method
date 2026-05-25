import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from demo import load_model, run_patches, select_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a DocNLC checkpoint on paired test images with PSNR and SSIM."
    )
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to a generator .pth file.")
    parser.add_argument(
        "--filelist",
        default=Path("./filelists/test.txt"),
        type=Path,
        help="TXT file whose lines are input_image|ground_truth_image.",
    )
    parser.add_argument(
        "--report",
        default=Path("./output/test_metrics.txt"),
        type=Path,
        help="TXT path for per-image and average metrics.",
    )
    parser.add_argument(
        "--save-images",
        type=Path,
        help="Optional directory for restored images.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Inference device. The default selects CUDA, MPS, then CPU.",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Threshold outputs at 0.95, matching the original test.py behavior.",
    )
    parser.add_argument(
        "--crop-border",
        default=0,
        type=int,
        help="Pixels excluded from every border before metric calculation.",
    )
    return parser.parse_args()


def read_pairs(filelist):
    if not filelist.is_file():
        raise FileNotFoundError("File list does not exist: {}".format(filelist))

    pairs = []
    for line_number, line in enumerate(filelist.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        paths = line.split("|")
        if len(paths) != 2:
            raise ValueError("Invalid line {} in {}: expected input|gt.".format(line_number, filelist))
        input_path, gt_path = (Path(path) for path in paths)
        if not input_path.is_file() or not gt_path.is_file():
            raise FileNotFoundError("Missing pair on line {}: {}".format(line_number, line))
        pairs.append((input_path, gt_path))

    if not pairs:
        raise ValueError("No image pairs found in {}.".format(filelist))
    return pairs


def calculate_psnr(image, target):
    error = np.mean((image.astype(np.float64) - target.astype(np.float64)) ** 2)
    if error == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(error))


def ssim_channel(image, target):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    image = image.astype(np.float64)
    target = target.astype(np.float64)
    mean_image = cv2.filter2D(image, -1, window)[5:-5, 5:-5]
    mean_target = cv2.filter2D(target, -1, window)[5:-5, 5:-5]
    image_sq = mean_image ** 2
    target_sq = mean_target ** 2
    image_target = mean_image * mean_target
    variance_image = cv2.filter2D(image ** 2, -1, window)[5:-5, 5:-5] - image_sq
    variance_target = cv2.filter2D(target ** 2, -1, window)[5:-5, 5:-5] - target_sq
    covariance = cv2.filter2D(image * target, -1, window)[5:-5, 5:-5] - image_target
    score = ((2 * image_target + c1) * (2 * covariance + c2)) / (
        (image_sq + target_sq + c1) * (variance_image + variance_target + c2)
    )
    return score.mean()


def calculate_ssim(image, target):
    if image.shape != target.shape:
        raise ValueError("Predicted and GT images must have equal dimensions.")
    if image.ndim == 2:
        return ssim_channel(image, target)
    return float(np.mean([ssim_channel(image[:, :, channel], target[:, :, channel]) for channel in range(3)]))


def crop_border(image, border):
    if border == 0:
        return image
    if border < 0 or border * 2 >= min(image.shape[:2]):
        raise ValueError("Invalid crop border {} for image shape {}.".format(border, image.shape))
    return image[border:-border, border:-border]


def predict(model, input_path, device, binary):
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read input image: {}".format(input_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    output_rgb = run_patches(model, image_rgb, device, binary)
    return cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)


def display_value(value):
    return "inf" if math.isinf(value) else "{:.6f}".format(value)


def main():
    args = parse_args()
    device = select_device(args.device)
    model = load_model(args.checkpoint, device)
    pairs = read_pairs(args.filelist)

    if args.save_images:
        args.save_images.mkdir(parents=True, exist_ok=True)
    lines = [
        "DocNLC test metrics",
        "checkpoint: {}".format(args.checkpoint),
        "filelist: {}".format(args.filelist),
        "device: {}".format(device),
        "binary: {}".format(args.binary),
        "crop_border: {}".format(args.crop_border),
        "",
        "{:<5} {:<40} {:>14} {:>14}".format("No.", "Image", "PSNR (dB)", "SSIM"),
    ]
    psnr_values = []
    ssim_values = []

    for index, (input_path, gt_path) in enumerate(pairs, start=1):
        prediction = predict(model, input_path, device, args.binary)
        target = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
        if target is None:
            raise ValueError("Could not read ground truth image: {}".format(gt_path))
        if prediction.shape != target.shape:
            raise ValueError(
                "Image shape mismatch for {}: prediction {}, GT {}.".format(
                    input_path.name, prediction.shape, target.shape
                )
            )

        prediction_metric = crop_border(prediction, args.crop_border)
        target_metric = crop_border(target, args.crop_border)
        psnr = calculate_psnr(prediction_metric, target_metric)
        ssim = calculate_ssim(prediction_metric, target_metric)
        psnr_values.append(psnr)
        ssim_values.append(ssim)

        line = "{:<5} {:<40} {:>14} {:>14.6f}".format(
            index, input_path.name, display_value(psnr), ssim
        )
        lines.append(line)
        print(line)
        if args.save_images:
            output_name = "{:04d}_{}".format(index, input_path.name)
            cv2.imwrite(str(args.save_images / output_name), prediction)

    average_psnr = float(np.mean(psnr_values))
    average_ssim = float(np.mean(ssim_values))
    summary = "Average over {} images: PSNR = {} dB, SSIM = {:.6f}".format(
        len(pairs), display_value(average_psnr), average_ssim
    )
    lines.extend(["", summary])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")
    print(summary)
    print("Saved report to {}.".format(args.report))


if __name__ == "__main__":
    main()
