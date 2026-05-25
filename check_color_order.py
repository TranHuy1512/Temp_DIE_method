import argparse
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch

from models.archs.EnhanceN_arch import SeeInDark


PATCH_SIZE = 256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--outdir", default=Path("color_order_check"), type=Path)
    parser.add_argument("--clean", default=None, type=Path, help="Optional clean/ground-truth image")
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def select_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint, device):
    state_dict = torch.load(checkpoint, map_location=device)

    clean_state_dict = OrderedDict()
    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith("module.") else key
        clean_state_dict[clean_key] = value

    model = SeeInDark().to(device)
    model.load_state_dict(clean_state_dict, strict=True)
    model.eval()
    return model


def run_patches(model, image, device, binary):
    height, width = image.shape[:2]

    padded_height = ((height + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
    padded_width = ((width + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE

    padded = np.ones((padded_height, padded_width, 3), dtype=np.float32)
    padded[:height, :width] = image

    output = np.empty_like(padded)

    with torch.no_grad():
        for top in range(0, padded_height, PATCH_SIZE):
            for left in range(0, padded_width, PATCH_SIZE):
                patch = padded[top:top + PATCH_SIZE, left:left + PATCH_SIZE]

                tensor = torch.from_numpy(
                    patch.transpose(2, 0, 1)
                ).unsqueeze(0).to(device)

                prediction = model(tensor)[0][0].permute(1, 2, 0).cpu().numpy()

                output[top:top + PATCH_SIZE, left:left + PATCH_SIZE] = prediction

    output = output[:height, :width]

    if binary:
        output = output > 0.95

    return np.clip(output * 255.0, 0, 255).round().astype(np.uint8)


def read_input(path, color_order):
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {path}")

    if color_order == "rgb":
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    elif color_order == "bgr":
        image = image_bgr
    else:
        raise ValueError(color_order)

    return image.astype(np.float32) / 255.0


def save_output(result, output_path, color_order):
    """
    Nếu model chạy RGB, output là RGB nên cần RGB -> BGR trước khi cv2.imwrite.
    Nếu model chạy BGR, output là BGR nên lưu trực tiếp.
    """
    if color_order == "rgb":
        result_to_save = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    else:
        result_to_save = result

    if not cv2.imwrite(str(output_path), result_to_save):
        raise OSError(f"Could not write output image: {output_path}")


def compute_metrics(pred_bgr, clean_bgr):
    if pred_bgr.shape != clean_bgr.shape:
        clean_bgr = cv2.resize(clean_bgr, (pred_bgr.shape[1], pred_bgr.shape[0]))

    pred = pred_bgr.astype(np.float32)
    clean = clean_bgr.astype(np.float32)

    mae = np.mean(np.abs(pred - clean))
    mse = np.mean((pred - clean) ** 2)

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10 * np.log10((255.0 ** 2) / mse)

    return mae, mse, psnr


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    model = load_model(args.checkpoint, device)

    print(f"Using device: {device}")

    results = {}

    for color_order in ["rgb", "bgr"]:
        image = read_input(args.input, color_order)
        result = run_patches(model, image, device, args.binary)

        output_path = args.outdir / f"output_{color_order}.png"
        save_output(result, output_path, color_order)

        results[color_order] = cv2.imread(str(output_path), cv2.IMREAD_COLOR)

        print(f"Saved {color_order.upper()} result: {output_path}")

    diff = cv2.absdiff(results["rgb"], results["bgr"])
    diff_vis = np.clip(diff * 4, 0, 255).astype(np.uint8)
    diff_path = args.outdir / "diff_rgb_vs_bgr_x4.png"
    cv2.imwrite(str(diff_path), diff_vis)

    mean_diff = float(np.mean(diff))
    max_diff = int(np.max(diff))

    print()
    print("RGB vs BGR output difference:")
    print(f"  mean diff: {mean_diff:.4f}")
    print(f"  max diff : {max_diff}")
    print(f"  diff image saved: {diff_path}")

    if args.clean is not None:
        clean_bgr = cv2.imread(str(args.clean), cv2.IMREAD_COLOR)
        if clean_bgr is None:
            raise ValueError(f"Could not read clean image: {args.clean}")

        print()
        print("Metrics against clean/ground-truth image:")

        scores = {}

        for color_order in ["rgb", "bgr"]:
            mae, mse, psnr = compute_metrics(results[color_order], clean_bgr)
            scores[color_order] = psnr

            print(f"  {color_order.upper()}:")
            print(f"    MAE : {mae:.4f}")
            print(f"    MSE : {mse:.4f}")
            print(f"    PSNR: {psnr:.4f}")

        best = max(scores, key=scores.get)
        print()
        print(f"Suggested color order by PSNR: {best.upper()}")
    else:
        print()
        print("No clean/ground-truth image provided.")
        print("Please compare output_rgb.png and output_bgr.png visually.")
        print("If one matches the original repo/test behavior better, use that color order.")


if __name__ == "__main__":
    main()