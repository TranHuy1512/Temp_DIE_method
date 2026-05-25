import argparse
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch

from models.archs.EnhanceN_arch import SeeInDark


PATCH_SIZE = 256


def parse_args():
    parser = argparse.ArgumentParser(description="Run DocNLC on a single document image.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to a generator .pth file.")
    parser.add_argument("--input", required=True, type=Path, help="Path to a degraded input image.")
    parser.add_argument("--output", required=True, type=Path, help="Path for the restored output image.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Inference device. The default selects CUDA, MPS, then CPU.",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Threshold output at 0.95, matching the original test() behavior.",
    )
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
        clean_state_dict[key[7:] if key.startswith("module.") else key] = value

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
                patch = padded[top : top + PATCH_SIZE, left : left + PATCH_SIZE]
                tensor = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).to(device)
                prediction = model(tensor)[0][0].permute(1, 2, 0).cpu().numpy()
                output[top : top + PATCH_SIZE, left : left + PATCH_SIZE] = prediction

    output = output[:height, :width]
    if binary:
        output = output > 0.95
    return np.clip(output * 255.0, 0, 255).round().astype(np.uint8)


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError("Checkpoint does not exist: {}".format(args.checkpoint))
    if not args.input.is_file():
        raise FileNotFoundError("Input image does not exist: {}".format(args.input))

    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read input image: {}".format(args.input))
    image = image.astype(np.float32) / 255.0

    device = select_device(args.device)
    model = load_model(args.checkpoint, device)
    result = run_patches(model, image, device, args.binary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), result):
        raise OSError("Could not write output image: {}".format(args.output))
    print("Saved output to {} using {}.".format(args.output, device))


if __name__ == "__main__":
    main()
