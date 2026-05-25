import os
import cv2
import torch
import numpy as np
import argparse
import pathlib

from utils import util
import options.options as option
from models import create_model

import torch
from PIL import Image
import torchvision.transforms as transforms


def predict(image_str: str):

    image: np.ndarray = cv2.imread(image_str)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opt",
        type=str,
        default="./options/test.yml",
        help="Path to option YAML file.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch"],
        default="pytorch",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=False)

    opt["dist"] = False

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define a transform to convert
    # the image to torch tensor
    transform = transforms.Compose([transforms.ToTensor()])

    # Convert the image to Torch tensor
    tensor_image = transform(image).unsqueeze(0)

    # Model & Metrics
    model = create_model(opt)
    para = {}
    for k, v in torch.load(opt["path"]["pretrain_model_G"]).items():
        k = "module." + k
        para[k] = v
    model.netG.load_state_dict(para, strict=True)

    model.netG.eval()

    output = util.single_forward(model.netG, tensor_image)

    np_image = util.tensor2img(output)

    np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

    cv2.imshow("Image Preview", np_image)
    cv2.waitKey()

    result_dir = opt["path"]["results_root"]

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    image_name = pathlib.Path(image_str).name

    saved_image_path = f"{result_dir}{image_name}"

    cv2.imwrite(saved_image_path, np_image)

    print(f"{saved_image_path} is saved")

    return np_image


if "__main__":

    predict("./input/3540289a59d449e080d5ea93d0538c17-ezsam.png")
