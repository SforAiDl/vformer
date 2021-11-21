import argparse

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision import io

from vformer.models import VanillaViT
from vformer.viz import ViTAttentionGradRollout, ViTAttentionRollout


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path", type=str, required=True, help="Input image path"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="The size of image to be resized to",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default=None,
        help="Input path where model is saved",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="attend",
        help="The name of the attention layer of the model",
    )
    parser.add_argument(
        "--head_fusion",
        type=str,
        default="mean",
        choices=["mean", "max", "min"],
        help="How to fuse the attention heads for attention rollout.\
                         Can be mean/max/min",
    )
    parser.add_argument(
        "--discard_ratio",
        type=float,
        default=0.9,
        help="How many of the lowest attention paths should we discard",
    )
    parser.add_argument(
        "--grad_rollout",
        default=False,
        action="store_true",
        help="Implement gradient rollout",
    )
    parser.add_argument(
        "--category_index",
        type=int,
        default=0,
        help="The category index for gradient rollout",
    )
    args = parser.parse_args()
    return args


def open_image(path, size=[256, 256]):
    """Opens the image in path and converts to tensor to act as input for the neural network"""
    img = io.read_image(path, io.ImageReadMode.RGB).to(torch.float32)
    img = F.resize(img, size)
    img.unsqueeze_(0)
    return img


def model_rollout(
    img,
    model,
    layer="attend",
    head_fusion="mean",
    discard_ratio=0.9,
    grad_rollout=False,
    category_index=0,
):
    """runs grad rollout on the given model"""
    if grad_rollout:
        model_attention_rollout = ViTAttentionGradRollout(model, layer, discard_ratio)
        rollout = model_attention_rollout(img, category_index)
    else:
        model_attention_rollout = ViTAttentionRollout(
            model, layer, head_fusion=head_fusion, discard_ratio=discard_ratio
        )
        rollout = model_attention_rollout(img)
    rollout = cv2.resize(rollout, (img[0].shape[1], img[0].shape[2]))
    heatmap = cv2.applyColorMap(np.uint8(255 * rollout), cv2.COLORMAP_JET)
    heatmap = heatmap.reshape((3, heatmap.shape[0], -1))
    return heatmap


def mask_over_image(img, heatmap):
    """merges the original image and the generated"""
    imgt = img[0].numpy()
    heatmap = heatmap / 255.0
    imgt = imgt / 255.0
    imgt = imgt + heatmap
    imgt = (imgt / np.max(imgt)) * 255
    return imgt


if __name__ == "__main__":
    args = get_args()
    img = open_image(args.image_path, args.image_size)
    if args.model_path == None:
        model = VanillaViT(img_size=256, patch_size=32, n_classes=10, in_channels=3)
    else:
        model = torch.load(args.model_path)
    heatmap = model_rollout(
        img,
        model,
        args.layer,
        args.head_fusion,
        args.discard_ratio,
        args.grad_rollout,
        args.category_index,
    )
    final_image = mask_over_image(img, heatmap)
    final_image = np.swapaxes(final_image, 0, 2)
    final_image = np.swapaxes(final_image, 0, 1)
    final_image = np.uint8(final_image)
    cv2.imshow("", final_image)
    cv2.waitKey(0)
