import argparse

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

from vformer.models import VanillaViT
from vformer.viz import ViTAttentionGradRollout, ViTAttentionRollout


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path", type=str, required=True, help="Input image path"
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
        default="max",
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
        "--grad_rollout", type=bool, default=False, help="Implement gradient rollout"
    )
    parser.add_argument(
        "--category_index",
        type=int,
        default=0,
        help="The category index for gradient rollout",
    )
    args = parser.parse_args()
    return args


def open_image(path):
    """Opens the image in path and converts to tensor to act as input for the neural network"""
    img = Image.open(path)
    img = F.to_tensor(F.resize(img, (256, 256)))
    img.unsqueeze_(0)
    return img


def model_rollout(
    img,
    model,
    layer="attend",
    grad_rollout=False,
    head_fusion="mean",
    discard_ratio=0.9,
    category_index=0,
):
    if grad_rollout:
        model_attention_rollout = ViTAttentionGradRollout(
            model=model, layer=layer, discard_ratio=discard_ratio
        )
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


def mask_over_image():
    imgt = img[0].numpy()
    heatmap = heatmap / 255.0
    imgt = imgt / 255.0
    imgt = imgt + heatmap
    imgt = (imgt / np.max(imgt)) * 255
    imgt = torch.from_numpy(imgt)
    imgt = F.to_pil_image(imgt)


args = get_args()
img = open_image(args.image_path)
