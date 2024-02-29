"""
Utilities : 유용 기능
"""
import random

import torch
from torchvision.transforms.functional import resize, to_tensor, normalize, to_pil_image

from PIL import Image
import numpy as np


MEAN = (0.485, 0.456, 0.406)    # 평균
STD = (0.229, 0.224, 0.225) # 표준편차


def set_seed(seed=None):
    """Sets the random seed.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device(device=None):
    """Sets the device.

    by default sets to gpu.
    """
    return torch.device("cuda")


def prep_img(image, size=None, mean=MEAN, std=STD):
    """Preprocess image.
    1) load as PIl
    2) resize
    3) convert to tensor
    4) normalize
    """
    im = Image.fromarray(image.astype(np.uint8))
    size = size or im.size[::-1]
    texture = im.resize(size)
    texture_tensor = to_tensor(texture).unsqueeze(0)
    texture_tensor = normalize(texture_tensor, mean=mean, std=std)
    return texture_tensor


def denormalize(tensor: torch.Tensor, mean=MEAN, std=STD, inplace: bool = False):
    """Based on torchvision.transforms.functional.normalize.
    """
    tensor = tensor.clone() if not inplace else tensor
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def to_pil(tensor: torch.Tensor):
    """Converts tensor to PIL Image.

    Args:
        tensor (torch.Temsor): input tensor to be converted to PIL Image of torch.Size([C, H, W]).
    Returns:
        PIL Image: converted img.
    """
    img = tensor.clone().detach().cpu()
    img = denormalize(img).clip(0, 1)
    img = to_pil_image(img)
    return img


def to_img(tensor):
    """To image tensor.
    """
    img = tensor.clone().detach().cpu()
    img = denormalize(img).clip(0, 1)
    img = img.permute(1, 2, 0)
    return img
