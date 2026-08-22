import random

import torch
import torch.nn.functional as F


def sample_three_response_maps(response_maps, eps=1e-12):
    """Probability-proportional sampling without replacement for each sample."""
    if response_maps.size(1) < 3:
        raise ValueError("At least three response-map channels are required")
    positive = response_maps.detach().clamp_min(0)
    strengths = torch.sqrt(positive.sum(dim=(2, 3)) + eps)
    sums = strengths.sum(dim=1, keepdim=True)
    uniform = torch.full_like(strengths, 1.0 / strengths.size(1))
    probabilities = torch.where(sums > eps, strengths / sums.clamp_min(eps), uniform)
    sampled = torch.multinomial(probabilities, 3, replacement=False)
    gather_index = sampled[:, :, None, None].expand(
        -1, -1, response_maps.size(2), response_maps.size(3)
    )
    selected = torch.gather(response_maps.detach(), 1, gather_index)
    return selected, sampled, probabilities


def crop_and_enlarge(images, attention, theta_range=(0.4, 0.6), padding=0.1):
    batch, _, height, width = images.shape
    attention = F.interpolate(attention, (height, width), mode="bilinear", align_corners=False)
    outputs = []
    for i in range(batch):
        threshold = random.uniform(*theta_range) * attention[i, 0].max()
        coordinates = torch.nonzero(attention[i, 0] >= threshold, as_tuple=False)
        if coordinates.numel() == 0:
            outputs.append(images[i : i + 1])
            continue
        pad_h, pad_w = int(height * padding), int(width * padding)
        y1 = max(int(coordinates[:, 0].min().item()) - pad_h, 0)
        y2 = min(int(coordinates[:, 0].max().item()) + pad_h + 1, height)
        x1 = max(int(coordinates[:, 1].min().item()) - pad_w, 0)
        x2 = min(int(coordinates[:, 1].max().item()) + pad_w + 1, width)
        crop = images[i : i + 1, :, y1:y2, x1:x2]
        outputs.append(F.interpolate(crop, (height, width), mode="bilinear", align_corners=False))
    return torch.cat(outputs, dim=0)


def local_horizontal_flip(images, attention, size_range=(0.05, 0.15)):
    batch, _, height, width = images.shape
    attention = F.interpolate(attention, (height, width), mode="bilinear", align_corners=False)
    output = images.clone()
    for i in range(batch):
        index = int(attention[i, 0].argmax().item())
        center_y, center_x = index // width, index % width
        region_h = max(int(random.uniform(*size_range) * height), 1)
        region_w = max(int(random.uniform(*size_range) * width), 1)
        y1 = max(center_y - region_h // 2, 0)
        x1 = max(center_x - region_w // 2, 0)
        y2 = min(y1 + region_h, height)
        x2 = min(x1 + region_w, width)
        output[i : i + 1, :, y1:y2, x1:x2] = torch.flip(
            output[i : i + 1, :, y1:y2, x1:x2], dims=(3,)
        )
    return output


def mask_high_response(images, attention, theta_range=(0.2, 0.5)):
    height, width = images.shape[-2:]
    attention = F.interpolate(attention, (height, width), mode="bilinear", align_corners=False)
    masks = []
    for i in range(images.size(0)):
        threshold = random.uniform(*theta_range) * attention[i, 0].max()
        masks.append((attention[i : i + 1] < threshold).to(images.dtype))
    return images * torch.cat(masks, dim=0)


def build_adaptive_views(
    images,
    response_maps,
    enable_enlarge=True,
    enable_flip=True,
    enable_mask=True,
):
    selected, indices, probabilities = sample_three_response_maps(response_maps)
    views = []
    names = []
    if enable_enlarge:
        views.append(crop_and_enlarge(images, selected[:, 0:1]))
        names.append("enlarge")
    if enable_flip:
        views.append(local_horizontal_flip(images, selected[:, 1:2]))
        names.append("flip")
    if enable_mask:
        views.append(mask_high_response(images, selected[:, 2:3]))
        names.append("mask")
    return tuple(views), {
        "sampled_channels": indices,
        "channel_probabilities": probabilities,
        "view_names": names,
    }
