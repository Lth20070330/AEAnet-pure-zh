"""A small synthetic forward/backward check; it does not access any dataset."""

import torch
import torch.nn as nn

from aeanet.augment import build_adaptive_views
from aeanet.loss_balancer import ParetoLossAdjuster
from aeanet.model import AEANet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Exercise the complete AEANet path with a reduced backbone so this check is
    # quick and does not require the memory needed by four ConvNeXt-Base passes.
    model = AEANet(
        num_classes=23,
        depths=(1, 1, 1, 1),
        dims=(16, 32, 64, 128),
        attention_channels=4,
        drop_path_rate=0.0,
    ).to(device)
    images = torch.randn(2, 3, 64, 64, device=device)
    targets = torch.tensor([0, 1], device=device)
    logits, response_maps = model(images)
    views, metadata = build_adaptive_views(images, response_maps)
    all_logits = [logits] + [model(view, return_attention=False)[0] for view in views]
    criterion = nn.CrossEntropyLoss()
    losses = [criterion(item, targets) for item in all_logits]
    balancer = ParetoLossAdjuster()
    weights = balancer.adjust_weights([item.item() for item in losses])
    total = sum(weight * loss for weight, loss in zip(weights, losses))
    total.backward()
    print("logits", tuple(logits.shape))
    print("response_maps", tuple(response_maps.shape))
    print("sampled_channels", metadata["sampled_channels"].cpu().tolist())
    print("weights", weights)
    print("loss", total.item())


if __name__ == "__main__":
    main()
