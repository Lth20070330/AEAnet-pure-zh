import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from aeanet.data import build_datasets
from aeanet.engine import evaluate, train_one_epoch
from aeanet.loss_balancer import ParetoLossAdjuster
from aeanet.model import convnext_base_aeanet


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="")
    known, _ = pre_parser.parse_known_args()
    config = {}
    if known.config:
        with open(known.config, "r", encoding="utf-8") as handle:
            config = json.load(handle)

    parser = argparse.ArgumentParser(
        "Train the complete AEA-Net model", parents=[pre_parser]
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--dataset-name", default="custom_dataset")
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--output-dir", default="runs/seed0")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--attention-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=4e-3)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument(
        "--selection-metric",
        choices=("accuracy", "macro_f1", "balanced_accuracy"),
        default="macro_f1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", default="")
    parser.set_defaults(**config)
    args = parser.parse_args()
    valid_keys = {action.dest for action in parser._actions}
    unknown_keys = sorted(set(config) - valid_keys)
    if unknown_keys:
        parser.error("Unknown config keys: {}".format(", ".join(unknown_keys)))
    if not args.data_root:
        parser.error("--data-root is required (or set data_root in --config)")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_learning_rate(optimizer, epoch, warmup_epochs, epochs):
    if epoch < warmup_epochs:
        scale = float(epoch + 1) / max(warmup_epochs, 1)
    else:
        progress = float(epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    values = []
    for group in optimizer.param_groups:
        value = group["base_lr"] * scale
        group["lr"] = value
        values.append(value)
    return values


def build_optimizer(model, args):
    if args.backbone_lr is None:
        return torch.optim.AdamW(
            [{"params": model.parameters(), "base_lr": args.lr}],
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )

    backbone_ids = {
        id(parameter)
        for module in (model.downsample_layers, model.stages)
        for parameter in module.parameters()
    }
    backbone_parameters = [
        parameter for parameter in model.parameters() if id(parameter) in backbone_ids
    ]
    new_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in backbone_ids
    ]
    return torch.optim.AdamW(
        [
            {"params": new_parameters, "lr": args.lr, "base_lr": args.lr},
            {
                "params": backbone_parameters,
                "lr": args.backbone_lr,
                "base_lr": args.backbone_lr,
            },
        ],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )


def make_loader(dataset, args, device, shuffle=False, generator=None, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=drop_last,
        generator=generator,
    )


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    train_dataset, val_dataset, test_dataset = build_datasets(
        args.data_root,
        args.input_size,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
    )
    with open(
        os.path.join(args.output_dir, "split_manifest.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "dataset": args.dataset_name,
                "seed": args.seed,
                "split_seed": args.split_seed,
                "val_ratio": args.val_ratio,
                "classes": train_dataset.classes,
                "train_size": len(train_dataset),
                "validation_size": len(val_dataset),
                "test_size": len(test_dataset),
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = make_loader(
        train_dataset, args, device, shuffle=True, generator=generator, drop_last=True
    )
    val_loader = make_loader(val_dataset, args, device)
    test_loader = make_loader(test_dataset, args, device)

    model = convnext_base_aeanet(
        num_classes=len(train_dataset.classes),
        attention_channels=args.attention_channels,
    )
    if args.pretrained:
        if not os.path.isfile(args.pretrained):
            raise FileNotFoundError(args.pretrained)
        missing, unexpected = model.load_convnext_pretrained(args.pretrained)
        print("pretrained missing keys:", missing)
        print("pretrained unexpected keys:", unexpected)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    balancer = ParetoLossAdjuster(num_losses=4, min_weight=0.1)
    start_epoch = 0
    best_validation_score = float("-inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        balancer.load_state_dict(checkpoint["balancer"])
        start_epoch = checkpoint["epoch"] + 1
        best_validation_score = checkpoint["best_validation_score"]

    history_path = os.path.join(args.output_dir, "metrics.jsonl")
    for epoch in range(start_epoch, args.epochs):
        learning_rates = set_learning_rate(
            optimizer, epoch, args.warmup_epochs, args.epochs
        )
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, balancer, amp=use_amp
        )
        validation_metrics = evaluate(model, val_loader, criterion, device, amp=use_amp)
        record = {
            "epoch": epoch,
            "learning_rates": learning_rates,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        print(json.dumps(record, sort_keys=True))
        with open(history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

        score = validation_metrics[args.selection_metric]
        is_best = score > best_validation_score
        best_validation_score = max(best_validation_score, score)
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "balancer": balancer.state_dict(),
            "best_validation_score": best_validation_score,
            "selection_metric": args.selection_metric,
            "args": vars(args),
            "classes": train_dataset.classes,
        }
        torch.save(state, os.path.join(args.output_dir, "checkpoint_last.pth"))
        if is_best:
            torch.save(state, os.path.join(args.output_dir, "checkpoint_best.pth"))

    best_checkpoint = torch.load(
        os.path.join(args.output_dir, "checkpoint_best.pth"),
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model"])
    test_metrics = evaluate(
        model, test_loader, criterion, device, amp=use_amp, include_details=True
    )
    with open(
        os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "best_epoch": best_checkpoint["epoch"],
                "selection_metric": args.selection_metric,
                "best_validation_score": best_checkpoint["best_validation_score"],
                "test": test_metrics,
            },
            handle,
            indent=2,
            sort_keys=True,
        )


if __name__ == "__main__":
    main()
