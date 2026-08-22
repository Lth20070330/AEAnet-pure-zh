import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from aeanet.data import build_datasets
from aeanet.engine import evaluate
from aeanet.model import convnext_base_aeanet


def main():
    parser = argparse.ArgumentParser("Evaluate AEANet")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {})
    input_size = saved_args.get("input_size", 224)
    _, _, test_dataset = build_datasets(args.data_root, input_size)
    model = convnext_base_aeanet(
        num_classes=len(test_dataset.classes),
        attention_channels=saved_args.get("attention_channels", 16),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    metrics = evaluate(
        model,
        loader,
        nn.CrossEntropyLoss(),
        device,
        amp=use_amp,
        include_details=bool(args.output),
    )
    if args.output:
        output = os.path.abspath(args.output)
        output_directory = os.path.dirname(output)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
        with open(output, "w") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
        summary = {
            key: value
            for key, value in metrics.items()
            if key not in ("confusion_matrix", "per_class", "predictions")
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        print("detailed evaluation written to {}".format(output))
    else:
        print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
