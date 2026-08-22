import contextlib
import os
import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .augment import build_adaptive_views


def _autocast(enabled):
    if enabled:
        return torch.amp.autocast("cuda")
    return _null_context()


@contextlib.contextmanager
def _null_context():
    """Python 3.6-compatible replacement for contextlib.nullcontext."""
    yield


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
    balancer,
    amp=False,
    log_interval=20,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    last_weights = [1.0 / balancer.num_losses] * balancer.num_losses
    start = time.time()

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()

        with _autocast(amp):
            original_logits, response_maps = model(images, return_attention=True)
            original_loss = criterion(original_logits, targets)
            with torch.no_grad():
                views, _ = build_adaptive_views(images, response_maps)
            view_logits = [model(view, return_attention=False)[0] for view in views]
            losses = [original_loss] + [criterion(logits, targets) for logits in view_logits]
            last_weights = balancer.adjust_weights(
                [item.detach().item() for item in losses]
            )
            loss = sum(weight * item for weight, item in zip(last_weights, losses))

        if not torch.isfinite(original_logits).all():
            raise FloatingPointError(
                "Non-finite training logits at step {}".format(step)
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite training loss at step {}".format(step)
            )

        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.detach().item() * batch_size
        total_correct += (original_logits.argmax(1) == targets).sum().item()
        total_samples += batch_size
        if step % log_interval == 0:
            print(
                "train step {}/{} loss={:.4f} acc={:.2f}% weights={}".format(
                    step,
                    len(loader),
                    total_loss / max(total_samples, 1),
                    100.0 * total_correct / max(total_samples, 1),
                    [round(x, 3) for x in last_weights],
                )
            )
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": 100.0 * total_correct / max(total_samples, 1),
        "weights": last_weights,
        "seconds": time.time() - start,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp=False, include_details=False):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    targets_all = []
    predictions_all = []
    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with _autocast(amp):
            logits, _ = model(images, return_attention=False)
            loss = criterion(logits, targets)
        if not torch.isfinite(logits).all():
            raise FloatingPointError(
                "Non-finite evaluation logits at step {}".format(step)
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite evaluation loss at step {}".format(step)
            )
        predictions = logits.argmax(1)
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        targets_all.extend(targets.cpu().tolist())
        predictions_all.extend(predictions.cpu().tolist())

    truth = np.asarray(targets_all)
    prediction = np.asarray(predictions_all)
    result = {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": 100.0 * accuracy_score(truth, prediction),
        "macro_precision": 100.0 * precision_score(truth, prediction, average="macro", zero_division=0),
        "macro_recall": 100.0 * recall_score(truth, prediction, average="macro", zero_division=0),
        "macro_f1": 100.0 * f1_score(truth, prediction, average="macro", zero_division=0),
        "weighted_f1": 100.0 * f1_score(truth, prediction, average="weighted", zero_division=0),
        "balanced_accuracy": 100.0 * balanced_accuracy_score(truth, prediction),
    }
    if include_details:
        dataset = loader.dataset
        classes = list(dataset.classes)
        labels = np.arange(len(classes))
        matrix = confusion_matrix(truth, prediction, labels=labels)
        per_class_recall = recall_score(
            truth,
            prediction,
            labels=labels,
            average=None,
            zero_division=0,
        )
        supports = np.bincount(truth, minlength=len(classes))
        samples = list(dataset.samples)
        if len(samples) != len(truth):
            raise RuntimeError("Detailed evaluation requires an unshuffled ImageFolder")
        result["class_to_idx"] = dict(dataset.class_to_idx)
        result["confusion_matrix"] = matrix.tolist()
        result["per_class"] = {
            class_name: {
                "index": index,
                "recall": 100.0 * float(per_class_recall[index]),
                "support": int(supports[index]),
            }
            for index, class_name in enumerate(classes)
        }
        result["predictions"] = [
            {
                "path": os.path.relpath(samples[index][0], dataset.root),
                "target_index": int(truth[index]),
                "target_class": classes[int(truth[index])],
                "prediction_index": int(prediction[index]),
                "prediction_class": classes[int(prediction[index])],
                "correct": bool(truth[index] == prediction[index]),
            }
            for index in range(len(truth))
        ]
    return result
