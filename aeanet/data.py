import os
import random

from PIL import Image, ImageOps
from torch.utils.data import Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


class ThesisResizeOrPad(object):
    """Resize large images and zero-pad genuinely small images, as in the thesis."""

    def __init__(self, input_size, resize_size):
        self.input_size = input_size
        self.resize_size = resize_size

    def __call__(self, image):
        width, height = image.size
        if width > self.input_size or height > self.input_size:
            return TF.resize(image, self.resize_size, interpolation=Image.BICUBIC)

        pad_width = self.input_size - width
        pad_height = self.input_size - height
        left = pad_width // 2
        top = pad_height // 2
        return ImageOps.expand(
            image,
            border=(left, top, pad_width - left, pad_height - top),
            fill=0,
        )


def _validate_nonempty_images(root):
    image_count = 0
    empty_examples = []
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue
            image_count += 1
            path = os.path.join(directory, filename)
            if os.path.getsize(path) == 0 and len(empty_examples) < 5:
                empty_examples.append(path)
    if image_count == 0:
        raise RuntimeError("No image files were found under {}".format(root))
    if empty_examples:
        raise RuntimeError(
            "Zero-byte image placeholders were found; restore the real dataset first. "
            "Examples: {}".format(", ".join(empty_examples))
        )


class ImageFolderSubset(Subset):
    """Subset that preserves ImageFolder metadata used by the training code."""

    @property
    def classes(self):
        return self.dataset.classes

    @property
    def class_to_idx(self):
        return self.dataset.class_to_idx


def _stratified_split_indices(targets, val_ratio, split_seed):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")

    by_class = {}
    for index, target in enumerate(targets):
        by_class.setdefault(target, []).append(index)

    train_indices = []
    val_indices = []
    rng = random.Random(split_seed)
    for target in sorted(by_class):
        indices = list(by_class[target])
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * val_ratio)))
        val_count = min(val_count, len(indices) - 1)
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    train_indices.sort()
    val_indices.sort()
    return train_indices, val_indices


def build_datasets(
    data_root,
    input_size=224,
    val_ratio=0.1,
    split_seed=2026,
):
    data_root = os.path.abspath(data_root)
    train_root = os.path.join(data_root, "train")
    val_root = os.path.join(data_root, "val")
    test_root = os.path.join(data_root, "test")
    if not os.path.isdir(train_root) or not os.path.isdir(test_root):
        raise FileNotFoundError(
            "Expected train/ and test/ directories under {}".format(data_root)
        )
    _validate_nonempty_images(train_root)
    _validate_nonempty_images(test_root)

    resize_size = int(round(input_size / (224.0 / 256.0)))
    train_transform = transforms.Compose(
        [
            ThesisResizeOrPad(input_size, resize_size),
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            ThesisResizeOrPad(input_size, resize_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train_source = datasets.ImageFolder(train_root, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_root, transform=test_transform)
    if train_source.class_to_idx != test_dataset.class_to_idx:
        raise ValueError("Train and test class mappings are different")

    if os.path.isdir(val_root):
        _validate_nonempty_images(val_root)
        val_dataset = datasets.ImageFolder(val_root, transform=test_transform)
        if train_source.class_to_idx != val_dataset.class_to_idx:
            raise ValueError("Train and validation class mappings are different")
        return train_source, val_dataset, test_dataset

    val_source = datasets.ImageFolder(train_root, transform=test_transform)
    train_indices, val_indices = _stratified_split_indices(
        train_source.targets,
        val_ratio=val_ratio,
        split_seed=split_seed,
    )
    train_dataset = ImageFolderSubset(train_source, train_indices)
    val_dataset = ImageFolderSubset(val_source, val_indices)
    return train_dataset, val_dataset, test_dataset
