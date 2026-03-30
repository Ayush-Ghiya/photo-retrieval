"""
CIFAR Dataset to Images Converter
Loads CIFAR-10 (or CIFAR-100) using torchvision and saves each image
to disk, organized by class label.
"""

import argparse
import os

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


def save_cifar_images(
    dataset_name: str = "cifar10",
    root: str = "./data",
    output_dir: str = "./cifar_images",
    splits: list[str] = ["train", "test"],
) -> None:
    """
    Download and convert a CIFAR dataset to individual image files.

    Directory structure created:
        <output_dir>/
            train/
                airplane/  0001.png  0002.png  ...
                automobile/ ...
                ...
            test/
                airplane/  ...
                ...

    Args:
        dataset_name: "cifar10" or "cifar100"
        root:         Directory where the raw dataset is downloaded.
        output_dir:   Root directory for the saved PNG images.
        splits:       Which splits to export ("train", "test", or both).
    """
    dataset_cls = {
        "cifar10": torchvision.datasets.CIFAR10,
        "cifar100": torchvision.datasets.CIFAR100,
    }.get(dataset_name.lower())

    if dataset_cls is None:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Choose 'cifar10' or 'cifar100'.")

    # No normalization — we want raw pixel values for saving
    transform = transforms.ToTensor()

    for split in splits:
        is_train = split == "train"
        print(f"\n{'='*50}")
        print(f"Processing {dataset_name.upper()} — {split} split")
        print(f"{'='*50}")

        dataset = dataset_cls(
            root=root,
            train=is_train,
            download=True,
            transform=transform,
        )

        # Build class-name list
        classes = dataset.classes  # list of str, index == label

        # Pre-create class subdirectories
        for cls_name in classes:
            cls_dir = os.path.join(output_dir, split, cls_name)
            os.makedirs(cls_dir, exist_ok=True)

        # Counter per class for unique filenames
        class_counters = {cls: 0 for cls in classes}

        for tensor, label in tqdm(dataset, desc=f"Saving {split}", unit="img"):
            cls_name = classes[label]
            class_counters[cls_name] += 1

            # tensor shape: (C, H, W), values in [0, 1]
            img_array = (tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
            img = Image.fromarray(img_array)

            filename = f"{class_counters[cls_name]:05d}.png"
            save_path = os.path.join(output_dir, split, cls_name, filename)
            img.save(save_path)

        # Summary
        total = sum(class_counters.values())
        print(f"\n✓ Saved {total:,} images to '{os.path.join(output_dir, split)}'")
        print(f"  Classes ({len(classes)}): {', '.join(classes)}")
        print(f"  Images per class: { {k: v for k, v in class_counters.items()} }")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CIFAR-10 or CIFAR-100 to PNG image files."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100"],
        help="Which CIFAR dataset to use (default: cifar10)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./data",
        help="Directory to download raw dataset files (default: ./data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./cifar_images",
        help="Output directory for saved images (default: ./cifar_images)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "test"],
        choices=["train", "test"],
        help="Which splits to export (default: train test)",
    )

    args = parser.parse_args()

    print(f"PyTorch version  : {torch.__version__}")
    print(f"Torchvision ver  : {torchvision.__version__}")
    print(f"Dataset          : {args.dataset}")
    print(f"Download root    : {args.root}")
    print(f"Output directory : {args.output}")
    print(f"Splits           : {args.splits}")

    save_cifar_images(
        dataset_name=args.dataset,
        root=args.root,
        output_dir=args.output,
        splits=args.splits,
    )

    print("\n✅ All done!")


if __name__ == "__main__":
    main()