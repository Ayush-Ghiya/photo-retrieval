"""
CIFAR Dataset to Images Converter
Loads CIFAR-10 using torchvision and saves each image
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
    root: str = "./data",
    output_dir: str = "./cifar_images",
) -> None:
    """
    Download and convert a CIFAR dataset to individual image files.

    Directory structure created:
        <output_dir>/
            0001.png  0002.png  ...

    Args:
        root:         Directory where the raw dataset is downloaded.
        output_dir:   Root directory for the saved PNG images.
    """
    

    # No normalization — we want raw pixel values for saving
    transform = transforms.ToTensor()
    os.makedirs(output_dir, exist_ok=True)
      
    dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=False,
            download=True,
            transform=transform,
        )
    
    counter = 0
    
    for tensor,label in tqdm(dataset, desc=f"Saving Images", unit="img"):
        counter += 1
         # tensor shape: (C, H, W), values in [0, 1]
        img_array = (tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
        img = Image.fromarray(img_array)
        
        filename = f"{counter:05d}.png"
        save_path = os.path.join(output_dir, filename)
        img.save(save_path)

    # Summary
    print(f"\n✓ Saved {counter:,} images to '{os.path.join(output_dir)}'")
        
    


def main():
    parser = argparse.ArgumentParser(
        description="Convert CIFAR-10 or CIFAR-100 to PNG image files."
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

    args = parser.parse_args()

    print(f"PyTorch version  : {torch.__version__}")
    print(f"Torchvision ver  : {torchvision.__version__}")
    print(f"Download root    : {args.root}")
    print(f"Output directory : {args.output}")

    save_cifar_images(
        root=args.root,
        output_dir=args.output
    )

    print("\n✅ All done!")


if __name__ == "__main__":
    main()