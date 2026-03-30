"""
CLIP Image Search over CIFAR Dataset
--------------------------------------
Given a text prompt from the CLI, this script uses OpenAI's CLIP model to
retrieve the most semantically similar images from a saved CIFAR image directory
(produced by cifar_to_images.py).

How it works:
  1. Encode the CLI text prompt into a CLIP text embedding.
  2. Walk every image in the dataset directory, encode each with CLIP's image encoder.
  3. Compute cosine similarity between the text embedding and every image embedding.
  4. Return the top-k matches and save them to an output folder as a contact sheet.

Usage examples:
  python clip_image_search.py --prompt "a red car on the road"
  python clip_image_search.py --prompt "small bird in the sky" --top_k 10 --split test
  python clip_image_search.py --prompt "animal near water" --dataset_dir ./cifar_images --output ./results
"""

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CLIP import — supports both openai/CLIP (pip install git+...) and
# the newer open_clip (pip install open-clip-torch).
# We try openai-clip first, then open_clip.
# ---------------------------------------------------------------------------
try:
    import clip  # openai/CLIP

    CLIP_BACKEND = "openai"
except ImportError:
    try:
        import open_clip  # open_clip-torch

        CLIP_BACKEND = "open_clip"
    except ImportError:
        sys.exit(
            "❌  No CLIP library found.\n"
            "Install one of:\n"
            "  pip install git+https://github.com/openai/CLIP.git\n"
            "  pip install open-clip-torch"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_clip_model(model_name: str, device: torch.device):
    """Load CLIP model and preprocessing transform."""
    if CLIP_BACKEND == "openai":
        model, preprocess = clip.load(model_name, device=device)
        print(f"✓ Loaded openai/CLIP  model='{model_name}'  device={device}")
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="openai"
        )
        model = model.to(device)
        print(f"✓ Loaded open_clip  model='{model_name}'  device={device}")
    model.eval()
    return model, preprocess


def encode_text(prompt: str, model, device: torch.device) -> torch.Tensor:
    """Return a normalised text embedding for *prompt*."""
    if CLIP_BACKEND == "openai":
        tokens = clip.tokenize([prompt]).to(device)
    else:
        tokens = open_clip.tokenize([prompt]).to(device)

    with torch.no_grad():
        text_feat = model.encode_text(tokens)  # (1, D)
    return F.normalize(text_feat, dim=-1)


def encode_images_batch(
    paths: list[Path],
    model,
    preprocess,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Encode a list of image paths in batches.
    Returns a (N, D) normalised tensor.
    """
    all_feats = []
    for start in tqdm(
        range(0, len(paths), batch_size), desc="Encoding images", unit="batch"
    ):
        batch_paths = paths[start : start + batch_size]
        tensors = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(preprocess(img))
            except Exception as e:
                print(f"  ⚠ Skipping {p.name}: {e}")
                tensors.append(torch.zeros(3, 224, 224))  # placeholder

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch)  # (B, D)
        all_feats.append(F.normalize(feats, dim=-1).cpu())

    return torch.cat(all_feats, dim=0)  # (N, D)


def collect_image_paths(dataset_dir: str, split: str) -> list[Path]:
    """Walk <dataset_dir>/<split>/ and collect all image file paths."""
    root = Path(dataset_dir) / split
    if not root.exists():
        sys.exit(
            f"❌  Directory not found: {root}\n"
            f"    Run cifar_to_images.py first or check --dataset_dir / --split."
        )
    paths = sorted(
        p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS
    )
    if not paths:
        sys.exit(f"❌  No images found under {root}")
    print(f"✓ Found {len(paths):,} images in '{root}'")
    return paths


def make_contact_sheet(
    image_paths: list[Path],
    similarities: list[float],
    prompt: str,
    output_path: Path,
    thumb_size: int = 96,
    cols: int = 5,
) -> None:
    """Tile the retrieved images into a single contact-sheet PNG."""
    n = len(image_paths)
    rows = math.ceil(n / cols)
    label_h = 28                    # pixels reserved for the score label
    header_h = 44                   # pixels reserved for the prompt header
    W = cols * thumb_size
    H = header_h + rows * (thumb_size + label_h)

    sheet = Image.new("RGB", (W, H), color=(30, 30, 30))
    draw = ImageDraw.Draw(sheet)

    # Try to load a font; fall back gracefully
    try:
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_header = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font_small = ImageFont.load_default()
        font_header = font_small

    # Header
    draw.text((8, 8), f'🔍  "{prompt}"', fill=(255, 220, 100), font=font_header)
    draw.text((8, 26), f"Top-{n} CLIP matches", fill=(180, 180, 180), font=font_small)

    for idx, (path, sim) in enumerate(zip(image_paths, similarities)):
        row, col = divmod(idx, cols)
        x = col * thumb_size
        y = header_h + row * (thumb_size + label_h)

        # Thumbnail
        try:
            thumb = Image.open(path).convert("RGB").resize(
                (thumb_size, thumb_size), Image.NEAREST   # CIFAR is 32×32; keep crisp
            )
        except Exception:
            thumb = Image.new("RGB", (thumb_size, thumb_size), (60, 60, 60))
        sheet.paste(thumb, (x, y))

        # Score label
        class_name = path.parent.name
        label = f"{class_name}  {sim:.3f}"
        draw.rectangle([x, y + thumb_size, x + thumb_size, y + thumb_size + label_h], fill=(20, 20, 20))
        draw.text((x + 3, y + thumb_size + 6), label, fill=(200, 255, 200), font=font_small)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    print(f"✓ Contact sheet saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Retrieve CIFAR images matching a text prompt using CLIP."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help='Text query, e.g. "a red car on the road"',
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./cifar_images",
        help="Root directory produced by cifar_to_images.py (default: ./cifar_images)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Which split to search (default: train)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of top results to return (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./clip_results",
        help="Directory to save the contact sheet and top images (default: ./clip_results)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B/32",
        help='CLIP model variant (default: ViT-B/32). For open_clip use e.g. "ViT-B-32"',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Image encoding batch size (default: 128)",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Also copy each top-k image individually to the output directory",
    )
    parser.add_argument(
        "--thumb_size",
        type=int,
        default=96,
        help="Thumbnail size in the contact sheet in pixels (default: 96)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=5,
        help="Columns in the contact sheet (default: 5)",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 55)
    print("  CLIP Image Search")
    print("=" * 55)
    print(f"  Prompt      : {args.prompt}")
    print(f"  Dataset dir : {args.dataset_dir}")
    print(f"  Split       : {args.split}")
    print(f"  Top-k       : {args.top_k}")
    print(f"  Device      : {device}")
    print("=" * 55 + "\n")

    # 1. Load model
    model, preprocess = load_clip_model(args.model, device)

    # 2. Encode text prompt
    print(f"\nEncoding prompt: \"{args.prompt}\"")
    text_feat = encode_text(args.prompt, model, device)  # (1, D)

    # 3. Collect image paths
    image_paths = collect_image_paths(args.dataset_dir, args.split)

    # 4. Encode all images
    print(f"\nEncoding {len(image_paths):,} images …")
    image_feats = encode_images_batch(
        image_paths, model, preprocess, device, batch_size=args.batch_size
    )  # (N, D)

    # 5. Cosine similarity  (text_feat already on CPU after encode_text?)
    text_feat_cpu = text_feat.cpu()
    sims = (image_feats @ text_feat_cpu.T).squeeze(1)  # (N,)

    # 6. Top-k
    k = min(args.top_k, len(image_paths))
    top_values, top_indices = torch.topk(sims, k)

    top_paths = [image_paths[i] for i in top_indices.tolist()]
    top_scores = top_values.tolist()

    print(f"\n🏆  Top-{k} results for: \"{args.prompt}\"\n")
    print(f"  {'Rank':<5}  {'Score':>6}  {'Class':<15}  {'File'}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*14}  {'-'*30}")
    for rank, (path, score) in enumerate(zip(top_paths, top_scores), 1):
        print(f"  {rank:<5}  {score:>6.4f}  {path.parent.name:<15}  {path.name}")

    # 7. Contact sheet
    output_dir = Path(args.output)
    sheet_path = output_dir / f"results_{args.split}.png"
    make_contact_sheet(
        top_paths,
        top_scores,
        args.prompt,
        sheet_path,
        thumb_size=args.thumb_size,
        cols=args.cols,
    )

    # 8. (Optional) copy individual images
    if args.save_images:
        indiv_dir = output_dir / "top_images"
        indiv_dir.mkdir(parents=True, exist_ok=True)
        for rank, (path, score) in enumerate(zip(top_paths, top_scores), 1):
            dest = indiv_dir / f"{rank:02d}_{score:.4f}_{path.parent.name}_{path.name}"
            Image.open(path).convert("RGB").save(dest)
        print(f"✓ Individual images saved → {indiv_dir}")

    print("\n✅  Search complete!")


if __name__ == "__main__":
    main()