"""
clip_image_search.py
----------------------------------------------------
Loads the ChromaDB collection built by build_index.py and returns
the top-k images matching a text prompt.

No re-encoding of images. No loading a giant tensor into RAM.
ChromaDB handles nearest-neighbour search entirely on disk via HNSW.

Usage
-----
  python clip_image_search.py --prompt "a red car on the road"
  python clip_image_search.py --prompt "small bird" --top_k 10
  python clip_image_search.py --prompt "dog"
  python clip_image_search.py --prompt "animal" --save_images --output ./results
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from search import search_images
from utils import get_chroma_db_client, load_clip_model,COLLECTION_NAME









# ---------------------------------------------------------------------------
# Contact sheet
# ---------------------------------------------------------------------------

def make_contact_sheet(
    records: list[dict],
    scores: list[float],
    prompt: str,
    output_path: Path,
    thumb_size: int = 96,
    cols: int = 5,
) -> None:
    n        = len(records)
    rows     = math.ceil(n / cols)
    label_h  = 28
    header_h = 44
    W = cols * thumb_size
    H = header_h + rows * (thumb_size + label_h)

    sheet = Image.new("RGB", (W, H), color=(30, 30, 30))
    draw  = ImageDraw.Draw(sheet)

    try:
        font_sm  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_hdr = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font_sm  = ImageFont.load_default()
        font_hdr = font_sm

    draw.text((8, 8),  f'"{prompt}"',           fill=(255, 220, 100), font=font_hdr)
    draw.text((8, 26), f"Top-{n} CLIP matches", fill=(180, 180, 180), font=font_sm)

    for idx, (rec, sim) in enumerate(zip(records, scores)):
        row, col = divmod(idx, cols)
        x = col * thumb_size
        y = header_h + row * (thumb_size + label_h)

        try:
            thumb = (
                Image.open(rec["path"])
                .convert("RGB")
                .resize((thumb_size, thumb_size), Image.NEAREST)
            )
        except Exception:
            thumb = Image.new("RGB", (thumb_size, thumb_size), (60, 60, 60))

        sheet.paste(thumb, (x, y))

        label = f"{sim:.3f}"
        draw.rectangle(
            [x, y + thumb_size, x + thumb_size, y + thumb_size + label_h],
            fill=(20, 20, 20),
        )
        draw.text((x + 3, y + thumb_size + 6), label, fill=(200, 255, 200), font=font_sm)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    print(f"Contact sheet saved -> {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Search CIFAR images by text prompt via ChromaDB + CLIP."
    )
    parser.add_argument("--prompt",       type=str, required=True,
                        help='Text query, e.g. "a red car on the road"')
    parser.add_argument("--db_dir",       type=str, default="./chroma_db",
                        help="ChromaDB directory built by build_index.py (default: ./chroma_db)")
    parser.add_argument("--top_k",        type=int, default=20,
                        help="Number of results to return (default: 20)")
    parser.add_argument("--output",       type=str, default="./clip_results",
                        help="Directory for the contact sheet (default: ./clip_results)")
    parser.add_argument("--save_images",  action="store_true",
                        help="Also copy each top-k image individually to --output")
    parser.add_argument("--thumb_size",   type=int, default=96,
                        help="Thumbnail size in the contact sheet (default: 96)")
    parser.add_argument("--cols",         type=int, default=5,
                        help="Columns in the contact sheet (default: 5)")

    args   = parser.parse_args()
   
    image_list,results,scores = search_images(
        prompt=args.prompt,
        top_k=args.top_k,
        db_dir=args.db_dir,
    )

    # 6. Contact sheet
    output_dir  = Path(args.output)
    sheet_path  = output_dir / f"results_{args.top_k}.png"
    make_contact_sheet(
        results, scores, args.prompt, sheet_path,
        thumb_size=args.thumb_size, cols=args.cols,
    )

    # 7. Optionally save individual images
    if args.save_images:
        indiv_dir = output_dir / "top_images"
        indiv_dir.mkdir(parents=True, exist_ok=True)
        for rank, (rec, score) in enumerate(zip(results, scores), 1):
            dest = indiv_dir / f"{rank:02d}_{score:.4f}_{rec['filename']}"
            Image.open(rec["path"]).convert("RGB").save(dest)
        print(f"Individual images saved -> {indiv_dir}")

    print("\nSearch complete!")


if __name__ == "__main__":
    main()