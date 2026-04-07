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
  python clip_image_search.py --prompt "small bird" --top_k 10 --split test
  python clip_image_search.py --prompt "dog" --class_filter dog --top_k 5
  python clip_image_search.py --prompt "animal" --save_images --output ./results
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

try:
    import chromadb
except ImportError:
    sys.exit("ChromaDB client not found.\nInstall it with:  pip install chromadb-client")

try:
    import clip
    CLIP_BACKEND = "openai"
except ImportError:
    try:
        import open_clip
        CLIP_BACKEND = "open_clip"
    except ImportError:
        sys.exit(
            "No CLIP library found.\n"
            "Install one of:\n"
            "  pip install git+https://github.com/openai/CLIP.git\n"
            "  pip install open-clip-torch"
        )

COLLECTION_NAME = "cifar_clip"


# ---------------------------------------------------------------------------
# DB + CLIP
# ---------------------------------------------------------------------------

def connect_collection(db_dir: str):
    client =  chromadb.HttpClient(host='localhost', port=8000)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        sys.exit(
            f"Collection '{COLLECTION_NAME}' not found in '{db_dir}'.\n"
            f"Run build_index.py first:\n"
            f"  python build_index.py --dataset_dir ./cifar_images"
        )
    meta  = collection.metadata or {}
    model = meta.get("model", "ViT-B/32")
    print(f"Connected to ChromaDB  —  {collection.count():,} vectors  |  model='{model}'")
    return collection, model


def load_clip_model(model_name: str, device: torch.device):
    if CLIP_BACKEND == "openai":
        model, _ = clip.load(model_name, device=device)
        print(f"Loaded openai/CLIP  model='{model_name}'  device={device}")
    else:
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained="openai"
        )
        model = model.to(device)
        print(f"Loaded open_clip  model='{model_name}'  device={device}")
    model.eval()
    return model


def encode_text(prompt: str, model, device: torch.device) -> list[float]:
    if CLIP_BACKEND == "openai":
        tokens = clip.tokenize([prompt]).to(device)
    else:
        tokens = open_clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        feat = model.encode_text(tokens)
    return F.normalize(feat, dim=-1).cpu().squeeze(0).tolist()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(
    collection,
    query_embedding: list[float],
    top_k: int,
    split_filter: str | None,
    class_filter: str | None,
) -> tuple[list[dict], list[float]]:
    """
    Query ChromaDB.  Supports optional metadata filters so you can restrict
    results to a specific split ('train'/'test') or class name.
    """
    where = {}
    if split_filter and class_filter:
        where = {"$and": [{"split": split_filter}, {"class_name": class_filter}]}
    elif split_filter:
        where = {"split": split_filter}
    elif class_filter:
        where = {"class_name": class_filter}

    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    metadatas = results["metadatas"][0]   # list of dicts
    distances = results["distances"][0]   # cosine distance: 0 = identical, 2 = opposite

    # Convert cosine distance -> similarity score in [0, 1]
    # similarity = 1 - (distance / 2)
    scores = [1.0 - (d / 2.0) for d in distances]

    return metadatas, scores


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

        label = f"{rec['class_name']}  {sim:.3f}"
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
    parser.add_argument("--split",        type=str, default=None,
                        choices=["train", "test"],
                        help="Restrict search to one split (default: search both)")
    parser.add_argument("--class_filter", type=str, default=None,
                        help="Restrict search to one class, e.g. 'dog' or 'airplane'")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 55)
    print("  CLIP Image Search  (ChromaDB)")
    print("=" * 55)
    print(f"  Prompt       : {args.prompt}")
    print(f"  DB dir       : {args.db_dir}")
    print(f"  Split filter : {args.split or 'all'}")
    print(f"  Class filter : {args.class_filter or 'all'}")
    print(f"  Top-k        : {args.top_k}")
    print(f"  Device       : {device}")
    print("=" * 55 + "\n")

    # 1. Connect to DB and get model name that was used to build the index
    collection, model_name = connect_collection(args.db_dir)

    # 2. Load CLIP (text encoder only — no image encoding needed)
    model = load_clip_model(model_name, device)

    # 3. Encode text prompt -> single vector
    print(f'\nEncoding prompt: "{args.prompt}"')
    query_embedding = encode_text(args.prompt, model, device)

    # 4. Search ChromaDB
    print("Searching ChromaDB ...")
    metadatas, scores = search(
        collection,
        query_embedding,
        args.top_k,
        split_filter=args.split,
        class_filter=args.class_filter,
    )

    # 5. Print results
    print(f"\nTop-{len(metadatas)} results for: \"{args.prompt}\"\n")
    print(f"  {'Rank':<5}  {'Score':>6}  {'Split':<6}  {'Class':<15}  {'File'}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*5}  {'-'*14}  {'-'*30}")
    for rank, (rec, score) in enumerate(zip(metadatas, scores), 1):
        print(
            f"  {rank:<5}  {score:>6.4f}  {rec['split']:<6}  "
            f"{rec['class_name']:<15}  {rec['filename']}"
        )

    # 6. Contact sheet
    output_dir  = Path(args.output)
    split_label = args.split or "all"
    cls_label   = f"_{args.class_filter}" if args.class_filter else ""
    sheet_path  = output_dir / f"results_{split_label}{cls_label}.png"
    make_contact_sheet(
        metadatas, scores, args.prompt, sheet_path,
        thumb_size=args.thumb_size, cols=args.cols,
    )

    # 7. Optionally save individual images
    if args.save_images:
        indiv_dir = output_dir / "top_images"
        indiv_dir.mkdir(parents=True, exist_ok=True)
        for rank, (rec, score) in enumerate(zip(metadatas, scores), 1):
            dest = indiv_dir / f"{rank:02d}_{score:.4f}_{rec['class_name']}_{rec['filename']}"
            Image.open(rec["path"]).convert("RGB").save(dest)
        print(f"Individual images saved -> {indiv_dir}")

    print("\nSearch complete!")


if __name__ == "__main__":
    main()