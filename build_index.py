"""
build_index.py  
---------------------------------------------
Walks the CIFAR image folder, encodes every image with CLIP,
and stores the embeddings + metadata in a persistent ChromaDB collection.

Run this ONCE.  Subsequent runs skip images already in the DB unless
you pass --rebuild to wipe and start fresh.

Usage
-----
  pip install chromadb-client
  python build_index.py
  python build_index.py --dataset_dir ./cifar_images --splits train test
  python build_index.py --model ViT-L/14 --batch_size 256
  python build_index.py --rebuild          # wipe DB and start over
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from utils import load_clip_model,COLLECTION_NAME,SUPPORTED_EXTS,get_chroma_db_client



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





def collect_image_paths(dataset_dir: Path) -> list[Path]:
    if not dataset_dir.exists():
        sys.exit(f"Directory not found: {dataset_dir}")

    # Update glob to rglob to get recursive search
    paths = sorted(
        p for p in dataset_dir.glob("*") if p.suffix.lower() in SUPPORTED_EXTS
    )

    if not paths:
        sys.exit(f"No images found under '{dataset_dir}'. Run build_dataset.py first.")

    print(f"  {dataset_dir.name} -> {len(paths):>7,} images")
    return paths


def path_to_id(path: Path) -> str:
    return str(path).replace("\\", "/")


def encode_batch(
    batch_paths: list[Path],
    model,
    preprocess,
    device: torch.device,
) -> list[list[float]]:
    tensors = []
    for p in batch_paths:
        try:
            img = Image.open(p).convert("RGB")
            tensors.append(preprocess(img))
        except Exception as e:
            print(f"\n  Warning: skipping {p.name}: {e}")
            tensors.append(torch.zeros(3, 224, 224))

    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        feats = model.encode_image(batch)
    normed = F.normalize(feats, dim=-1).cpu()
    return normed.tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Build a ChromaDB vector index from a given dataset image folder."
    )
    parser.add_argument("--dataset_dir", type=str, default="./cifar_images",
                        help="Root image directory from build_dataset.py (default: ./cifar_images)")
    parser.add_argument("--model", type=str, default="ViT-B/32",
                        help="CLIP model variant (default: ViT-B/32)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Encoding batch size (default: 128)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Delete existing collection and rebuild from scratch")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 55)
    print("  CLIP Index Builder  ->  ChromaDB")
    print("=" * 55)
    print(f"  Dataset dir  : {args.dataset_dir}")
    print(f"  Model        : {args.model}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Device       : {device}")
    print(f"  Rebuild      : {args.rebuild}")
    print("=" * 55 + "\n")

    dataset_dir = Path(args.dataset_dir)

    # 1. Connect to ChromaDB
    print(f"Connecting to ChromaDB client ...")
    client =  get_chroma_db_client()

    if args.rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"  Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "model": args.model,
            "clip_backend": CLIP_BACKEND,
        },
    )
    existing_count = collection.count()
    print(f"  Collection '{COLLECTION_NAME}' — {existing_count:,} existing vectors")

    # 2. Collect paths
    print(f"\nScanning '{dataset_dir}' ...")
    all_paths = collect_image_paths(dataset_dir)
    print(f"  Total found  : {len(all_paths):,} images")

    # 3. Skip already-indexed images (incremental)
    if existing_count > 0 and not args.rebuild:
        existing_ids = set(collection.get(include=[])["ids"])
        all_paths = [p for p in all_paths if path_to_id(p) not in existing_ids]
        print(f"  To index     : {len(all_paths):,} new images (skipping already indexed)")

    if not all_paths:
        print("\n  Nothing new to index. Use --rebuild to re-index everything.")
        return

    # 4. Load CLIP
    model, preprocess = load_clip_model(args.model, device,CLIP_BACKEND)

    # 5. Encode and upsert in batches
    print(f"\nEncoding and inserting into ChromaDB ...")
    t0 = time.time()
    total_inserted = 0

    for start in tqdm(range(0, len(all_paths), args.batch_size), unit="batch"):
        batch_paths = all_paths[start : start + args.batch_size]
        embeddings = encode_batch(batch_paths, model, preprocess, device)

        ids       = [path_to_id(p) for p in batch_paths]
        # Review this metadata extraction logic to ensure it correctly captures the intended structure
        # The current logic assumes a directory structure where the first level is the split (e.g., "train", "test") and the second level is the class name. If the structure is different, this may need to be adjusted.
        # For example, if the images are directly under the dataset_dir without subdirectories, the split will be "unknown" and the class_name will be derived from the parent directory of the image. If there are more nested directories, the logic may need to be updated to correctly identify the split and class name.
        metadatas = []
        for p in batch_paths:
            metadatas.append({
                "path":       str(p),
                "filename":   p.name,
            })


            

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        total_inserted += len(batch_paths)

    elapsed = time.time() - t0
    rate    = total_inserted / elapsed if elapsed > 0 else 0

    print(f"\n  Inserted {total_inserted:,} vectors in {elapsed:.1f}s  ({rate:.0f} img/s)")
    print(f"  Collection total: {collection.count():,} vectors")
    print("\nIndex build complete!")
    print(f"  Run:  python clip_image_search.py --prompt \"your query\"")


if __name__ == "__main__":
    main()