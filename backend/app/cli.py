"""Maintenance commands.

  python -m app.cli reindex          # index images with indexed=false
  python -m app.cli reindex --all    # recreate the collection and re-embed everything from S3
  python -m app.cli seed-demo [--limit N] [--data-dir ../data]
"""
import argparse
import io
import logging
import sys
from collections import Counter
from collections.abc import Iterable, Iterator

from PIL import Image

from app.config import get_settings
from app.services.container import Services, build_services
from app.tags import normalize_tag


def cmd_reindex(services: Services, *, all_images: bool) -> int:
    if all_images:
        # Flag everything first so an interrupted run is resumed by a plain `reindex`.
        services.images.mark_all_unindexed()
        services.index.recreate()
    ids = services.images.unindexed_ids()
    print(f"Reindexing {len(ids)} image(s) ...")
    failed = 0
    for n, image_id in enumerate(ids, 1):
        if services.images.index_image(image_id) is not None:
            failed += 1
        if n % 100 == 0:
            print(f"  {n}/{len(ids)}")
    print(f"Done: {len(ids) - failed} indexed, {failed} failed")
    return 1 if failed else 0


def cmd_seed_demo(services: Services, samples: Iterable[tuple[Image.Image, str]]) -> Counter:
    counts: Counter = Counter()
    for i, (img, label) in enumerate(samples, 1):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = services.images.upload(
            f"cifar10-{i:05d}.png", buf.getvalue(), tags=[normalize_tag(label)], source="demo"
        )
        counts[result.status] += 1
        if i % 500 == 0:
            print(f"  {i} processed {dict(counts)}")
    print(f"Seed complete: {dict(counts)}")
    return counts


def cifar_samples(data_dir: str, limit: int | None) -> Iterator[tuple[Image.Image, str]]:
    from torchvision.datasets import CIFAR10

    ds = CIFAR10(root=data_dir, train=False, download=True)
    n = len(ds) if limit is None else min(limit, len(ds))
    for i in range(n):
        img, label = ds[i]
        yield img, ds.classes[label]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    p_re = sub.add_parser("reindex", help="Rebuild missing (or all) vectors in ChromaDB")
    p_re.add_argument("--all", action="store_true", dest="all_images", help="Recreate collection, re-embed all")
    p_seed = sub.add_parser("seed-demo", help="Load the CIFAR-10 test set as demo images")
    p_seed.add_argument("--limit", type=int, default=None)
    p_seed.add_argument("--data-dir", default="../data")
    args = parser.parse_args(argv)

    services = build_services(get_settings())
    services.startup(recreate_index=args.command == "reindex" and args.all_images)
    if args.command == "reindex":
        return cmd_reindex(services, all_images=args.all_images)
    counts = cmd_seed_demo(services, cifar_samples(args.data_dir, args.limit))
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
