from PIL import Image

from app.cli import cmd_reindex, cmd_seed_demo
from app.services.images import ImageService
from tests.fakes import BrokenIndex
from tests.helpers import png_bytes


def test_reindex_catches_up_unindexed(services, sessions, storage, encoder, settings, index):
    broken = ImageService(sessions, storage, BrokenIndex(), encoder, settings)
    r = broken.upload("red.png", png_bytes(), tags=["goa"])
    assert services.images.get(r.id).indexed is False

    assert cmd_reindex(services, all_images=False) == 0
    assert services.images.get(r.id).indexed is True
    assert index.ids() == [f"{r.id}:img"]


def test_reindex_all_rebuilds_from_s3(services, index):
    r = services.images.upload("red.png", png_bytes(), tags=[])
    index.recreate()  # simulate a wiped / new collection
    assert cmd_reindex(services, all_images=True) == 0
    assert index.ids() == [f"{r.id}:img"]


def test_seed_demo_is_idempotent(services):
    samples = [
        (Image.new("RGB", (32, 32), (200, 0, 0)), "automobile"),
        (Image.new("RGB", (32, 32), (0, 200, 0)), "frog"),
    ]
    first = cmd_seed_demo(services, samples)
    assert first["created"] == 2
    second = cmd_seed_demo(services, samples)
    assert second["duplicate"] == 2
    items, total = services.images.list_images(page=1, page_size=10, tags=["frog"], source="demo", sort="taken")
    assert total == 1 and items[0].filename == "cifar10-00002.png"


def test_interrupted_reindex_all_leaves_rest_unindexed(services, monkeypatch):
    import pytest

    ids = [services.images.upload(f"{i}.png", png_bytes(color=(i, 5, 5)), tags=[]).id for i in range(3)]
    real = services.images.index_image
    calls = {"n": 0}

    def flaky(image_id, rgb=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real(image_id, rgb)

    monkeypatch.setattr(services.images, "index_image", flaky)
    with pytest.raises(KeyboardInterrupt):
        cmd_reindex(services, all_images=True)
    assert sorted(services.images.unindexed_ids(), key=str) == sorted(ids[1:], key=str)
