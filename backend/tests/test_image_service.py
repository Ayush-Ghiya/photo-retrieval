import os
import uuid

import pytest
from botocore.exceptions import ClientError

from app.errors import NotFoundError
from app.models import Image
from app.services.images import INDEX_WARNING, ImageService
from app.services.storage import Storage
from tests.fakes import BrokenIndex
from tests.helpers import jpeg_with_exif, png_bytes


def objects(storage, kind):
    return [o["Key"] for o in storage.client.list_objects_v2(Bucket=storage.bucket(kind)).get("Contents", [])]


def test_upload_creates_row_objects_and_vectors(image_service, storage, index, sessions):
    r = image_service.upload("red.png", png_bytes(), tags=["goa"], title="Beach", description=None)
    assert r.status == "created" and r.message is None
    with sessions() as s:
        img = s.get(Image, r.id)
        assert img.indexed is True
        assert [t.name for t in img.tags] == ["goa"]
        assert (img.filename, img.mime_type, img.source) == ("red.png", "image/png", "upload")
    assert objects(storage, "originals") == [f"originals/{r.id}.png"]
    assert objects(storage, "thumbs") == [f"thumbs/{r.id}.webp"]
    assert index.ids() == [f"{r.id}:img"]


def test_upload_reads_exif_date(image_service):
    r = image_service.upload("p.jpg", jpeg_with_exif(orientation=6, taken="2023:01:02 03:04:05"), tags=[])
    img = image_service.get(r.id)
    assert img.taken_at.year == 2023 and (img.width, img.height) == (100, 200)


def test_duplicate_upload(image_service):
    first = image_service.upload("a.png", png_bytes(), tags=[])
    second = image_service.upload("b.png", png_bytes(), tags=["x"])
    assert second.status == "duplicate" and second.id == first.id


def test_upload_rejects_non_image_and_leaves_nothing(image_service, storage, sessions):
    r = image_service.upload("notes.txt", b"hello", tags=[])
    assert r.status == "error" and "image" in r.message.lower()
    assert objects(storage, "originals") == [] and objects(storage, "thumbs") == []
    with sessions() as s:
        assert s.query(Image).count() == 0


def test_upload_oversize(image_service):
    r = image_service.upload("big.png", os.urandom(1024 * 1024 + 1), tags=[])  # test limit is 1 MB
    assert r.status == "error" and "1 MB" in r.message


def test_storage_failure_cleans_up_original(sessions, index, encoder, settings, storage):
    class ThumbFails(Storage):
        def put(self, kind, key, data, content_type):
            if kind == "thumbs":
                raise ClientError({"Error": {"Code": "500", "Message": "boom"}}, "PutObject")
            super().put(kind, key, data, content_type)

    svc = ImageService(sessions, ThumbFails(settings), index, encoder, settings)
    r = svc.upload("red.png", png_bytes(), tags=[])
    assert r.status == "error"
    assert objects(storage, "originals") == []
    with sessions() as s:
        assert s.query(Image).count() == 0


def test_upload_when_index_down_keeps_row_unindexed(sessions, storage, encoder, settings):
    svc = ImageService(sessions, storage, BrokenIndex(), encoder, settings)
    r = svc.upload("red.png", png_bytes(), tags=["goa"])
    assert r.status == "created" and r.message == INDEX_WARNING
    assert svc.get(r.id).indexed is False
    assert svc.unindexed_ids() == [r.id]


def test_update_tags_and_title_refreshes_index_filters(image_service, index):
    r = image_service.upload("red.png", png_bytes(), tags=[])
    img, warning = image_service.update(r.id, {"title": "  Sunset ", "tags": ["goa", "beach"]})
    assert warning is None
    assert img.title == "Sunset"
    assert [t.name for t in img.tags] == ["beach", "goa"]
    assert index.ids() == [f"{r.id}:img"]
    assert {h.image_id for h in index.query([1.0] + [0.0] * 7, n=10, tags=["beach"])} == {r.id}


def test_update_only_provided_fields(image_service):
    r = image_service.upload("red.png", png_bytes(), tags=["goa"], title="T", description="D")
    img, _ = image_service.update(r.id, {"description": None})
    assert (img.title, img.description, [t.name for t in img.tags]) == ("T", None, ["goa"])


def test_update_clearing_tags_removes_filter_match(image_service, index):
    r = image_service.upload("red.png", png_bytes(), tags=["goa"], title="T")
    image_service.update(r.id, {"title": "", "tags": []})
    assert index.ids() == [f"{r.id}:img"]
    assert index.query([1.0] + [0.0] * 7, n=10, tags=["goa"]) == []


def test_update_unknown_image(image_service):
    with pytest.raises(NotFoundError):
        image_service.update(uuid.uuid4(), {"title": "x"})


def test_delete_removes_everything(image_service, storage, index):
    r = image_service.upload("red.png", png_bytes(), tags=["goa"])
    image_service.delete(r.id)
    assert index.ids() == []
    assert objects(storage, "originals") == [] and objects(storage, "thumbs") == []
    with pytest.raises(NotFoundError):
        image_service.get(r.id)
    with pytest.raises(NotFoundError):
        image_service.delete(r.id)


def test_list_images_pagination_filter_and_sort(image_service):
    ids = [
        image_service.upload(f"{i}.png", png_bytes(color=(i * 20, 0, 0)), tags=["even"] if i % 2 == 0 else []).id
        for i in range(5)
    ]
    page1, total = image_service.list_images(page=1, page_size=2, tags=[], source=None, sort="uploaded")
    assert total == 5 and [i.id for i in page1] == [ids[4], ids[3]]
    page3, _ = image_service.list_images(page=3, page_size=2, tags=[], source=None, sort="uploaded")
    assert [i.id for i in page3] == [ids[0]]
    even, total_even = image_service.list_images(page=1, page_size=10, tags=["even"], source=None, sort="taken")
    assert total_even == 3 and {i.id for i in even} == {ids[0], ids[2], ids[4]}
    none, total_none = image_service.list_images(page=1, page_size=10, tags=["nope"], source=None, sort="taken")
    assert (none, total_none) == ([], 0)
    demo, _ = image_service.list_images(page=1, page_size=10, tags=[], source="demo", sort="taken")
    assert demo == []


def test_tag_counts(image_service):
    image_service.upload("a.png", png_bytes(color=(1, 1, 1)), tags=["goa", "beach"])
    image_service.upload("b.png", png_bytes(color=(2, 2, 2)), tags=["goa"])
    assert image_service.tag_counts() == [("goa", 2), ("beach", 1)]


def test_unexpected_decode_error_is_a_per_file_error(image_service, monkeypatch):
    import app.services.images as images_module

    def boom(data):
        raise ValueError("weird exif")

    monkeypatch.setattr(images_module, "process_image", boom)
    r = image_service.upload("odd.jpg", png_bytes(), tags=[])
    assert r.status == "error" and r.message == "Could not read image"


def test_find_by_metadata_prefers_photos_matching_more_terms(image_service):
    for i in range(3):
        image_service.upload(f"b{i}.png", png_bytes(color=(i, 0, 0)), tags=["beach"])
    both = image_service.upload("goa.png", png_bytes(color=(9, 9, 9)), tags=["beach", "goa"]).id
    found = image_service.find_by_metadata(["goa", "beach"], tags=[], source=None, limit=1)
    assert [i.id for i in found] == [both]


def test_find_by_metadata_matches_word_starts_not_substrings(image_service):
    image_service.upload("p.png", png_bytes(color=(1, 1, 1)), tags=["party"], title="Start of summer")
    hit = image_service.upload("a.png", png_bytes(color=(2, 2, 2)), tags=["street-art"]).id
    assert [i.id for i in image_service.find_by_metadata(["art"], tags=[], source=None, limit=10)] == [hit]
