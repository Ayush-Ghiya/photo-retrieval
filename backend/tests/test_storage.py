import httpx
import pytest
from botocore.exceptions import ClientError


def test_put_get_delete_roundtrip(storage):
    storage.put("originals", "originals/a.png", b"hello", "image/png")
    assert storage.get("originals", "originals/a.png") == b"hello"
    storage.delete("originals", "originals/a.png")
    with pytest.raises(ClientError):
        storage.get("originals", "originals/a.png")


def test_delete_missing_key_is_noop(storage):
    storage.delete("thumbs", "thumbs/does-not-exist.webp")


def test_ensure_buckets_is_idempotent(storage):
    storage.ensure_buckets()
    storage.ping()


def test_presigned_url_serves_object(storage):
    storage.put("thumbs", "thumbs/t.webp", b"thumb-bytes", "image/webp")
    url = storage.presign("thumbs", "thumbs/t.webp")
    assert url.startswith("http://localhost:4566/")
    resp = httpx.get(url)
    assert resp.status_code == 200
    assert resp.content == b"thumb-bytes"
