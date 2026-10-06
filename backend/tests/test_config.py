import pytest
from pydantic import ValidationError

from app.config import Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.s3_bucket_originals == "photos-originals"
    assert s.s3_bucket_thumbs == "photos-thumbs"
    assert s.chroma_collection == "photos_v1"
    assert s.clip_model == "ViT-B/32"
    assert s.search_img_weight == 0.7
    assert s.max_upload_mb == 25
    assert s.max_batch_files == 50
    assert s.presign_ttl_seconds == 3600


def test_env_overrides_and_derived_values(monkeypatch):
    monkeypatch.setenv("SEARCH_IMG_WEIGHT", "0.5")
    monkeypatch.setenv("CORS_ORIGINS", "http://a:1, http://b:2")
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://localhost:4566")
    monkeypatch.delenv("S3_PUBLIC_ENDPOINT_URL", raising=False)
    s = Settings(_env_file=None)
    assert s.search_img_weight == 0.5
    assert s.cors_origin_list == ["http://a:1", "http://b:2"]
    assert s.s3_public_endpoint == "http://localhost:4566"


def test_empty_public_endpoint_falls_back(monkeypatch):
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://localhost:4566")
    monkeypatch.setenv("S3_PUBLIC_ENDPOINT_URL", "")
    assert Settings(_env_file=None).s3_public_endpoint == "http://localhost:4566"


def test_weight_must_be_between_0_and_1(monkeypatch):
    monkeypatch.setenv("SEARCH_IMG_WEIGHT", "1.5")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
