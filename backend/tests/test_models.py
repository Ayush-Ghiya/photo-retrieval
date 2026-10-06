import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.models import Image, Tag, image_tags
from app.repo import get_or_create_tags, tag_counts


def make_image(**overrides) -> Image:
    values = dict(
        id=uuid.uuid4(), s3_key="originals/x.png", thumb_key="thumbs/x.webp",
        filename="x.png", mime_type="image/png", width=10, height=10,
        content_hash=uuid.uuid4().hex, source="upload",
    )
    values.update(overrides)
    return Image(**values)


def test_image_with_tags_roundtrip(sessions):
    with sessions.begin() as s:
        img = make_image()
        img.tags = get_or_create_tags(s, ["goa", "beach"])
        s.add(img)
    with sessions() as s:
        loaded = s.get(Image, img.id)
        assert [t.name for t in loaded.tags] == ["beach", "goa"]
        assert loaded.indexed is False
        assert loaded.created_at is not None


def test_get_or_create_tags_reuses_existing(sessions):
    with sessions.begin() as s:
        first = get_or_create_tags(s, ["goa"])
    with sessions.begin() as s:
        again = get_or_create_tags(s, ["goa", "new"])
        assert again[0].id == first[0].id
    with sessions() as s:
        assert sorted(s.scalars(select(Tag.name))) == ["goa", "new"]


def test_content_hash_is_unique(sessions):
    with sessions.begin() as s:
        s.add(make_image(content_hash="same"))
    with pytest.raises(IntegrityError):
        with sessions.begin() as s:
            s.add(make_image(content_hash="same"))


def test_deleting_image_removes_links_and_counts(sessions):
    with sessions.begin() as s:
        a, b = make_image(), make_image()
        a.tags = get_or_create_tags(s, ["goa", "family"])
        b.tags = get_or_create_tags(s, ["goa"])
        s.add_all([a, b])
    with sessions() as s:
        assert tag_counts(s) == [("goa", 2), ("family", 1)]
    with sessions.begin() as s:
        s.delete(s.get(Image, a.id))
    with sessions() as s:
        assert tag_counts(s) == [("goa", 1)]
        assert s.execute(select(image_tags)).all() != []
