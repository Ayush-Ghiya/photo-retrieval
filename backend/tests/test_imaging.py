import hashlib
import io
from datetime import datetime, timezone

import pytest
from PIL import Image

from app.services.imaging import UnsupportedImage, open_rgb, process_image
from tests.helpers import jpeg_with_exif, png_bytes


def test_png_basic_properties():
    data = png_bytes(size=(64, 32))
    p = process_image(data)
    assert p.content_hash == hashlib.sha256(data).hexdigest()
    assert (p.mime_type, p.ext) == ("image/png", "png")
    assert (p.width, p.height) == (64, 32)
    assert p.taken_at is None
    thumb = Image.open(io.BytesIO(p.thumbnail))
    assert thumb.format == "WEBP"
    assert p.rgb.mode == "RGB"


def test_jpeg_orientation_and_taken_at():
    data = jpeg_with_exif(size=(200, 100), orientation=6, taken="2024:05:01 10:20:30")
    p = process_image(data)
    assert (p.mime_type, p.ext) == ("image/jpeg", "jpg")
    assert (p.width, p.height) == (100, 200)  # rotated 90°
    assert p.rgb.size == (100, 200)
    assert p.taken_at == datetime(2024, 5, 1, 10, 20, 30, tzinfo=timezone.utc)


def test_bad_exif_date_is_ignored():
    p = process_image(jpeg_with_exif(taken="not a date"))
    assert p.taken_at is None


def test_thumbnail_longest_side_is_512():
    p = process_image(png_bytes(size=(2000, 1000)))
    assert Image.open(io.BytesIO(p.thumbnail)).size == (512, 256)


def test_small_image_is_not_upscaled():
    p = process_image(png_bytes(size=(32, 32)))
    assert Image.open(io.BytesIO(p.thumbnail)).size == (32, 32)


def test_non_image_rejected():
    with pytest.raises(UnsupportedImage):
        process_image(b"definitely not an image")


def test_disallowed_format_rejected():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="GIF")
    with pytest.raises(UnsupportedImage):
        process_image(buf.getvalue())


def test_truncated_jpeg_rejected():
    data = jpeg_with_exif(size=(400, 400))
    with pytest.raises(UnsupportedImage):
        process_image(data[: len(data) // 2])


def test_open_rgb_applies_orientation():
    img = open_rgb(jpeg_with_exif(size=(200, 100), orientation=6))
    assert img.size == (100, 200) and img.mode == "RGB"
