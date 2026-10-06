import math

import pytest
from PIL import Image

from tests.fakes import FakeEncoder


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_fake_encoder_is_deterministic_and_normalised():
    enc = FakeEncoder()
    a1, a2 = enc.encode_text(["hello", "hello"])
    assert a1 == a2
    assert math.isclose(_dot(a1, a1), 1.0, rel_tol=1e-9)
    (img,) = enc.encode_images([Image.new("RGB", (4, 4), (255, 0, 0))])
    assert math.isclose(_dot(img, img), 1.0, rel_tol=1e-9)


def test_fake_encoder_text_override():
    enc = FakeEncoder({"red": [1, 0, 0, 0, 0, 0, 0, 0]})
    (vec,) = enc.encode_text(["red"])
    assert vec[0] == pytest.approx(1.0)


@pytest.mark.slow
@pytest.mark.filterwarnings("error:QuickGELU mismatch")
def test_real_clip_prefers_matching_colour():
    from app.services.clip_model import ClipEncoder

    enc = ClipEncoder("ViT-B/32")
    red, blue = enc.encode_images(
        [Image.new("RGB", (224, 224), (220, 20, 20)), Image.new("RGB", (224, 224), (20, 20, 220))]
    )
    (text,) = enc.encode_text(["a plain red square"])
    assert len(text) == 512
    assert math.isclose(_dot(text, text), 1.0, rel_tol=1e-3)
    assert _dot(text, red) > _dot(text, blue)
