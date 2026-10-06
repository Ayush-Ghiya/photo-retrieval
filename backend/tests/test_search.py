import uuid

import pytest

from app.services.images import ImageService
from app.services.search import SearchService, blend
from app.services.vector_index import Hit
from tests.fakes import FakeEncoder
from tests.helpers import png_bytes

A, B = uuid.uuid4(), uuid.uuid4()


def test_blend_image_only_uses_image_similarity():
    [(iid, score)] = blend([Hit(A, "img", 0.8)], {}, 0.7)
    assert iid == A and score == pytest.approx(0.8)


def test_blend_combines_image_and_text():
    [(iid, score)] = blend([Hit(A, "img", 0.6), Hit(A, "txt", 1.0)], {}, 0.7)
    assert score == pytest.approx(0.7 * 0.6 + 0.3 * 1.0)


def test_blend_text_only_hit_uses_fetched_image_similarity():
    [(iid, score)] = blend([Hit(A, "txt", 0.9)], {A: 0.5}, 0.7)
    assert score == pytest.approx(0.7 * 0.5 + 0.3 * 0.9)


def test_blend_orders_by_score():
    ranked = blend([Hit(A, "img", 0.6), Hit(B, "img", 0.9)], {}, 0.7)
    assert [i for i, _ in ranked] == [B, A]


RED_Q = [1, 0, 0, 0, 0, 0, 0, 0]
GOA_Q = [0, 0, 0, 1, 0, 0, 0, 0]


@pytest.fixture
def mapped(sessions, storage, index, settings):
    enc = FakeEncoder({"red": RED_Q, "goa": GOA_Q, "tags: goa": GOA_Q})
    images = ImageService(sessions, storage, index, enc, settings)
    return images, SearchService(images, index, enc, settings)


def test_visual_query_ranks_matching_colour_first(mapped):
    images, search = mapped
    red = images.upload("red.png", png_bytes(color=(220, 20, 20)), tags=[]).id
    blue = images.upload("blue.png", png_bytes(color=(20, 20, 220)), tags=[]).id
    results = search.search("red", tags=[], source=None, limit=10)
    assert [img.id for img, _ in results] == [red, blue]
    assert results[0][1] > results[1][1]


def test_metadata_lifts_tagged_image(mapped):
    images, search = mapped
    red = images.upload("red.png", png_bytes(color=(220, 20, 20)), tags=[]).id
    blue = images.upload("blue.png", png_bytes(color=(20, 20, 220)), tags=["goa"]).id
    results = search.search("goa", tags=[], source=None, limit=10)
    assert [img.id for img, _ in results] == [blue, red]
    assert results[0][1] == pytest.approx(0.7 * 0.5 + 0.3 * 1.0, abs=1e-3)


def test_tag_filter_restricts_results(mapped):
    images, search = mapped
    images.upload("red.png", png_bytes(color=(220, 20, 20)), tags=[])
    blue = images.upload("blue.png", png_bytes(color=(20, 20, 220)), tags=["goa"]).id
    results = search.search("red", tags=["goa"], source=None, limit=10)
    assert [img.id for img, _ in results] == [blue]


def test_search_unknown_tag_returns_empty(search_service, image_service):
    image_service.upload("red.png", png_bytes(), tags=[])
    assert search_service.search("anything", tags=["nobody-uses-this"], source=None, limit=10) == []


def test_limit_is_respected(mapped):
    images, search = mapped
    for i in range(5):
        images.upload(f"{i}.png", png_bytes(color=(50 * i, 10, 10)), tags=[])
    assert len(search.search("red", tags=[], source=None, limit=2)) == 2


def test_search_skips_vectors_without_rows(mapped, index):
    images, search = mapped
    index.upsert(uuid.uuid4(), image_vector=RED_Q, text_vector=None, tags=[], source="upload")
    real = images.upload("red.png", png_bytes(), tags=[]).id
    assert [img.id for img, _ in search.search("red", tags=[], source=None, limit=10)] == [real]
