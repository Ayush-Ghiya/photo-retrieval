import uuid

import pytest

from app.services.images import ImageService
from app.services.search import SearchService, blend
from app.services.vector_index import VectorIndex
from tests.fakes import FakeEncoder
from tests.helpers import png_bytes

A, B = uuid.uuid4(), uuid.uuid4()


def test_blend_without_metadata_match_scales_image_similarity():
    [(iid, score)] = blend({A: 0.8}, {}, 0.7)
    assert iid == A and score == pytest.approx(0.7 * 0.8)


def test_blend_combines_image_and_metadata():
    [(iid, score)] = blend({A: 0.6}, {A: 1.0}, 0.7)
    assert score == pytest.approx(0.7 * 0.6 + 0.3 * 1.0)


def test_blend_metadata_only_candidate_without_image_vector():
    [(iid, score)] = blend({}, {A: 0.5}, 0.7)
    assert score == pytest.approx(0.3 * 0.5)


def test_blend_orders_by_score():
    ranked = blend({A: 0.6, B: 0.9}, {}, 0.7)
    assert [i for i, _ in ranked] == [B, A]


RED_Q = [1, 0, 0, 0, 0, 0, 0, 0]
# Realistic CLIP-like scale: the query is only loosely aligned with red images (cos ~0.3).
LOOSE_RED_Q = [0.3, 0, 0, 0.954, 0, 0, 0, 0]


@pytest.fixture
def mapped(sessions, storage, index, settings):
    enc = FakeEncoder({"red": RED_Q, "goa trip": LOOSE_RED_Q})
    images = ImageService(sessions, storage, index, enc, settings)
    return images, SearchService(images, index, enc, settings)


def test_visual_query_ranks_matching_colour_first(mapped):
    images, search = mapped
    red = images.upload("red.png", png_bytes(color=(220, 20, 20)), tags=[]).id
    blue = images.upload("blue.png", png_bytes(color=(20, 20, 220)), tags=[]).id
    results = search.search("red", tags=[], source=None, limit=10)
    assert [img.id for img, _ in results] == [red, blue]
    assert results[0][1] > results[1][1]


def test_metadata_match_outranks_visual_matches_outside_clip_top_n(mapped):
    images, search = mapped
    for i in range(12):  # visually closer to the query than the blue photo
        images.upload(f"red{i}.png", png_bytes(color=(200 + i, 20, 20)), tags=["ship"])
    blue = images.upload("blue.png", png_bytes(color=(20, 20, 220)), tags=["goa-trip"], title="Sunset drive").id
    results = search.search("goa trip", tags=[], source=None, limit=3)  # CLIP top-N is 9 red images
    assert results[0][0].id == blue
    assert len(results) == 3


def test_title_and_description_words_match(mapped):
    images, search = mapped
    images.upload("red.png", png_bytes(color=(220, 20, 20)), tags=[])
    blue = images.upload("blue.png", png_bytes(color=(20, 20, 220)), tags=[], description="Our Goa trip").id
    assert search.search("goa trip", tags=[], source=None, limit=10)[0][0].id == blue


def test_tag_filter_restricts_results(mapped):
    images, search = mapped
    images.upload("red.png", png_bytes(color=(220, 20, 20)), tags=[])
    blue = images.upload("blue.png", png_bytes(color=(20, 20, 220)), tags=["goa"]).id
    results = search.search("red", tags=["goa"], source=None, limit=10)
    assert [img.id for img, _ in results] == [blue]


def test_metadata_candidates_respect_filters(mapped):
    images, search = mapped
    images.upload("a.png", png_bytes(color=(1, 1, 1)), tags=["goa-trip"])
    keep = images.upload("b.png", png_bytes(color=(2, 2, 2)), tags=["goa-trip", "family"]).id
    results = search.search("goa trip", tags=["family"], source=None, limit=10)
    assert [img.id for img, _ in results] == [keep]
    assert search.search("goa trip", tags=[], source="demo", limit=10) == []


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
    index.upsert(uuid.uuid4(), image_vector=RED_Q, tags=[], source="upload")
    real = images.upload("red.png", png_bytes(), tags=[]).id
    assert [img.id for img, _ in search.search("red", tags=[], source=None, limit=10)] == [real]


@pytest.mark.slow
def test_real_clip_user_tag_beats_demo_class_tags(sessions, storage, settings):
    """Regression: CLIP text-text similarity made 'tags: ship' outrank a 'goa-trip' tag."""
    from app.services.clip_model import ClipEncoder

    enc = ClipEncoder("ViT-B/32")
    index = VectorIndex(settings.chroma_host, settings.chroma_port, settings.chroma_collection, enc.model_name)
    index.recreate()
    images = ImageService(sessions, storage, index, enc, settings)
    search = SearchService(images, index, enc, settings)
    for i in range(20):
        images.upload(f"ship{i}.png", png_bytes(color=(20, 60 + i, 160)), tags=["ship"])
    tagged = images.upload("truck.png", png_bytes(color=(200, 30, 30)), tags=["truck", "goa-trip"], title="Sunset drive").id
    assert search.search("goa trip", tags=[], source=None, limit=5)[0][0].id == tagged
