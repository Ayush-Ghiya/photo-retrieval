import uuid

import pytest

from app.services.vector_index import ModelMismatchError, VectorIndex, build_where

E = [[1.0 if i == j else 0.0 for i in range(8)] for j in range(8)]  # unit basis vectors


def test_build_where():
    assert build_where([], None) is None
    assert build_where(["goa"], None) == {"tag_goa": True}
    assert build_where(["a", "b"], "demo") == {
        "$and": [{"tag_a": True}, {"tag_b": True}, {"source": "demo"}]
    }


def test_upsert_and_query(index):
    a, b = uuid.uuid4(), uuid.uuid4()
    index.upsert(a, image_vector=E[0], tags=["goa"], source="upload")
    index.upsert(b, image_vector=E[1], tags=[], source="upload")
    assert sorted(index.ids()) == sorted([f"{a}:img", f"{b}:img"])
    hits = index.query(E[0], n=10)
    assert [h.image_id for h in hits] == [a, b]
    assert hits[0].similarity == pytest.approx(1.0)
    assert hits[1].similarity == pytest.approx(0.5)  # orthogonal -> distance 1 -> 1 - 1/2


def test_reupsert_replaces_tags(index):
    a = uuid.uuid4()
    index.upsert(a, image_vector=E[0], tags=["goa"], source="upload")
    index.upsert(a, image_vector=E[0], tags=[], source="upload")
    assert index.ids() == [f"{a}:img"]
    assert index.query(E[0], n=10, tags=["goa"]) == []


def test_tag_and_source_filters(index):
    a, b = uuid.uuid4(), uuid.uuid4()
    index.upsert(a, image_vector=E[0], tags=["goa", "beach"], source="upload")
    index.upsert(b, image_vector=E[0], tags=["goa"], source="demo")
    assert {h.image_id for h in index.query(E[0], n=10, tags=["goa"])} == {a, b}
    assert {h.image_id for h in index.query(E[0], n=10, tags=["goa", "beach"])} == {a}
    assert {h.image_id for h in index.query(E[0], n=10, source="demo")} == {b}
    assert index.query(E[0], n=10, tags=["nobody-uses-this"]) == []


def test_query_on_empty_collection(index):
    assert index.query(E[0], n=5) == []


def test_get_image_vector_and_similarities(index):
    a, b = uuid.uuid4(), uuid.uuid4()
    index.upsert(a, image_vector=E[0], tags=[], source="upload")
    index.upsert(b, image_vector=E[2], tags=[], source="upload")
    assert index.get_image_vector(a) == pytest.approx(E[0])
    assert index.get_image_vector(uuid.uuid4()) is None
    sims = index.image_similarities({a, b, uuid.uuid4()}, E[0])
    assert sims[a] == pytest.approx(1.0)
    assert sims[b] == pytest.approx(0.5)
    assert len(sims) == 2


def test_delete(index):
    a = uuid.uuid4()
    index.upsert(a, image_vector=E[0], tags=[], source="upload")
    index.delete(a)
    index.delete(a)  # idempotent
    assert index.ids() == []


def test_model_mismatch(settings, index):
    other = VectorIndex(settings.chroma_host, settings.chroma_port, settings.chroma_collection, "ViT-L/14")
    with pytest.raises(ModelMismatchError):
        other.ensure_collection()
