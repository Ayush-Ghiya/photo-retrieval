import uuid

from tests.helpers import png_bytes


def upload(client, *files, **form):
    return client.post(
        "/api/images",
        files=[("files", (name, data, "image/png")) for name, data in files],
        data=form,
    )


def test_upload_list_detail(client):
    r = upload(client, ("red.png", png_bytes()), tags="Goa Trip, family", title="Beach")
    assert r.status_code == 200
    [res] = r.json()
    assert res["status"] == "created"

    page = client.get("/api/images").json()
    assert page["total"] == 1 and page["page"] == 1 and page["page_size"] == 60
    item = page["items"][0]
    assert item["tags"] == ["family", "goa-trip"]
    assert item["thumb_url"].startswith("http://localhost:4566/")
    assert item["indexed"] is True

    detail = client.get(f"/api/images/{res['id']}").json()
    assert detail["title"] == "Beach" and detail["mime_type"] == "image/png"
    assert "original_url" in detail


def test_batch_upload_mixed_results(client):
    r = upload(client, ("a.png", png_bytes(color=(1, 2, 3))), ("bad.txt", b"nope"), ("a-again.png", png_bytes(color=(1, 2, 3))))
    statuses = [x["status"] for x in r.json()]
    assert statuses == ["created", "error", "duplicate"]


def test_too_many_files(client, settings, monkeypatch):
    monkeypatch.setattr(settings, "max_batch_files", 1)
    r = upload(client, ("a.png", png_bytes(color=(1, 1, 1))), ("b.png", png_bytes(color=(2, 2, 2))))
    assert r.status_code == 400 and r.json()["error"]["code"] == "too_many_files"


def test_patch_updates_and_returns_detail(client):
    [res] = upload(client, ("red.png", png_bytes())).json()
    r = client.patch(f"/api/images/{res['id']}", json={"title": "New", "tags": ["Sun Set"]})
    assert r.status_code == 200
    body = r.json()
    assert body["warning"] is None
    assert body["image"]["title"] == "New" and body["image"]["tags"] == ["sun-set"]


def test_patch_invalid_tag_returns_envelope(client):
    [res] = upload(client, ("red.png", png_bytes())).json()
    r = client.patch(f"/api/images/{res['id']}", json={"tags": ["#bad!"]})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_tag"


def test_patch_rejects_unknown_fields(client):
    [res] = upload(client, ("red.png", png_bytes())).json()
    r = client.patch(f"/api/images/{res['id']}", json={"filename": "hack.png"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "validation_error"


def test_delete_then_404(client):
    [res] = upload(client, ("red.png", png_bytes())).json()
    assert client.delete(f"/api/images/{res['id']}").status_code == 204
    r = client.delete(f"/api/images/{res['id']}")
    assert r.status_code == 404 and r.json() == {"error": {"code": "not_found", "message": "Image not found"}}


def test_get_unknown_and_malformed_id(client):
    assert client.get(f"/api/images/{uuid.uuid4()}").status_code == 404
    assert client.get("/api/images/not-a-uuid").json()["error"]["code"] == "validation_error"


def test_list_validation_error_envelope(client):
    r = client.get("/api/images?page=0")
    assert r.status_code == 400 and r.json()["error"]["code"] == "validation_error"


def test_search_with_query_returns_scores(client):
    upload(client, ("red.png", png_bytes()))
    items = client.get("/api/search?q=red%20car").json()["items"]
    assert len(items) == 1 and 0.0 <= items[0]["score"] <= 1.0


def test_search_empty_query_lists_images(client):
    upload(client, ("red.png", png_bytes()), tags="goa")
    items = client.get("/api/search?q=&tags=goa").json()["items"]
    assert len(items) == 1 and items[0]["score"] is None


def test_search_unknown_tag_is_empty(client):
    upload(client, ("red.png", png_bytes()))
    assert client.get("/api/search?q=red&tags=nobody").json()["items"] == []


def test_tags_endpoint(client):
    upload(client, ("a.png", png_bytes(color=(1, 1, 1))), tags="goa,beach")
    upload(client, ("b.png", png_bytes(color=(2, 2, 2))), tags="goa")
    assert client.get("/api/tags").json() == [{"name": "goa", "count": 2}, {"name": "beach", "count": 1}]


def test_health_ok(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_unknown_route_uses_envelope(client):
    r = client.get("/api/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "http_error"
