from fastapi import APIRouter, Query

from app.routers.deps import ServicesDep
from app.schemas import SearchItem, SearchResponse, Source, image_out
from app.tags import parse_tags

router = APIRouter(tags=["search"])


@router.get("/search", response_model=SearchResponse)
def search(
    services: ServicesDep,
    q: str = Query("", max_length=500),
    tags: list[str] = Query(default=[]),
    source: Source | None = None,
    limit: int = Query(40, ge=1, le=200),
):
    tag_list = parse_tags(tags)
    storage = services.storage
    if not q.strip():
        items, _ = services.images.list_images(page=1, page_size=limit, tags=tag_list, source=source, sort="taken")
        return SearchResponse(items=[SearchItem(**image_out(i, storage).model_dump(), score=None) for i in items])
    ranked = services.search.search(q.strip(), tags=tag_list, source=source, limit=limit)
    return SearchResponse(
        items=[SearchItem(**image_out(img, storage).model_dump(), score=score) for img, score in ranked]
    )
