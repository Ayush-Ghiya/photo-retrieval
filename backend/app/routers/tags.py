from fastapi import APIRouter

from app.routers.deps import ServicesDep
from app.schemas import TagCount

router = APIRouter(tags=["tags"])


@router.get("/tags", response_model=list[TagCount])
def list_tags(services: ServicesDep):
    return [TagCount(name=n, count=c) for n, c in services.images.tag_counts()]
