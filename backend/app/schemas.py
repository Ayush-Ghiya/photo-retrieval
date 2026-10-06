from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models import Image
from app.services.storage import Storage

Source = Literal["upload", "demo"]


class ImageOut(BaseModel):
    id: UUID
    filename: str
    title: str | None
    description: str | None
    tags: list[str]
    width: int
    height: int
    taken_at: datetime | None
    created_at: datetime
    source: Source
    indexed: bool
    thumb_url: str


class ImageDetail(ImageOut):
    original_url: str
    mime_type: str


class ImagePage(BaseModel):
    items: list[ImageOut]
    page: int
    page_size: int
    total: int


class SearchItem(ImageOut):
    score: float | None


class SearchResponse(BaseModel):
    items: list[SearchItem]


class TagCount(BaseModel):
    name: str
    count: int


class UploadResultOut(BaseModel):
    filename: str
    status: Literal["created", "duplicate", "error"]
    id: UUID | None = None
    message: str | None = None


class ImagePatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(None, max_length=200)
    description: str | None = Field(None, max_length=2000)
    tags: list[str] | None = None


class UpdateResponse(BaseModel):
    image: ImageDetail
    warning: str | None


def image_out(image: Image, storage: Storage) -> ImageOut:
    return ImageOut(
        id=image.id, filename=image.filename, title=image.title, description=image.description,
        tags=[t.name for t in image.tags], width=image.width, height=image.height,
        taken_at=image.taken_at, created_at=image.created_at, source=image.source,
        indexed=image.indexed, thumb_url=storage.presign("thumbs", image.thumb_key),
    )


def image_detail(image: Image, storage: Storage) -> ImageDetail:
    return ImageDetail(
        **image_out(image, storage).model_dump(),
        original_url=storage.presign("originals", image.s3_key),
        mime_type=image.mime_type,
    )
