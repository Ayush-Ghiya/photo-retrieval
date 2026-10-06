from dataclasses import asdict
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, File, Form, Query, Response, UploadFile

from app.errors import AppError
from app.routers.deps import ServicesDep
from app.schemas import (
    ImageDetail, ImagePage, ImagePatch, Source, UpdateResponse, UploadResultOut, image_detail, image_out,
)
from app.tags import parse_tags

router = APIRouter(prefix="/images", tags=["images"])


@router.get("", response_model=ImagePage)
def list_images(
    services: ServicesDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(60, ge=1, le=200),
    tags: list[str] = Query(default=[]),
    source: Source | None = None,
    sort: Literal["taken", "uploaded"] = "taken",
):
    items, total = services.images.list_images(
        page=page, page_size=page_size, tags=parse_tags(tags), source=source, sort=sort
    )
    return ImagePage(
        items=[image_out(i, services.storage) for i in items], page=page, page_size=page_size, total=total
    )


@router.post("", response_model=list[UploadResultOut])
def upload_images(
    services: ServicesDep,
    files: list[UploadFile] = File(...),
    tags: str | None = Form(None),
    title: str | None = Form(None, max_length=200),
    description: str | None = Form(None, max_length=2000),
):
    max_files = services.settings.max_batch_files
    if len(files) > max_files:
        raise AppError(400, "too_many_files", f"Upload at most {max_files} files at once")
    tag_list = parse_tags(tags)
    limit = services.settings.max_upload_mb * 1024 * 1024
    results = []
    for f in files:
        data = f.file.read(limit + 1)  # +1 so oversize files are detected without reading them fully
        result = services.images.upload(
            f.filename or "upload", data, tags=tag_list, title=title, description=description
        )
        results.append(UploadResultOut(**asdict(result)))
    return results


@router.get("/{image_id}", response_model=ImageDetail)
def get_image(image_id: UUID, services: ServicesDep):
    return image_detail(services.images.get(image_id), services.storage)


@router.patch("/{image_id}", response_model=UpdateResponse)
def update_image(image_id: UUID, body: ImagePatch, services: ServicesDep):
    changes = body.model_dump(include=body.model_fields_set)
    if "tags" in changes:
        changes["tags"] = parse_tags(changes["tags"])
    image, warning = services.images.update(image_id, changes)
    return UpdateResponse(image=image_detail(image, services.storage), warning=warning)


@router.delete("/{image_id}", status_code=204)
def delete_image(image_id: UUID, services: ServicesDep):
    services.images.delete(image_id)
    return Response(status_code=204)
