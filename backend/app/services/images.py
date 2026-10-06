import logging
from dataclasses import dataclass
from typing import Literal
from uuid import UUID, uuid4

from PIL import Image as PILImage
from sqlalchemy import Select, case, func, or_, select
from sqlalchemy import update as sa_update
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.errors import NotFoundError
from app.models import Image, Tag
from app.repo import get_or_create_tags, tag_counts
from app.services.clip_model import Encoder
from app.services.imaging import UnsupportedImage, open_rgb, process_image
from app.services.storage import Storage
from app.services.vector_index import VectorIndex

logger = logging.getLogger(__name__)

INDEX_WARNING = (
    "Saved, but search indexing failed. Run `python -m app.cli reindex` once ChromaDB is reachable."
)


@dataclass(frozen=True)
class UploadResult:
    filename: str
    status: Literal["created", "duplicate", "error"]
    id: UUID | None = None
    message: str | None = None


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


class ImageService:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        storage: Storage,
        index: VectorIndex,
        encoder: Encoder,
        settings: Settings,
    ):
        self.sessions = sessions
        self.storage = storage
        self.index = index
        self.encoder = encoder
        self.settings = settings

    # ---- upload -------------------------------------------------------------

    def upload(
        self,
        filename: str,
        data: bytes,
        *,
        tags: list[str],
        title: str | None = None,
        description: str | None = None,
        source: str = "upload",
    ) -> UploadResult:
        max_mb = self.settings.max_upload_mb
        if len(data) > max_mb * 1024 * 1024:
            return UploadResult(filename, "error", message=f"File exceeds {max_mb} MB")
        try:
            processed = process_image(data)
        except UnsupportedImage as e:
            return UploadResult(filename, "error", message=str(e))
        except Exception:
            logger.exception("Could not decode %s", filename)
            return UploadResult(filename, "error", message="Could not read image")

        with self.sessions() as s:
            existing = s.scalar(select(Image.id).where(Image.content_hash == processed.content_hash))
        if existing:
            return UploadResult(filename, "duplicate", id=existing, message="Already in your library")

        image_id = uuid4()
        s3_key = f"originals/{image_id}.{processed.ext}"
        thumb_key = f"thumbs/{image_id}.webp"
        try:
            self.storage.put("originals", s3_key, data, processed.mime_type)
            self.storage.put("thumbs", thumb_key, processed.thumbnail, "image/webp")
            with self.sessions.begin() as s:
                image = Image(
                    id=image_id, s3_key=s3_key, thumb_key=thumb_key, filename=filename,
                    mime_type=processed.mime_type, title=_clean(title), description=_clean(description),
                    width=processed.width, height=processed.height, content_hash=processed.content_hash,
                    taken_at=processed.taken_at, source=source, indexed=False,
                )
                image.tags = get_or_create_tags(s, tags)
                s.add(image)
        except Exception:
            logger.exception("Failed to store %s", filename)
            self._remove_objects(s3_key, thumb_key)
            return UploadResult(filename, "error", message="Failed to store image")

        warning = self.index_image(image_id, rgb=processed.rgb)
        return UploadResult(filename, "created", id=image_id, message=warning)

    def _remove_objects(self, s3_key: str, thumb_key: str) -> None:
        for kind, key in (("originals", s3_key), ("thumbs", thumb_key)):
            try:
                self.storage.delete(kind, key)
            except Exception:
                logger.warning("Could not delete s3 object %s/%s", kind, key)

    # ---- indexing -----------------------------------------------------------

    def index_image(self, image_id: UUID, rgb: PILImage.Image | None = None) -> str | None:
        """(Re)write this image's vector and filter metadata. Returns a warning string on failure."""
        image = self.get(image_id)
        tags = [t.name for t in image.tags]
        try:
            if rgb is not None:
                image_vector = self.encoder.encode_images([rgb])[0]
            else:
                image_vector = self.index.get_image_vector(image_id)
                if image_vector is None:
                    original = open_rgb(self.storage.get("originals", image.s3_key))
                    image_vector = self.encoder.encode_images([original])[0]
            self.index.upsert(image_id, image_vector=image_vector, tags=tags, source=image.source)
            indexed, warning = True, None
        except Exception:
            logger.exception("Indexing failed for %s", image_id)
            indexed, warning = False, INDEX_WARNING
        with self.sessions.begin() as s:
            s.execute(sa_update(Image).where(Image.id == image_id).values(indexed=indexed))
        return warning

    def mark_all_unindexed(self) -> None:
        with self.sessions.begin() as s:
            s.execute(sa_update(Image).values(indexed=False))

    def unindexed_ids(self, include_all: bool = False) -> list[UUID]:
        stmt = select(Image.id).order_by(Image.created_at)
        if not include_all:
            stmt = stmt.where(Image.indexed.is_(False))
        with self.sessions() as s:
            return list(s.scalars(stmt))

    # ---- reads --------------------------------------------------------------

    def get(self, image_id: UUID) -> Image:
        with self.sessions() as s:
            image = s.get(Image, image_id)
            if image is None:
                raise NotFoundError()
            return image

    def get_many(self, ids: list[UUID]) -> dict[UUID, Image]:
        if not ids:
            return {}
        with self.sessions() as s:
            return {i.id: i for i in s.scalars(select(Image).where(Image.id.in_(ids)))}

    @staticmethod
    def _filtered(stmt: Select, tags: list[str], source: str | None) -> Select:
        for tag in tags:
            stmt = stmt.where(Image.tags.any(Tag.name == tag))
        if source:
            stmt = stmt.where(Image.source == source)
        return stmt

    def find_by_metadata(
        self, terms: list[str], *, tags: list[str], source: str | None, limit: int
    ) -> list[Image]:
        """Images with a word in title/description/tags starting with any term, most terms matched first.

        Terms are alphanumeric (see text_match.query_terms), so they are safe inside a regex.
        """
        if not terms:
            return []
        per_term = []
        for term in terms:
            pattern = rf"\m{term}"  # Postgres word-start boundary: "art" matches "street-art", not "party"
            per_term.append(
                or_(
                    Image.title.regexp_match(pattern, flags="i"),
                    Image.description.regexp_match(pattern, flags="i"),
                    Image.tags.any(Tag.name.regexp_match(pattern, flags="i")),
                )
            )
        matched = sum(case((cond, 1), else_=0) for cond in per_term)
        stmt = (
            self._filtered(select(Image), tags, source)
            .where(or_(*per_term))
            .order_by(matched.desc(), Image.created_at.desc())
            .limit(limit)
        )
        with self.sessions() as s:
            return list(s.scalars(stmt))

    def list_images(
        self, *, page: int, page_size: int, tags: list[str], source: str | None, sort: str
    ) -> tuple[list[Image], int]:
        stmt = self._filtered(select(Image), tags, source)
        if sort == "uploaded":
            order = Image.created_at.desc()
        else:
            order = func.coalesce(Image.taken_at, Image.created_at).desc()
        with self.sessions() as s:
            total = s.scalar(select(func.count()).select_from(stmt.subquery()))
            items = s.scalars(
                stmt.order_by(order, Image.id).offset((page - 1) * page_size).limit(page_size)
            ).all()
        return list(items), total

    def tag_counts(self) -> list[tuple[str, int]]:
        with self.sessions() as s:
            return tag_counts(s)

    # ---- writes -------------------------------------------------------------

    def update(self, image_id: UUID, changes: dict) -> tuple[Image, str | None]:
        with self.sessions.begin() as s:
            image = s.get(Image, image_id)
            if image is None:
                raise NotFoundError()
            if "title" in changes:
                image.title = _clean(changes["title"])
            if "description" in changes:
                image.description = _clean(changes["description"])
            if "tags" in changes and changes["tags"] is not None:
                image.tags = get_or_create_tags(s, changes["tags"])
        warning = self.index_image(image_id)
        return self.get(image_id), warning

    def delete(self, image_id: UUID) -> None:
        image = self.get(image_id)
        self.index.delete(image_id)
        self._remove_objects(image.s3_key, image.thumb_key)
        with self.sessions.begin() as s:
            row = s.get(Image, image_id)
            if row is not None:
                s.delete(row)
