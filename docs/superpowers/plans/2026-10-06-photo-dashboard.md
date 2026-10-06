# Photo Retrieval Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the CLIP + ChromaDB proof of concept into a single-user photo dashboard with upload, tagging/editing, and blended text search, backed by Dockerised S3 (floci), Postgres and ChromaDB.

**Architecture:** Infra (floci S3, Postgres 16, ChromaDB) runs in `docker compose`; the FastAPI backend (CLIP loaded once, on GPU) and the React/Vite dashboard run natively. Postgres is the source of truth; ChromaDB is a derived index holding an image vector and an optional metadata-text vector per photo; search blends both. The browser loads images straight from S3 via presigned URLs.

**Tech Stack:** Python 3.14 (existing venv, torch nightly cu128), FastAPI, SQLAlchemy 2.0, Alembic, psycopg 3, boto3, chromadb-client 1.5.6, open_clip_torch, Pillow + pillow-heif, pytest · React 19 + TypeScript + Vite, TanStack Query v5, Tailwind CSS v4, sonner, Vitest + Testing Library, Playwright.

**Spec:** `docs/superpowers/specs/2026-10-06-photo-dashboard-design.md`

## Global Constraints

- Only infra runs in Docker locally: floci, Postgres, ChromaDB. The API and web app run natively. Dockerfiles for api/web exist for future hosting only.
- Use the existing venv at `venv/` (Python 3.14, torch `2.12.0.dev…+cu128`). **Never** `pip install torch`/`torchvision` into it; `backend/requirements.txt` excludes them.
- ChromaDB client is `chromadb-client==1.5.6`; server image `chromadb/chroma:1.5.6`.
- S3 emulator: `floci/floci:latest`, `FLOCI_STORAGE_MODE=persistent`, port 4566, path-style addressing.
- Env var names exactly: `DATABASE_URL`, `S3_ENDPOINT_URL`, `S3_PUBLIC_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `S3_BUCKET_ORIGINALS`, `S3_BUCKET_THUMBS`, `CHROMA_HOST`, `CHROMA_PORT`, `CLIP_MODEL`, `SEARCH_IMG_WEIGHT`, `MAX_UPLOAD_MB`, `CORS_ORIGINS`.
- Defaults: buckets `photos-originals` / `photos-thumbs`; collection `photos_v1`; `CLIP_MODEL=ViT-B/32`; `SEARCH_IMG_WEIGHT=0.7`; `MAX_UPLOAD_MB=25`; batch 1–50 files; presigned URL TTL 3600 s; thumbnails WebP, longest side 512 px.
- Tags: trim, lowercase, whitespace runs → `-`, must match `^[a-z0-9-]{1,40}$`.
- Chroma ids `<uuid>:img` / `<uuid>:txt`; metadata `image_id`, `kind`, `source`, `tag_<name>=True`. Similarity `s = 1 − distance/2`. Score `w·img + (1−w)·txt`, or `img` when no txt vector.
- Search: `limit` default 40, max 200; Chroma `n_results = limit × 3`. Listing: `page_size` default 60, max 200.
- API error envelope: `{"error": {"code": "...", "message": "..."}}`.
- API runs on port 8080; Vite dev server on 5173 and proxies `/api` → `http://localhost:8080`.
- Backend commands run from `backend/` using `../venv/Scripts/python` (Git Bash). Frontend commands run from `frontend/`.
- Every commit message ends with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Corrupt / non-image / oversize file inside a batch.** That file reports `error`, nothing is left behind in S3 or Postgres, and the other files still upload. → Task 8 (`test_upload_rejects_non_image_and_leaves_nothing`, `test_upload_oversize`, `test_storage_failure_cleans_up_original`) and Task 10 (`test_batch_upload_mixed_results`).
2. **Messy tag input** (`"  Goa Trip "`, duplicates, `"#fun!"`). Valid tags are normalised and deduplicated; invalid ones get a 400 with the `invalid_tag` code, never a 500. → Task 3 (`test_parse_tags_*`) and Task 10 (`test_patch_invalid_tag_returns_envelope`).
3. **Removing all metadata from an image.** The stale `:txt` vector is deleted, so the old description stops matching searches. → Task 8 (`test_update_clearing_metadata_removes_text_vector`).
4. **ChromaDB down during upload or edit.** The photo is still saved with `indexed=false` and a warning, and `reindex` repairs it later. → Task 8 (`test_upload_when_index_down_keeps_row_unindexed`) and Task 11 (`test_reindex_catches_up_unindexed`).
5. **Search or filter on a tag nobody uses, or a search with an empty query.** These return an empty list or fall back to the date-ordered listing, never an error. → Task 9 (`test_search_unknown_tag_returns_empty`) and Task 10 (`test_search_empty_query_lists_images`).

---

## File Structure

```
docker-compose.yml                 # floci, postgres, chromadb (+ volumes)
docker/postgres/init.sql           # creates photos_test DB
.env.example                       # all env vars, local values
.gitignore                         # extended
README.md                          # setup / run / seed / test / deploy
backend/
  requirements.txt                 # runtime deps (no torch)
  requirements-dev.txt             # + pytest
  pytest.ini
  alembic.ini
  alembic/env.py
  alembic/script.py.mako
  alembic/versions/0001_initial.py
  Dockerfile, .dockerignore
  app/__init__.py
  app/config.py                    # Settings (pydantic-settings), get_settings()
  app/db.py                        # Base, make_session_factory()
  app/models.py                    # Image, Tag, image_tags
  app/tags.py                      # normalize_tag, parse_tags, InvalidTag
  app/repo.py                      # get_or_create_tags, tag_counts
  app/errors.py                    # AppError, NotFoundError, handlers
  app/schemas.py                   # Pydantic API models + image_out/image_detail
  app/main.py                      # create_app(), app
  app/cli.py                       # reindex, seed-demo
  app/routers/__init__.py
  app/routers/deps.py              # get_services, ServicesDep
  app/routers/images.py
  app/routers/search.py
  app/routers/tags.py
  app/routers/health.py
  app/services/__init__.py
  app/services/imaging.py          # process_image, open_rgb, UnsupportedImage
  app/services/storage.py          # Storage (S3)
  app/services/clip_model.py       # Encoder protocol, ClipEncoder
  app/services/vector_index.py     # VectorIndex, Hit, ModelMismatchError, build_where
  app/services/images.py           # ImageService, UploadResult, build_text_document
  app/services/search.py           # blend, SearchService
  app/services/container.py        # Services, build_services
  tests/conftest.py
  tests/fakes.py                   # FakeEncoder, BrokenIndex
  tests/helpers.py                 # png_bytes, jpeg_with_exif
  tests/test_*.py
frontend/
  package.json, tsconfig.json, vite.config.ts, index.html, playwright.config.ts
  Dockerfile, .dockerignore, nginx.conf.template
  src/main.tsx, src/index.css, src/App.tsx, src/vite-env.d.ts
  src/api/types.ts, src/api/client.ts
  src/lib/tags.ts
  src/hooks/useSearchState.ts
  src/components/TagInput.tsx, SearchBar.tsx, Gallery.tsx, ImageCard.tsx,
                 HealthBanner.tsx, DetailDrawer.tsx, UploadDialog.tsx
  src/test/setup.ts, src/test/utils.tsx
  src/**/*.test.ts(x)
  e2e/smoke.spec.ts, e2e/fixtures/red.png
Deleted: app.py build_dataset.py build_index.py clip_image_search.py search.py search_LEGACY.py utils.py
```

---

### Task 1: Docker infrastructure and env config

**Files:**
- Create: `docker-compose.yml`, `docker/postgres/init.sql`, `.env.example`
- Modify: `.gitignore`

**Interfaces:**
- Produces: S3 at `http://localhost:4566`, Postgres `photos:photos@localhost:5432` with DBs `photos` and `photos_test`, Chroma at `localhost:8000`.

- [ ] **Step 1: Start Docker Desktop**

The daemon is currently not running. Start Docker Desktop, then check:

Run: `docker info --format '{{.ServerVersion}}'`
Expected: prints a version (no "failed to connect").

- [ ] **Step 2: Write `docker-compose.yml`**

```yaml
name: photo-retrieval

services:
  floci:
    image: floci/floci:latest
    ports:
      - "4566:4566"
    environment:
      FLOCI_STORAGE_MODE: persistent
      FLOCI_STORAGE_PERSISTENT_PATH: /data
    volumes:
      - floci-data:/data
    restart: unless-stopped

  postgres:
    image: postgres:16-alpine
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: photos
      POSTGRES_PASSWORD: photos
      POSTGRES_DB: photos
    volumes:
      - pg-data:/var/lib/postgresql/data
      - ./docker/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U photos"]
      interval: 5s
      timeout: 3s
      retries: 10
    restart: unless-stopped

  chromadb:
    image: chromadb/chroma:1.5.6
    ports:
      - "8000:8000"
    environment:
      IS_PERSISTENT: "TRUE"
      ANONYMIZED_TELEMETRY: "FALSE"
    volumes:
      - chroma-data:/data
    restart: unless-stopped

volumes:
  floci-data:
  pg-data:
  chroma-data:
```

- [ ] **Step 3: Write `docker/postgres/init.sql`**

```sql
CREATE DATABASE photos_test OWNER photos;
```

- [ ] **Step 4: Write `.env.example`, then copy it to `.env`**

```dotenv
# --- Postgres (source of truth) ---
DATABASE_URL=postgresql+psycopg://photos:photos@localhost:5432/photos

# --- S3 (floci locally; unset S3_ENDPOINT_URL for real AWS) ---
S3_ENDPOINT_URL=http://localhost:4566
# Endpoint used in presigned URLs given to the browser (defaults to S3_ENDPOINT_URL)
S3_PUBLIC_ENDPOINT_URL=
AWS_ACCESS_KEY_ID=test
AWS_SECRET_ACCESS_KEY=test
AWS_REGION=us-east-1
S3_BUCKET_ORIGINALS=photos-originals
S3_BUCKET_THUMBS=photos-thumbs

# --- ChromaDB (derived vector index) ---
CHROMA_HOST=localhost
CHROMA_PORT=8000

# --- Search / model ---
CLIP_MODEL=ViT-B/32
SEARCH_IMG_WEIGHT=0.7

# --- API ---
MAX_UPLOAD_MB=25
CORS_ORIGINS=http://localhost:5173
```

Run: `cp .env.example .env`. This replaces the old `.env`, whose `CHROMA_DB_*` keys are no longer used.

- [ ] **Step 5: Extend `.gitignore`**

Append:

```
node_modules/
frontend/dist/
.pytest_cache/
test-results/
playwright-report/
*.egg-info/
```

- [ ] **Step 6: Bring up the infra and verify each service**

Run: `docker compose up -d && docker compose ps`
Expected: `floci`, `postgres` (healthy) and `chromadb` are all `running`.

If `chromadb/chroma:1.5.6` fails to pull, run `docker search chromadb/chroma` or check Docker Hub for the closest `1.5.x` tag, then use that tag in both the compose file and `requirements.txt` (`chromadb-client==<same>`).

Run: `curl -s http://localhost:8000/api/v2/heartbeat`
Expected: JSON containing `nanosecond heartbeat`.

Run: `docker compose exec postgres psql -U photos -lqt | cut -d'|' -f1 | grep -w photos_test`
Expected: `photos_test`.

Run: `curl -s -o /dev/null -w '%{http_code}\n' http://localhost:4566/`
Expected: an HTTP status code (any of 200/403/404), which proves floci is listening. Full S3 verification comes in Task 5.

- [ ] **Step 7: Commit**

```bash
git add docker-compose.yml docker/postgres/init.sql .env.example .gitignore
git commit -m "chore: add docker compose infra (floci, postgres, chromadb)"
```

---

### Task 2: Backend scaffold and settings

**Files:**
- Create: `backend/requirements.txt`, `backend/requirements-dev.txt`, `backend/pytest.ini`, `backend/app/__init__.py` (empty), `backend/app/config.py`
- Test: `backend/tests/test_config.py`

**Interfaces:**
- Produces: `app.config.Settings` (fields below), `Settings.cors_origin_list: list[str]`, `Settings.s3_public_endpoint: str | None`, `app.config.get_settings() -> Settings` (cached).

- [ ] **Step 1: Write the requirements and pytest config**

`backend/requirements.txt`:

```
# torch / torchvision are installed separately (GPU build locally, CPU build in Dockerfile)
fastapi>=0.115
uvicorn[standard]>=0.30
python-multipart>=0.0.9
sqlalchemy>=2.0.30
psycopg[binary]>=3.2
alembic>=1.13
pydantic>=2.8
pydantic-settings>=2.4
boto3>=1.35
chromadb-client==1.5.6
open_clip_torch>=2.26
pillow>=11
pillow-heif>=0.18
numpy>=2
```

`backend/requirements-dev.txt`:

```
-r requirements.txt
pytest>=8
httpx>=0.27
```

`backend/pytest.ini`:

```ini
[pytest]
pythonpath = .
testpaths = tests
addopts = -m "not slow"
markers =
    slow: uses the real CLIP model (run with: pytest -m slow)
```

Run (from `backend/`): `../venv/Scripts/python -m pip install -r requirements-dev.txt`
Expected: finishes successfully, and torch is not reinstalled. Confirm with `../venv/Scripts/python -c "import torch;print(torch.__version__)"`, which should print `2.12.0.dev…+cu128`.
If `pillow-heif` has no wheel for Python 3.14, remove that line from `requirements.txt`. HEIC is then skipped automatically, because `imaging.py` guards the import (Task 4).

- [ ] **Step 2: Write the failing test**

`backend/tests/test_config.py`:

```python
import pytest
from pydantic import ValidationError

from app.config import Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.s3_bucket_originals == "photos-originals"
    assert s.s3_bucket_thumbs == "photos-thumbs"
    assert s.chroma_collection == "photos_v1"
    assert s.clip_model == "ViT-B/32"
    assert s.search_img_weight == 0.7
    assert s.max_upload_mb == 25
    assert s.max_batch_files == 50
    assert s.presign_ttl_seconds == 3600


def test_env_overrides_and_derived_values(monkeypatch):
    monkeypatch.setenv("SEARCH_IMG_WEIGHT", "0.5")
    monkeypatch.setenv("CORS_ORIGINS", "http://a:1, http://b:2")
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://localhost:4566")
    monkeypatch.delenv("S3_PUBLIC_ENDPOINT_URL", raising=False)
    s = Settings(_env_file=None)
    assert s.search_img_weight == 0.5
    assert s.cors_origin_list == ["http://a:1", "http://b:2"]
    assert s.s3_public_endpoint == "http://localhost:4566"


def test_empty_public_endpoint_falls_back(monkeypatch):
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://localhost:4566")
    monkeypatch.setenv("S3_PUBLIC_ENDPOINT_URL", "")
    assert Settings(_env_file=None).s3_public_endpoint == "http://localhost:4566"


def test_weight_must_be_between_0_and_1(monkeypatch):
    monkeypatch.setenv("SEARCH_IMG_WEIGHT", "1.5")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `../venv/Scripts/python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.config'`.

- [ ] **Step 4: Implement `backend/app/config.py`**

```python
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration, read from env vars / .env (repo root or backend/)."""

    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore")

    database_url: str = "postgresql+psycopg://photos:photos@localhost:5432/photos"

    s3_endpoint_url: str | None = None
    s3_public_endpoint_url: str | None = None
    aws_access_key_id: str = "test"
    aws_secret_access_key: str = "test"
    aws_region: str = "us-east-1"
    s3_bucket_originals: str = "photos-originals"
    s3_bucket_thumbs: str = "photos-thumbs"
    presign_ttl_seconds: int = 3600

    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "photos_v1"

    clip_model: str = "ViT-B/32"
    search_img_weight: float = Field(0.7, ge=0.0, le=1.0)

    max_upload_mb: int = 25
    max_batch_files: int = 50
    cors_origins: str = "http://localhost:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def s3_public_endpoint(self) -> str | None:
        return self.s3_public_endpoint_url or self.s3_endpoint_url


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `../venv/Scripts/python -m pytest tests/test_config.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/
git commit -m "feat(backend): scaffold backend package and settings"
```

---

### Task 3: Database models, tags, repository and Alembic migration

**Files:**
- Create: `backend/app/db.py`, `backend/app/models.py`, `backend/app/tags.py`, `backend/app/repo.py`, `backend/alembic.ini`, `backend/alembic/env.py`, `backend/alembic/script.py.mako`, `backend/alembic/versions/0001_initial.py`, `backend/tests/conftest.py`
- Test: `backend/tests/test_tags.py`, `backend/tests/test_models.py`

**Interfaces:**
- Consumes: `Settings` (Task 2).
- Produces:
  - `app.db.Base`, `app.db.make_session_factory(url: str) -> sessionmaker[Session]` (`expire_on_commit=False`)
  - `app.models.Image` (columns per spec §4; `tags: list[Tag]`, loaded with selectin and ordered by name), `app.models.Tag(id, name)`, `app.models.image_tags`
  - `app.tags.InvalidTag(ValueError)`, `normalize_tag(raw: str) -> str`, `parse_tags(raw: str | list[str] | None) -> list[str]` (sorted, unique)
  - `app.repo.get_or_create_tags(session, names: list[str]) -> list[Tag]`, `app.repo.tag_counts(session) -> list[tuple[str, int]]`
  - conftest fixtures `settings` (session scope) and `sessions` (fresh schema for each test)

- [ ] **Step 1: Write the failing tag tests**

`backend/tests/test_tags.py`:

```python
import pytest

from app.tags import InvalidTag, normalize_tag, parse_tags


@pytest.mark.parametrize(
    "raw,expected",
    [("Goa", "goa"), ("  Goa Trip ", "goa-trip"), ("new\tyear  2024", "new-year-2024"), ("a-b", "a-b")],
)
def test_normalize_tag(raw, expected):
    assert normalize_tag(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "#fun!", "café", "x" * 41])
def test_normalize_tag_rejects_invalid(raw):
    with pytest.raises(InvalidTag):
        normalize_tag(raw)


def test_parse_tags_from_comma_string_dedupes_and_sorts():
    assert parse_tags("Family, goa trip,family,, ") == ["family", "goa-trip"]


def test_parse_tags_from_list_and_none():
    assert parse_tags(["B", "a"]) == ["a", "b"]
    assert parse_tags(None) == []
    assert parse_tags([]) == []


def test_parse_tags_raises_on_any_invalid():
    with pytest.raises(InvalidTag):
        parse_tags("ok, #bad")
```

- [ ] **Step 2: Run to verify it fails**

Run: `../venv/Scripts/python -m pytest tests/test_tags.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.tags'`.

- [ ] **Step 3: Implement `backend/app/tags.py`**

```python
import re

_VALID = re.compile(r"[a-z0-9-]{1,40}")


class InvalidTag(ValueError):
    pass


def normalize_tag(raw: str) -> str:
    name = re.sub(r"\s+", "-", raw.strip().lower())
    if not _VALID.fullmatch(name):
        raise InvalidTag(f"Invalid tag {raw!r}: use 1-40 letters, digits or hyphens")
    return name


def parse_tags(raw: str | list[str] | None) -> list[str]:
    if raw is None:
        return []
    items = raw.split(",") if isinstance(raw, str) else raw
    return sorted({normalize_tag(t) for t in items if t.strip()})
```

- [ ] **Step 4: Run to verify it passes**

Run: `../venv/Scripts/python -m pytest tests/test_tags.py -v`
Expected: all passed.

- [ ] **Step 5: Write `db.py`, `models.py`, `repo.py`**

`backend/app/db.py`:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


def make_session_factory(url: str) -> sessionmaker[Session]:
    engine = create_engine(url, pool_pre_ping=True)
    return sessionmaker(bind=engine, expire_on_commit=False)
```

`backend/app/models.py`:

```python
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer, Table, Text, Uuid, false, func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

image_tags = Table(
    "image_tags",
    Base.metadata,
    Column("image_id", Uuid, ForeignKey("images.id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id", Integer, ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True),
)


class Tag(Base):
    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)


class Image(Base):
    __tablename__ = "images"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    s3_key: Mapped[str] = mapped_column(Text, nullable=False)
    thumb_key: Mapped[str] = mapped_column(Text, nullable=False)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    mime_type: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    taken_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source: Mapped[str] = mapped_column(Text, nullable=False)
    indexed: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=false(), default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    tags: Mapped[list[Tag]] = relationship(
        secondary=image_tags, lazy="selectin", order_by="Tag.name"
    )
```

`backend/app/repo.py`:

```python
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Tag, image_tags


def get_or_create_tags(session: Session, names: list[str]) -> list[Tag]:
    """Return Tag rows for already-normalised names, creating missing ones."""
    if not names:
        return []
    existing = {t.name: t for t in session.scalars(select(Tag).where(Tag.name.in_(names)))}
    tags = []
    for name in names:
        tag = existing.get(name)
        if tag is None:
            tag = Tag(name=name)
            session.add(tag)
            existing[name] = tag
        tags.append(tag)
    return tags


def tag_counts(session: Session) -> list[tuple[str, int]]:
    """Tags used by at least one image, most used first."""
    count = func.count(image_tags.c.image_id)
    stmt = (
        select(Tag.name, count)
        .join(image_tags, image_tags.c.tag_id == Tag.id)
        .group_by(Tag.name)
        .order_by(count.desc(), Tag.name)
    )
    return [(name, n) for name, n in session.execute(stmt)]
```

- [ ] **Step 6: Write `backend/tests/conftest.py`**

```python
import os

import pytest

import app.models  # noqa: F401  (registers tables on Base.metadata)
from app.config import Settings
from app.db import Base, make_session_factory

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL", "postgresql+psycopg://photos:photos@localhost:5432/photos_test"
)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(
        _env_file=None,
        database_url=TEST_DATABASE_URL,
        s3_endpoint_url="http://localhost:4566",
        s3_bucket_originals="test-photos-originals",
        s3_bucket_thumbs="test-photos-thumbs",
        chroma_collection="photos_test",
        max_upload_mb=1,
    )


@pytest.fixture(scope="session")
def _session_factory(settings):
    return make_session_factory(settings.database_url)


@pytest.fixture
def sessions(_session_factory):
    engine = _session_factory.kw["bind"]
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return _session_factory
```

- [ ] **Step 7: Write the failing model tests**

`backend/tests/test_models.py`:

```python
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.models import Image, Tag, image_tags
from app.repo import get_or_create_tags, tag_counts


def make_image(**overrides) -> Image:
    values = dict(
        id=uuid.uuid4(), s3_key="originals/x.png", thumb_key="thumbs/x.webp",
        filename="x.png", mime_type="image/png", width=10, height=10,
        content_hash=uuid.uuid4().hex, source="upload",
    )
    values.update(overrides)
    return Image(**values)


def test_image_with_tags_roundtrip(sessions):
    with sessions.begin() as s:
        img = make_image()
        img.tags = get_or_create_tags(s, ["goa", "beach"])
        s.add(img)
    with sessions() as s:
        loaded = s.get(Image, img.id)
        assert [t.name for t in loaded.tags] == ["beach", "goa"]
        assert loaded.indexed is False
        assert loaded.created_at is not None


def test_get_or_create_tags_reuses_existing(sessions):
    with sessions.begin() as s:
        first = get_or_create_tags(s, ["goa"])
    with sessions.begin() as s:
        again = get_or_create_tags(s, ["goa", "new"])
        assert again[0].id == first[0].id
    with sessions() as s:
        assert sorted(s.scalars(select(Tag.name))) == ["goa", "new"]


def test_content_hash_is_unique(sessions):
    with sessions.begin() as s:
        s.add(make_image(content_hash="same"))
    with pytest.raises(IntegrityError):
        with sessions.begin() as s:
            s.add(make_image(content_hash="same"))


def test_deleting_image_removes_links_and_counts(sessions):
    with sessions.begin() as s:
        a, b = make_image(), make_image()
        a.tags = get_or_create_tags(s, ["goa", "family"])
        b.tags = get_or_create_tags(s, ["goa"])
        s.add_all([a, b])
    with sessions() as s:
        assert tag_counts(s) == [("goa", 2), ("family", 1)]
    with sessions.begin() as s:
        s.delete(s.get(Image, a.id))
    with sessions() as s:
        assert tag_counts(s) == [("goa", 1)]
        assert s.execute(select(image_tags)).all() != []
```

- [ ] **Step 8: Run all tests**

Run: `../venv/Scripts/python -m pytest -v`
Expected: all passed. This requires the compose Postgres to be running.

- [ ] **Step 9: Add Alembic**

`backend/alembic.ini`:

```ini
[alembic]
script_location = alembic
prepend_sys_path = .

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARNING
handlers = console

[logger_sqlalchemy]
level = WARNING
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
```

`backend/alembic/env.py`:

```python
from alembic import context
from sqlalchemy import create_engine

import app.models  # noqa: F401
from app.config import get_settings
from app.db import Base

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(url=get_settings().database_url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(get_settings().database_url)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

`backend/alembic/script.py.mako`:

```mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = None
depends_on = None


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

`backend/alembic/versions/0001_initial.py`:

```python
"""initial schema

Revision ID: 0001
Revises:
"""
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "images",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("s3_key", sa.Text(), nullable=False),
        sa.Column("thumb_key", sa.Text(), nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("mime_type", sa.Text(), nullable=False),
        sa.Column("title", sa.Text()),
        sa.Column("description", sa.Text()),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False, unique=True),
        sa.Column("taken_at", sa.DateTime(timezone=True)),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("indexed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "tags",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False, unique=True),
    )
    op.create_table(
        "image_tags",
        sa.Column("image_id", sa.Uuid(), sa.ForeignKey("images.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("tag_id", sa.Integer(), sa.ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True),
    )


def downgrade() -> None:
    op.drop_table("image_tags")
    op.drop_table("tags")
    op.drop_table("images")
```

- [ ] **Step 10: Apply the migration to the dev DB and check it matches the models**

Run: `../venv/Scripts/python -m alembic upgrade head && ../venv/Scripts/python -m alembic check`
Expected: `Running upgrade  -> 0001, initial schema`, then `No new upgrade operations detected.`

- [ ] **Step 11: Commit**

```bash
git add backend/
git commit -m "feat(backend): add image/tag models, tag normalisation and initial migration"
```

---

### Task 4: Image processing (validation, EXIF, thumbnails)

**Files:**
- Create: `backend/app/services/__init__.py` (empty), `backend/app/services/imaging.py`, `backend/tests/helpers.py`
- Test: `backend/tests/test_imaging.py`

**Interfaces:**
- Produces:
  - `UnsupportedImage(Exception)`
  - `ProcessedImage` (frozen dataclass): `content_hash: str`, `mime_type: str`, `ext: str`, `width: int`, `height: int`, `taken_at: datetime | None` (UTC), `thumbnail: bytes` (WebP), `rgb: PIL.Image.Image` (oriented RGB)
  - `process_image(data: bytes, thumb_size: int = 512) -> ProcessedImage`
  - `open_rgb(data: bytes) -> PIL.Image.Image`, which applies EXIF orientation and converts to RGB
  - `sha256_hex(data: bytes) -> str`
  - test helpers `png_bytes(color=(220, 20, 20), size=(64, 64)) -> bytes` and `jpeg_with_exif(size, orientation, taken) -> bytes`

- [ ] **Step 1: Write the test helpers**

`backend/tests/helpers.py`:

```python
import io

from PIL import Image


def png_bytes(color=(220, 20, 20), size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def jpeg_with_exif(size=(200, 100), orientation: int = 1, taken: str | None = None) -> bytes:
    img = Image.new("RGB", size, (10, 120, 200))
    exif = Image.Exif()
    exif[0x0112] = orientation
    if taken:
        exif.get_ifd(0x8769)[36867] = taken  # DateTimeOriginal
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()
```

- [ ] **Step 2: Write the failing tests**

`backend/tests/test_imaging.py`:

```python
import hashlib
import io
from datetime import datetime, timezone

import pytest
from PIL import Image

from app.services.imaging import UnsupportedImage, open_rgb, process_image
from tests.helpers import jpeg_with_exif, png_bytes


def test_png_basic_properties():
    data = png_bytes(size=(64, 32))
    p = process_image(data)
    assert p.content_hash == hashlib.sha256(data).hexdigest()
    assert (p.mime_type, p.ext) == ("image/png", "png")
    assert (p.width, p.height) == (64, 32)
    assert p.taken_at is None
    thumb = Image.open(io.BytesIO(p.thumbnail))
    assert thumb.format == "WEBP"
    assert p.rgb.mode == "RGB"


def test_jpeg_orientation_and_taken_at():
    data = jpeg_with_exif(size=(200, 100), orientation=6, taken="2024:05:01 10:20:30")
    p = process_image(data)
    assert (p.mime_type, p.ext) == ("image/jpeg", "jpg")
    assert (p.width, p.height) == (100, 200)  # rotated 90°
    assert p.rgb.size == (100, 200)
    assert p.taken_at == datetime(2024, 5, 1, 10, 20, 30, tzinfo=timezone.utc)


def test_bad_exif_date_is_ignored():
    p = process_image(jpeg_with_exif(taken="not a date"))
    assert p.taken_at is None


def test_thumbnail_longest_side_is_512():
    p = process_image(png_bytes(size=(2000, 1000)))
    assert Image.open(io.BytesIO(p.thumbnail)).size == (512, 256)


def test_small_image_is_not_upscaled():
    p = process_image(png_bytes(size=(32, 32)))
    assert Image.open(io.BytesIO(p.thumbnail)).size == (32, 32)


def test_non_image_rejected():
    with pytest.raises(UnsupportedImage):
        process_image(b"definitely not an image")


def test_disallowed_format_rejected():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="GIF")
    with pytest.raises(UnsupportedImage):
        process_image(buf.getvalue())


def test_truncated_jpeg_rejected():
    data = jpeg_with_exif(size=(400, 400))
    with pytest.raises(UnsupportedImage):
        process_image(data[: len(data) // 2])


def test_open_rgb_applies_orientation():
    img = open_rgb(jpeg_with_exif(size=(200, 100), orientation=6))
    assert img.size == (100, 200) and img.mode == "RGB"
```

- [ ] **Step 3: Run to verify it fails**

Run: `../venv/Scripts/python -m pytest tests/test_imaging.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.imaging'`.

- [ ] **Step 4: Implement `backend/app/services/imaging.py`**

```python
import hashlib
import io
from dataclasses import dataclass
from datetime import datetime, timezone

from PIL import Image, ImageOps, UnidentifiedImageError

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # HEIC support is optional
    pillow_heif = None

# Pillow format name -> (mime type, file extension)
ALLOWED_FORMATS = {
    "JPEG": ("image/jpeg", "jpg"),
    "PNG": ("image/png", "png"),
    "WEBP": ("image/webp", "webp"),
    "HEIF": ("image/heic", "heic"),
}
_EXIF_IFD = 0x8769
_DATETIME_ORIGINAL = 36867
_DATETIME = 306


class UnsupportedImage(Exception):
    pass


@dataclass(frozen=True)
class ProcessedImage:
    content_hash: str
    mime_type: str
    ext: str
    width: int
    height: int
    taken_at: datetime | None
    thumbnail: bytes
    rgb: Image.Image


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _open(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise UnsupportedImage("File is not a readable image") from e
    return img


def open_rgb(data: bytes) -> Image.Image:
    """Decode bytes into an EXIF-oriented RGB image (used for CLIP encoding)."""
    return ImageOps.exif_transpose(_open(data)).convert("RGB")


def _taken_at(img: Image.Image) -> datetime | None:
    exif = img.getexif()
    raw = exif.get_ifd(_EXIF_IFD).get(_DATETIME_ORIGINAL) or exif.get(_DATETIME)
    if not isinstance(raw, str):
        return None
    try:
        return datetime.strptime(raw.strip(), "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def process_image(data: bytes, thumb_size: int = 512) -> ProcessedImage:
    img = _open(data)
    if img.format not in ALLOWED_FORMATS:
        raise UnsupportedImage(f"Unsupported image type {img.format}; use JPEG, PNG, WebP or HEIC")
    mime_type, ext = ALLOWED_FORMATS[img.format]
    taken_at = _taken_at(img)

    rgb = ImageOps.exif_transpose(img).convert("RGB")
    thumb = rgb.copy()
    thumb.thumbnail((thumb_size, thumb_size))
    buf = io.BytesIO()
    thumb.save(buf, format="WEBP", quality=80)

    return ProcessedImage(
        content_hash=sha256_hex(data),
        mime_type=mime_type,
        ext=ext,
        width=rgb.width,
        height=rgb.height,
        taken_at=taken_at,
        thumbnail=buf.getvalue(),
        rgb=rgb,
    )
```

- [ ] **Step 5: Run to verify it passes**

Run: `../venv/Scripts/python -m pytest tests/test_imaging.py -v`
Expected: all passed. If `test_jpeg_orientation_and_taken_at` fails only on `taken_at`, the installed Pillow is not writing the nested EXIF IFD. In that case change the helper to also set `exif[306] = taken`; `_taken_at` already falls back to tag 306.

- [ ] **Step 6: Commit**

```bash
git add backend/
git commit -m "feat(backend): image validation, EXIF orientation/date and WebP thumbnails"
```

---

### Task 5: S3 storage service (floci)

**Files:**
- Create: `backend/app/services/storage.py`
- Modify: `backend/tests/conftest.py` (add the `storage` fixture)
- Test: `backend/tests/test_storage.py`

**Interfaces:**
- Consumes: `Settings`.
- Produces: `Kind = Literal["originals", "thumbs"]`; class `Storage(settings)` with `client` (boto3 S3 client), `bucket(kind) -> str`, `ensure_buckets() -> None`, `put(kind, key, data: bytes, content_type: str) -> None`, `get(kind, key) -> bytes`, `delete(kind, key) -> None` (no error if the key is missing), `presign(kind, key) -> str`, `ping() -> None`.

- [ ] **Step 1: Add the fixture to `conftest.py`**

Append:

```python
from app.services.storage import Storage


def _empty_bucket(storage: Storage, kind: str) -> None:
    bucket = storage.bucket(kind)
    for obj in storage.client.list_objects_v2(Bucket=bucket).get("Contents", []):
        storage.client.delete_object(Bucket=bucket, Key=obj["Key"])


@pytest.fixture
def storage(settings) -> Storage:
    s = Storage(settings)
    s.ensure_buckets()
    _empty_bucket(s, "originals")
    _empty_bucket(s, "thumbs")
    return s
```

- [ ] **Step 2: Write the failing tests**

`backend/tests/test_storage.py`:

```python
import httpx
import pytest
from botocore.exceptions import ClientError


def test_put_get_delete_roundtrip(storage):
    storage.put("originals", "originals/a.png", b"hello", "image/png")
    assert storage.get("originals", "originals/a.png") == b"hello"
    storage.delete("originals", "originals/a.png")
    with pytest.raises(ClientError):
        storage.get("originals", "originals/a.png")


def test_delete_missing_key_is_noop(storage):
    storage.delete("thumbs", "thumbs/does-not-exist.webp")


def test_ensure_buckets_is_idempotent(storage):
    storage.ensure_buckets()
    storage.ping()


def test_presigned_url_serves_object(storage):
    storage.put("thumbs", "thumbs/t.webp", b"thumb-bytes", "image/webp")
    url = storage.presign("thumbs", "thumbs/t.webp")
    assert url.startswith("http://localhost:4566/")
    resp = httpx.get(url)
    assert resp.status_code == 200
    assert resp.content == b"thumb-bytes"
```

- [ ] **Step 3: Run to verify it fails**

Run: `../venv/Scripts/python -m pytest tests/test_storage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.storage'`.

- [ ] **Step 4: Implement `backend/app/services/storage.py`**

```python
from typing import Literal

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from app.config import Settings

Kind = Literal["originals", "thumbs"]


class Storage:
    """S3 access. Works against floci locally and real S3 when S3_ENDPOINT_URL is unset."""

    def __init__(self, settings: Settings):
        self._buckets: dict[str, str] = {
            "originals": settings.s3_bucket_originals,
            "thumbs": settings.s3_bucket_thumbs,
        }
        self._ttl = settings.presign_ttl_seconds
        self._region = settings.aws_region
        common = dict(
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region,
            config=BotoConfig(signature_version="s3v4", s3={"addressing_style": "path"}),
        )
        self.client = boto3.client("s3", endpoint_url=settings.s3_endpoint_url, **common)
        # Separate client so presigned URLs use the browser-reachable endpoint.
        self._signer = boto3.client("s3", endpoint_url=settings.s3_public_endpoint, **common)

    def bucket(self, kind: Kind) -> str:
        return self._buckets[kind]

    def ensure_buckets(self) -> None:
        for bucket in self._buckets.values():
            try:
                self.client.head_bucket(Bucket=bucket)
            except ClientError:
                kwargs = {}
                if self._region != "us-east-1":
                    kwargs["CreateBucketConfiguration"] = {"LocationConstraint": self._region}
                self.client.create_bucket(Bucket=bucket, **kwargs)

    def put(self, kind: Kind, key: str, data: bytes, content_type: str) -> None:
        self.client.put_object(Bucket=self.bucket(kind), Key=key, Body=data, ContentType=content_type)

    def get(self, kind: Kind, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket(kind), Key=key)["Body"].read()

    def delete(self, kind: Kind, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket(kind), Key=key)

    def presign(self, kind: Kind, key: str) -> str:
        return self._signer.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket(kind), "Key": key}, ExpiresIn=self._ttl
        )

    def ping(self) -> None:
        self.client.head_bucket(Bucket=self._buckets["originals"])
```

- [ ] **Step 5: Run to verify it passes**

Run: `../venv/Scripts/python -m pytest tests/test_storage.py -v`
Expected: 4 passed.

- [ ] **Step 6: Verify persistence across a restart**

Run: `docker compose restart floci` and wait about 3 s. Then run `../venv/Scripts/python -c "from app.config import Settings; from app.services.storage import Storage; s=Storage(Settings(_env_file=None, s3_endpoint_url='http://localhost:4566', s3_bucket_originals='test-photos-originals')); s.put('originals','persist.txt',b'x','text/plain')"`, then `docker compose restart floci`, then `../venv/Scripts/python -c "from app.config import Settings; from app.services.storage import Storage; s=Storage(Settings(_env_file=None, s3_endpoint_url='http://localhost:4566', s3_bucket_originals='test-photos-originals')); print(s.get('originals','persist.txt'))"`
Expected: `b'x'`. If the object is gone, set `FLOCI_STORAGE_MODE: wal` in compose, run `docker compose up -d floci`, and repeat.

- [ ] **Step 7: Commit**

```bash
git add backend/
git commit -m "feat(backend): S3 storage service with presigned URLs"
```

---

### Task 6: CLIP encoder and test fake

**Files:**
- Create: `backend/app/services/clip_model.py`, `backend/tests/fakes.py`
- Modify: `backend/tests/conftest.py` (add the `encoder` fixture)
- Test: `backend/tests/test_clip_model.py`

**Interfaces:**
- Produces:
  - `Encoder` Protocol: `model_name: str`, `encode_images(images: list[PIL.Image.Image]) -> list[list[float]]`, `encode_text(texts: list[str]) -> list[list[float]]`. All vectors are L2-normalised.
  - `ClipEncoder(model_name: str, device: str | None = None)`
  - `tests.fakes.FakeEncoder(text_vectors: dict[str, list[float]] | None = None)` with `model_name = "fake"` and 8-dimensional vectors. An image vector is its normalised `[r, g, b, 0, 0, 0, 0, 0.01]` mean colour. A text vector is the mapped vector, or a sha256-derived one.
  - `tests.fakes.BrokenIndex`: every method raises `ConnectionError`.

- [ ] **Step 1: Write `backend/tests/fakes.py`**

```python
import hashlib
import math

from PIL import Image

DIM = 8


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FakeEncoder:
    """Deterministic stand-in for CLIP: colour-based image vectors, hash-based text vectors."""

    model_name = "fake"

    def __init__(self, text_vectors: dict[str, list[float]] | None = None):
        self.text_vectors = text_vectors or {}

    def encode_images(self, images: list[Image.Image]) -> list[list[float]]:
        out = []
        for im in images:
            r, g, b = im.convert("RGB").resize((1, 1)).getpixel((0, 0))
            out.append(_normalize([r / 255, g / 255, b / 255, 0, 0, 0, 0, 0.01]))
        return out

    def encode_text(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            if t in self.text_vectors:
                out.append(_normalize(self.text_vectors[t]))
            else:
                digest = hashlib.sha256(t.encode()).digest()
                out.append(_normalize([b - 128 for b in digest[:DIM]]))
        return out


class BrokenIndex:
    """VectorIndex stand-in whose every call fails, simulating ChromaDB being down."""

    def __getattr__(self, name):
        def fail(*args, **kwargs):
            raise ConnectionError("chroma is down")

        return fail
```

Append to `conftest.py`:

```python
from tests.fakes import FakeEncoder


@pytest.fixture
def encoder() -> FakeEncoder:
    return FakeEncoder()
```

- [ ] **Step 2: Write the failing tests**

`backend/tests/test_clip_model.py`:

```python
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
```

- [ ] **Step 3: Run to verify the slow test fails**

Run: `../venv/Scripts/python -m pytest tests/test_clip_model.py -v -m slow`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.clip_model'`.

- [ ] **Step 4: Implement `backend/app/services/clip_model.py`**

```python
from typing import Protocol

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image


class Encoder(Protocol):
    model_name: str

    def encode_images(self, images: list[Image.Image]) -> list[list[float]]: ...

    def encode_text(self, texts: list[str]) -> list[list[float]]: ...


class ClipEncoder:
    """OpenAI CLIP weights via open_clip. Load once per process."""

    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        arch = model_name.replace("/", "-")  # "ViT-B/32" -> "ViT-B-32"
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch, pretrained="openai", device=self.device
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(arch)  # truncates to 77 tokens

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> list[list[float]]:
        batch = torch.stack([self._preprocess(im.convert("RGB")) for im in images]).to(self.device)
        feats = self._model.encode_image(batch).float()
        return F.normalize(feats, dim=-1).cpu().tolist()

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> list[list[float]]:
        tokens = self._tokenizer(texts).to(self.device)
        feats = self._model.encode_text(tokens).float()
        return F.normalize(feats, dim=-1).cpu().tolist()
```

- [ ] **Step 5: Run to verify all pass, including the slow test**

Run: `../venv/Scripts/python -m pytest tests/test_clip_model.py -v -m "slow or not slow"`
Expected: 3 passed. The first run downloads the CLIP weights.

- [ ] **Step 6: Commit**

```bash
git add backend/
git commit -m "feat(backend): CLIP encoder service and deterministic fake for tests"
```

---

### Task 7: Vector index (ChromaDB)

**Files:**
- Create: `backend/app/services/vector_index.py`
- Modify: `backend/tests/conftest.py` (add the `index` fixture)
- Test: `backend/tests/test_vector_index.py`

**Interfaces:**
- Consumes: `Settings`, `Encoder.model_name`.
- Produces:
  - `Hit` (frozen dataclass): `image_id: UUID`, `kind: Literal["img", "txt"]`, `similarity: float`
  - `ModelMismatchError(RuntimeError)`
  - `build_where(tags: list[str], source: str | None) -> dict | None`
  - `VectorIndex(host: str, port: int, collection_name: str, model_name: str)`, which connects lazily
    - `ensure_collection() -> None` (raises `ModelMismatchError`), `recreate() -> None`, `ping() -> None`
    - `upsert(image_id, *, image_vector: list[float], text_vector: list[float] | None, tags: list[str], source: str) -> None`, which replaces both of the image's entries
    - `get_image_vector(image_id) -> list[float] | None`
    - `delete(image_id) -> None`
    - `query(vector, n: int, tags: list[str] = (), source: str | None = None) -> list[Hit]`
    - `image_similarities(image_ids: set[UUID], vector) -> dict[UUID, float]`
    - `ids() -> list[str]` (used by tests and diagnostics)

- [ ] **Step 1: Add the fixture to `conftest.py`**

Append:

```python
from app.services.vector_index import VectorIndex


@pytest.fixture
def index(settings, encoder) -> VectorIndex:
    idx = VectorIndex(settings.chroma_host, settings.chroma_port, settings.chroma_collection, encoder.model_name)
    idx.recreate()
    return idx
```

- [ ] **Step 2: Write the failing tests**

`backend/tests/test_vector_index.py`:

```python
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


def test_upsert_and_query_returns_both_kinds(index):
    a = uuid.uuid4()
    index.upsert(a, image_vector=E[0], text_vector=E[1], tags=["goa"], source="upload")
    assert sorted(index.ids()) == sorted([f"{a}:img", f"{a}:txt"])
    hits = index.query(E[0], n=10)
    assert hits[0].image_id == a and hits[0].kind == "img"
    assert hits[0].similarity == pytest.approx(1.0)
    assert {h.kind for h in hits} == {"img", "txt"}
    txt = next(h for h in hits if h.kind == "txt")
    assert txt.similarity == pytest.approx(0.5)  # orthogonal -> distance 1 -> 1 - 1/2


def test_reupsert_without_text_removes_txt(index):
    a = uuid.uuid4()
    index.upsert(a, image_vector=E[0], text_vector=E[1], tags=["goa"], source="upload")
    index.upsert(a, image_vector=E[0], text_vector=None, tags=[], source="upload")
    assert index.ids() == [f"{a}:img"]


def test_tag_and_source_filters(index):
    a, b = uuid.uuid4(), uuid.uuid4()
    index.upsert(a, image_vector=E[0], text_vector=None, tags=["goa", "beach"], source="upload")
    index.upsert(b, image_vector=E[0], text_vector=None, tags=["goa"], source="demo")
    assert {h.image_id for h in index.query(E[0], n=10, tags=["goa"])} == {a, b}
    assert {h.image_id for h in index.query(E[0], n=10, tags=["goa", "beach"])} == {a}
    assert {h.image_id for h in index.query(E[0], n=10, source="demo")} == {b}
    assert index.query(E[0], n=10, tags=["nobody-uses-this"]) == []


def test_query_on_empty_collection(index):
    assert index.query(E[0], n=5) == []


def test_get_image_vector_and_similarities(index):
    a, b = uuid.uuid4(), uuid.uuid4()
    index.upsert(a, image_vector=E[0], text_vector=None, tags=[], source="upload")
    index.upsert(b, image_vector=E[2], text_vector=None, tags=[], source="upload")
    assert index.get_image_vector(a) == pytest.approx(E[0])
    assert index.get_image_vector(uuid.uuid4()) is None
    sims = index.image_similarities({a, b}, E[0])
    assert sims[a] == pytest.approx(1.0)
    assert sims[b] == pytest.approx(0.5)


def test_delete(index):
    a = uuid.uuid4()
    index.upsert(a, image_vector=E[0], text_vector=E[1], tags=[], source="upload")
    index.delete(a)
    index.delete(a)  # idempotent
    assert index.ids() == []


def test_model_mismatch(settings, index):
    other = VectorIndex(settings.chroma_host, settings.chroma_port, settings.chroma_collection, "ViT-L/14")
    with pytest.raises(ModelMismatchError):
        other.ensure_collection()
```

- [ ] **Step 3: Run to verify it fails**

Run: `../venv/Scripts/python -m pytest tests/test_vector_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.vector_index'`.

- [ ] **Step 4: Implement `backend/app/services/vector_index.py`**

```python
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

import chromadb


@dataclass(frozen=True)
class Hit:
    image_id: UUID
    kind: Literal["img", "txt"]
    similarity: float


class ModelMismatchError(RuntimeError):
    pass


def build_where(tags: list[str], source: str | None) -> dict | None:
    conditions: list[dict] = [{f"tag_{t}": True} for t in tags]
    if source:
        conditions.append({"source": source})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _ids(image_id: UUID) -> list[str]:
    return [f"{image_id}:img", f"{image_id}:txt"]


class VectorIndex:
    """Derived CLIP index: per image an ':img' vector and optional ':txt' metadata vector."""

    def __init__(self, host: str, port: int, collection_name: str, model_name: str):
        self._host, self._port = host, port
        self._name = collection_name
        self._model = model_name
        self._client = None
        self._col = None

    @property
    def client(self):
        if self._client is None:
            self._client = chromadb.HttpClient(host=self._host, port=self._port)
        return self._client

    @property
    def collection(self):
        if self._col is None:
            self.ensure_collection()
        return self._col

    def _metadata(self) -> dict:
        return {"hnsw:space": "cosine", "clip_model": self._model}

    def ensure_collection(self) -> None:
        col = self.client.get_or_create_collection(self._name, metadata=self._metadata())
        recorded = (col.metadata or {}).get("clip_model")
        if recorded != self._model:
            raise ModelMismatchError(
                f"Collection '{self._name}' was built with CLIP model '{recorded}' but "
                f"CLIP_MODEL is '{self._model}'. Run: python -m app.cli reindex --all"
            )
        self._col = col

    def recreate(self) -> None:
        try:
            self.client.delete_collection(self._name)
        except Exception:
            pass  # did not exist
        self._col = self.client.create_collection(self._name, metadata=self._metadata())

    def ping(self) -> None:
        self.client.heartbeat()

    def ids(self) -> list[str]:
        return list(self.collection.get(include=[])["ids"])

    def upsert(
        self,
        image_id: UUID,
        *,
        image_vector: list[float],
        text_vector: list[float] | None,
        tags: list[str],
        source: str,
    ) -> None:
        # Delete + add (rather than update) so removed tags / text never linger.
        self.delete(image_id)
        base = {"image_id": str(image_id), "source": source, **{f"tag_{t}": True for t in tags}}
        ids, embeddings, metadatas = [f"{image_id}:img"], [image_vector], [{**base, "kind": "img"}]
        if text_vector is not None:
            ids.append(f"{image_id}:txt")
            embeddings.append(text_vector)
            metadatas.append({**base, "kind": "txt"})
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def get_image_vector(self, image_id: UUID) -> list[float] | None:
        res = self.collection.get(ids=[f"{image_id}:img"], include=["embeddings"])
        if not res["ids"]:
            return None
        return [float(x) for x in res["embeddings"][0]]

    def delete(self, image_id: UUID) -> None:
        self.collection.delete(ids=_ids(image_id))

    def query(
        self, vector: list[float], n: int, tags: list[str] = (), source: str | None = None
    ) -> list[Hit]:
        total = self.collection.count()
        if total == 0:
            return []
        res = self.collection.query(
            query_embeddings=[vector],
            n_results=min(n, total),
            where=build_where(list(tags), source),
            include=["metadatas", "distances"],
        )
        return [
            Hit(UUID(meta["image_id"]), meta["kind"], 1.0 - dist / 2.0)
            for meta, dist in zip(res["metadatas"][0], res["distances"][0])
        ]

    def image_similarities(self, image_ids: set[UUID], vector: list[float]) -> dict[UUID, float]:
        if not image_ids:
            return {}
        res = self.collection.get(
            ids=[f"{i}:img" for i in image_ids], include=["embeddings", "metadatas"]
        )
        sims = {}
        for meta, emb in zip(res["metadatas"], res["embeddings"]):
            cos = sum(float(a) * b for a, b in zip(emb, vector))
            sims[UUID(meta["image_id"])] = (1.0 + cos) / 2.0  # == 1 - (1 - cos) / 2
        return sims
```

- [ ] **Step 5: Run to verify it passes**

Run: `../venv/Scripts/python -m pytest tests/test_vector_index.py -v`
Expected: all passed.

- [ ] **Step 6: Commit**

```bash
git add backend/
git commit -m "feat(backend): ChromaDB vector index with image+text vectors and tag filters"
```

---

### Task 8: Image service (upload, edit, delete, list)

**Files:**
- Create: `backend/app/errors.py` (exception classes only for now), `backend/app/services/images.py`
- Modify: `backend/tests/conftest.py` (add the `image_service` fixture)
- Test: `backend/tests/test_image_service.py`

**Interfaces:**
- Consumes: `make_session_factory` sessions, `Storage`, `VectorIndex`, `Encoder`, `process_image`, `open_rgb`, `get_or_create_tags`, `tag_counts`.
- Produces:
  - `app.errors.AppError(status: int, code: str, message: str)`, `app.errors.NotFoundError(what="Image")` (status 404, code `not_found`)
  - `UploadResult` (frozen dataclass): `filename`, `status: Literal["created", "duplicate", "error"]`, `id: UUID | None`, `message: str | None`
  - `build_text_document(title, description, tags) -> str | None`
  - `INDEX_WARNING: str`
  - `ImageService(sessions, storage, index, encoder, settings)` with:
    - `upload(filename, data, *, tags: list[str], title=None, description=None, source="upload") -> UploadResult`
    - `index_image(image_id, rgb=None) -> str | None` (returns a warning on failure)
    - `get(image_id) -> Image`, which raises `NotFoundError`
    - `get_many(ids) -> dict[UUID, Image]`
    - `list_images(*, page, page_size, tags, source, sort) -> tuple[list[Image], int]`
    - `update(image_id, changes: dict) -> tuple[Image, str | None]`; `changes` keys are a subset of `title`, `description`, `tags`, and tags are already normalised
    - `delete(image_id) -> None`, which raises `NotFoundError`
    - `unindexed_ids(include_all=False) -> list[UUID]`
    - `tag_counts() -> list[tuple[str, int]]`

- [ ] **Step 1: Write `backend/app/errors.py` (exception classes; handlers come in Task 10)**

```python
class AppError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class NotFoundError(AppError):
    def __init__(self, what: str = "Image"):
        super().__init__(404, "not_found", f"{what} not found")
```

- [ ] **Step 2: Add the fixture to `conftest.py`**

Append:

```python
from app.services.images import ImageService


@pytest.fixture
def image_service(sessions, storage, index, encoder, settings) -> ImageService:
    return ImageService(sessions, storage, index, encoder, settings)
```

- [ ] **Step 3: Write the failing tests**

`backend/tests/test_image_service.py`:

```python
import os
import uuid

import pytest
from botocore.exceptions import ClientError

from app.errors import NotFoundError
from app.models import Image
from app.services.images import INDEX_WARNING, ImageService, build_text_document
from app.services.storage import Storage
from tests.fakes import BrokenIndex
from tests.helpers import jpeg_with_exif, png_bytes


def objects(storage, kind):
    return [o["Key"] for o in storage.client.list_objects_v2(Bucket=storage.bucket(kind)).get("Contents", [])]


def test_build_text_document():
    assert build_text_document(None, None, []) is None
    assert build_text_document(" ", "", []) is None
    assert build_text_document("Sunset", None, ["goa"]) == "Sunset. tags: goa"
    assert build_text_document("A", "B", ["x", "y"]) == "A. B. tags: x, y"


def test_upload_creates_row_objects_and_vectors(image_service, storage, index, sessions):
    r = image_service.upload("red.png", png_bytes(), tags=["goa"], title="Beach", description=None)
    assert r.status == "created" and r.message is None
    with sessions() as s:
        img = s.get(Image, r.id)
        assert img.indexed is True
        assert [t.name for t in img.tags] == ["goa"]
        assert (img.filename, img.mime_type, img.source) == ("red.png", "image/png", "upload")
    assert objects(storage, "originals") == [f"originals/{r.id}.png"]
    assert objects(storage, "thumbs") == [f"thumbs/{r.id}.webp"]
    assert sorted(index.ids()) == sorted([f"{r.id}:img", f"{r.id}:txt"])


def test_upload_without_metadata_has_only_image_vector(image_service, index):
    r = image_service.upload("red.png", png_bytes(), tags=[])
    assert index.ids() == [f"{r.id}:img"]


def test_upload_reads_exif_date(image_service):
    r = image_service.upload("p.jpg", jpeg_with_exif(orientation=6, taken="2023:01:02 03:04:05"), tags=[])
    img = image_service.get(r.id)
    assert img.taken_at.year == 2023 and (img.width, img.height) == (100, 200)


def test_duplicate_upload(image_service):
    first = image_service.upload("a.png", png_bytes(), tags=[])
    second = image_service.upload("b.png", png_bytes(), tags=["x"])
    assert second.status == "duplicate" and second.id == first.id


def test_upload_rejects_non_image_and_leaves_nothing(image_service, storage, sessions):
    r = image_service.upload("notes.txt", b"hello", tags=[])
    assert r.status == "error" and "image" in r.message.lower()
    assert objects(storage, "originals") == [] and objects(storage, "thumbs") == []
    with sessions() as s:
        assert s.query(Image).count() == 0


def test_upload_oversize(image_service):
    r = image_service.upload("big.png", os.urandom(1024 * 1024 + 1), tags=[])  # test limit is 1 MB
    assert r.status == "error" and "1 MB" in r.message


def test_storage_failure_cleans_up_original(sessions, index, encoder, settings, storage):
    class ThumbFails(Storage):
        def put(self, kind, key, data, content_type):
            if kind == "thumbs":
                raise ClientError({"Error": {"Code": "500", "Message": "boom"}}, "PutObject")
            super().put(kind, key, data, content_type)

    svc = ImageService(sessions, ThumbFails(settings), index, encoder, settings)
    r = svc.upload("red.png", png_bytes(), tags=[])
    assert r.status == "error"
    assert objects(storage, "originals") == []
    with sessions() as s:
        assert s.query(Image).count() == 0


def test_upload_when_index_down_keeps_row_unindexed(sessions, storage, encoder, settings):
    svc = ImageService(sessions, storage, BrokenIndex(), encoder, settings)
    r = svc.upload("red.png", png_bytes(), tags=["goa"])
    assert r.status == "created" and r.message == INDEX_WARNING
    assert svc.get(r.id).indexed is False
    assert svc.unindexed_ids() == [r.id]


def test_update_tags_and_title_reembeds_text(image_service, index):
    r = image_service.upload("red.png", png_bytes(), tags=[])
    assert index.ids() == [f"{r.id}:img"]
    img, warning = image_service.update(r.id, {"title": "  Sunset ", "tags": ["goa", "beach"]})
    assert warning is None
    assert img.title == "Sunset"
    assert [t.name for t in img.tags] == ["beach", "goa"]
    assert sorted(index.ids()) == sorted([f"{r.id}:img", f"{r.id}:txt"])
    assert {h.image_id for h in index.query([1.0] + [0.0] * 7, n=10, tags=["beach"])} == {r.id}


def test_update_only_provided_fields(image_service):
    r = image_service.upload("red.png", png_bytes(), tags=["goa"], title="T", description="D")
    img, _ = image_service.update(r.id, {"description": None})
    assert (img.title, img.description, [t.name for t in img.tags]) == ("T", None, ["goa"])


def test_update_clearing_metadata_removes_text_vector(image_service, index):
    r = image_service.upload("red.png", png_bytes(), tags=["goa"], title="T")
    image_service.update(r.id, {"title": "", "tags": []})
    assert index.ids() == [f"{r.id}:img"]
    assert index.query([1.0] + [0.0] * 7, n=10, tags=["goa"]) == []


def test_update_unknown_image(image_service):
    with pytest.raises(NotFoundError):
        image_service.update(uuid.uuid4(), {"title": "x"})


def test_delete_removes_everything(image_service, storage, index):
    r = image_service.upload("red.png", png_bytes(), tags=["goa"])
    image_service.delete(r.id)
    assert index.ids() == []
    assert objects(storage, "originals") == [] and objects(storage, "thumbs") == []
    with pytest.raises(NotFoundError):
        image_service.get(r.id)
    with pytest.raises(NotFoundError):
        image_service.delete(r.id)


def test_list_images_pagination_filter_and_sort(image_service):
    ids = [
        image_service.upload(f"{i}.png", png_bytes(color=(i * 20, 0, 0)), tags=["even"] if i % 2 == 0 else []).id
        for i in range(5)
    ]
    page1, total = image_service.list_images(page=1, page_size=2, tags=[], source=None, sort="uploaded")
    assert total == 5 and [i.id for i in page1] == [ids[4], ids[3]]
    page3, _ = image_service.list_images(page=3, page_size=2, tags=[], source=None, sort="uploaded")
    assert [i.id for i in page3] == [ids[0]]
    even, total_even = image_service.list_images(page=1, page_size=10, tags=["even"], source=None, sort="taken")
    assert total_even == 3 and {i.id for i in even} == {ids[0], ids[2], ids[4]}
    none, total_none = image_service.list_images(page=1, page_size=10, tags=["nope"], source=None, sort="taken")
    assert (none, total_none) == ([], 0)
    demo, _ = image_service.list_images(page=1, page_size=10, tags=[], source="demo", sort="taken")
    assert demo == []


def test_tag_counts(image_service):
    image_service.upload("a.png", png_bytes(color=(1, 1, 1)), tags=["goa", "beach"])
    image_service.upload("b.png", png_bytes(color=(2, 2, 2)), tags=["goa"])
    assert image_service.tag_counts() == [("goa", 2), ("beach", 1)]
```

- [ ] **Step 4: Run to verify it fails**

Run: `../venv/Scripts/python -m pytest tests/test_image_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.images'`.

- [ ] **Step 5: Implement `backend/app/services/images.py`**

```python
import logging
from dataclasses import dataclass
from typing import Literal
from uuid import UUID, uuid4

from PIL import Image as PILImage
from sqlalchemy import func, select
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


def build_text_document(title: str | None, description: str | None, tags: list[str]) -> str | None:
    parts = [p for p in (_clean(title), _clean(description)) if p]
    if tags:
        parts.append("tags: " + ", ".join(tags))
    return ". ".join(parts) or None


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
        """(Re)write this image's vectors. Returns a warning string if indexing failed."""
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
            doc = build_text_document(image.title, image.description, tags)
            text_vector = self.encoder.encode_text([doc])[0] if doc else None
            self.index.upsert(
                image_id, image_vector=image_vector, text_vector=text_vector, tags=tags, source=image.source
            )
            indexed, warning = True, None
        except Exception:
            logger.exception("Indexing failed for %s", image_id)
            indexed, warning = False, INDEX_WARNING
        with self.sessions.begin() as s:
            s.execute(sa_update(Image).where(Image.id == image_id).values(indexed=indexed))
        return warning

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

    def list_images(
        self, *, page: int, page_size: int, tags: list[str], source: str | None, sort: str
    ) -> tuple[list[Image], int]:
        stmt = select(Image)
        for tag in tags:
            stmt = stmt.where(Image.tags.any(Tag.name == tag))
        if source:
            stmt = stmt.where(Image.source == source)
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
```

- [ ] **Step 6: Run to verify it passes**

Run: `../venv/Scripts/python -m pytest tests/test_image_service.py -v`
Expected: all passed.

- [ ] **Step 7: Run the full suite**

Run: `../venv/Scripts/python -m pytest -q`
Expected: all passed.

- [ ] **Step 8: Commit**

```bash
git add backend/
git commit -m "feat(backend): image service for upload, edit, delete, listing and indexing"
```

---

### Task 9: Blended search service

**Files:**
- Create: `backend/app/services/search.py`
- Modify: `backend/tests/conftest.py` (add the `search_service` fixture)
- Test: `backend/tests/test_search.py`

**Interfaces:**
- Consumes: `Hit`, `VectorIndex.query`, `VectorIndex.image_similarities`, `ImageService.get_many`, `Encoder.encode_text`.
- Produces: `blend(hits: list[Hit], img_sims: dict[UUID, float], img_weight: float) -> list[tuple[UUID, float]]`, sorted by score descending. `SearchService(images: ImageService, index, encoder, settings)` with `search(q: str, *, tags: list[str], source: str | None, limit: int) -> list[tuple[Image, float]]`.

- [ ] **Step 1: Add the fixture to `conftest.py`**

Append:

```python
from app.services.search import SearchService


@pytest.fixture
def search_service(image_service, index, encoder, settings) -> SearchService:
    return SearchService(image_service, index, encoder, settings)
```

- [ ] **Step 2: Write the failing tests**

`backend/tests/test_search.py`:

```python
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
```

- [ ] **Step 3: Run to verify it fails**

Run: `../venv/Scripts/python -m pytest tests/test_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.search'`.

- [ ] **Step 4: Implement `backend/app/services/search.py`**

```python
from uuid import UUID

from app.config import Settings
from app.models import Image
from app.services.clip_model import Encoder
from app.services.images import ImageService
from app.services.vector_index import Hit, VectorIndex


def blend(hits: list[Hit], img_sims: dict[UUID, float], img_weight: float) -> list[tuple[UUID, float]]:
    """Combine per-image 'img' and 'txt' similarities into one score, best first."""
    img: dict[UUID, float] = {}
    txt: dict[UUID, float] = {}
    for h in hits:
        bucket = img if h.kind == "img" else txt
        bucket[h.image_id] = max(bucket.get(h.image_id, 0.0), h.similarity)

    scored = []
    for image_id in img.keys() | txt.keys():
        i = img.get(image_id, img_sims.get(image_id))
        t = txt.get(image_id)
        if t is None:
            score = i
        elif i is None:
            score = (1 - img_weight) * t
        else:
            score = img_weight * i + (1 - img_weight) * t
        scored.append((image_id, score))
    scored.sort(key=lambda pair: (-pair[1], str(pair[0])))
    return scored


class SearchService:
    def __init__(self, images: ImageService, index: VectorIndex, encoder: Encoder, settings: Settings):
        self.images = images
        self.index = index
        self.encoder = encoder
        self.settings = settings

    def search(
        self, q: str, *, tags: list[str], source: str | None, limit: int
    ) -> list[tuple[Image, float]]:
        vector = self.encoder.encode_text([q])[0]
        hits = self.index.query(vector, n=limit * 3, tags=tags, source=source)
        with_img = {h.image_id for h in hits if h.kind == "img"}
        txt_only = {h.image_id for h in hits if h.kind == "txt"} - with_img
        img_sims = self.index.image_similarities(txt_only, vector)
        ranked = blend(hits, img_sims, self.settings.search_img_weight)
        rows = self.images.get_many([image_id for image_id, _ in ranked])
        return [(rows[i], score) for i, score in ranked if i in rows][:limit]
```

- [ ] **Step 5: Run to verify it passes**

Run: `../venv/Scripts/python -m pytest tests/test_search.py -v`
Expected: all passed.

- [ ] **Step 6: Commit**

```bash
git add backend/
git commit -m "feat(backend): blended image+metadata search service"
```

---

### Task 10: FastAPI app, routers and error envelope

**Files:**
- Create: `backend/app/services/container.py`, `backend/app/schemas.py`, `backend/app/main.py`, `backend/app/routers/__init__.py` (empty), `backend/app/routers/deps.py`, `backend/app/routers/images.py`, `backend/app/routers/search.py`, `backend/app/routers/tags.py`, `backend/app/routers/health.py`
- Modify: `backend/app/errors.py` (add handlers), `backend/tests/conftest.py` (add `services` and `client`)
- Test: `backend/tests/test_api.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `Services` dataclass (`settings`, `sessions`, `storage`, `index`, `encoder`, `images`, `search`) with `ping_db()` and `startup(*, recreate_index: bool = False)`
  - `build_services(settings, encoder=None) -> Services`
  - `create_app(services_factory: Callable[[], Services] | None = None) -> FastAPI`, plus module-level `app`
  - HTTP API exactly as in spec §5; `PATCH` returns `{image: ImageDetail, warning}`, and `DELETE` returns 204
  - JSON shapes used by the frontend (Task 12): `ImageOut`, `ImageDetail`, `ImagePage`, `SearchItem`, `SearchResponse`, `TagCount`, `UploadResultOut`, `UpdateResponse`

- [ ] **Step 1: Write `backend/app/services/container.py`**

```python
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.db import make_session_factory
from app.services.clip_model import ClipEncoder, Encoder
from app.services.images import ImageService
from app.services.search import SearchService
from app.services.storage import Storage
from app.services.vector_index import ModelMismatchError, VectorIndex


@dataclass
class Services:
    settings: Settings
    sessions: sessionmaker[Session]
    storage: Storage
    index: VectorIndex
    encoder: Encoder
    images: ImageService
    search: SearchService

    def ping_db(self) -> None:
        with self.sessions() as s:
            s.execute(text("select 1"))

    def startup(self, *, recreate_index: bool = False) -> None:
        try:
            self.ping_db()
            self.storage.ensure_buckets()
            if recreate_index:
                self.index.recreate()
            else:
                self.index.ensure_collection()
        except ModelMismatchError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Startup check failed: {e}. Is `docker compose up -d` running and .env configured?"
            ) from e


def build_services(settings: Settings, encoder: Encoder | None = None) -> Services:
    encoder = encoder or ClipEncoder(settings.clip_model)
    sessions = make_session_factory(settings.database_url)
    storage = Storage(settings)
    index = VectorIndex(settings.chroma_host, settings.chroma_port, settings.chroma_collection, encoder.model_name)
    images = ImageService(sessions, storage, index, encoder, settings)
    search = SearchService(images, index, encoder, settings)
    return Services(settings, sessions, storage, index, encoder, images, search)
```

- [ ] **Step 2: Add the error handlers to `backend/app/errors.py`**

Append to the existing file:

```python
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.tags import InvalidTag


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError):
        return error_response(exc.status, exc.code, exc.message)

    @app.exception_handler(InvalidTag)
    async def _invalid_tag(_: Request, exc: InvalidTag):
        return error_response(400, "invalid_tag", str(exc))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        message = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        return error_response(400, "validation_error", message)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        return error_response(exc.status_code, "http_error", str(exc.detail))
```

- [ ] **Step 3: Write `backend/app/schemas.py`**

```python
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
```

- [ ] **Step 4: Write the routers**

`backend/app/routers/deps.py`:

```python
from typing import Annotated

from fastapi import Depends, Request

from app.services.container import Services


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]
```

`backend/app/routers/images.py`:

```python
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
```

`backend/app/routers/search.py`:

```python
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
```

`backend/app/routers/tags.py`:

```python
from fastapi import APIRouter

from app.routers.deps import ServicesDep
from app.schemas import TagCount

router = APIRouter(tags=["tags"])


@router.get("/tags", response_model=list[TagCount])
def list_tags(services: ServicesDep):
    return [TagCount(name=n, count=c) for n, c in services.images.tag_counts()]
```

`backend/app/routers/health.py`:

```python
from collections.abc import Callable

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.routers.deps import ServicesDep

router = APIRouter(tags=["health"])


def _ok(check: Callable[[], None]) -> bool:
    try:
        check()
        return True
    except Exception:
        return False


@router.get("/health")
def health(services: ServicesDep):
    checks = {
        "postgres": _ok(services.ping_db),
        "s3": _ok(services.storage.ping),
        "chroma": _ok(services.index.ping),
        "model": services.encoder is not None,
    }
    healthy = all(checks.values())
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={"status": "ok" if healthy else "degraded", "checks": checks},
    )
```

- [ ] **Step 5: Write `backend/app/main.py`**

```python
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.errors import register_error_handlers
from app.routers import health, images, search, tags
from app.services.container import Services, build_services


def create_app(services_factory: Callable[[], Services] | None = None) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=logging.INFO)
        services = services_factory() if services_factory else build_services(get_settings())
        services.startup()
        app.state.services = services
        yield

    app = FastAPI(title="Photo Retrieval", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_error_handlers(app)
    for module in (images, search, tags, health):
        app.include_router(module.router, prefix="/api")
    return app


app = create_app()
```

- [ ] **Step 6: Add the API fixtures to `conftest.py`**

Append:

```python
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.container import Services


@pytest.fixture
def services(settings, sessions, storage, index, encoder, image_service, search_service) -> Services:
    return Services(settings, sessions, storage, index, encoder, image_service, search_service)


@pytest.fixture
def client(services):
    with TestClient(create_app(lambda: services)) as c:
        yield c
```

- [ ] **Step 7: Write the failing API tests**

`backend/tests/test_api.py`:

```python
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
```

- [ ] **Step 8: Run to verify the tests pass**

Run: `../venv/Scripts/python -m pytest tests/test_api.py -v`
Expected: all passed. If a test fails, fix the code until it passes, then run the full suite: `../venv/Scripts/python -m pytest -q`.

- [ ] **Step 9: Smoke-run the real server with CLIP**

Run: `../venv/Scripts/python -m uvicorn app.main:app --port 8080` (in a background terminal). Then run `curl -s localhost:8080/api/health`.
Expected: `{"status":"ok","checks":{"postgres":true,"s3":true,"chroma":true,"model":true}}`. Stop the server afterwards.

- [ ] **Step 10: Commit**

```bash
git add backend/
git commit -m "feat(backend): FastAPI app with images, search, tags and health endpoints"
```

---

### Task 11: Maintenance CLI (reindex, seed-demo)

**Files:**
- Create: `backend/app/cli.py`
- Test: `backend/tests/test_cli.py`

**Interfaces:**
- Consumes: `Services`, `build_services`, `ImageService.unindexed_ids`, `ImageService.index_image`, `ImageService.upload`, `normalize_tag`.
- Produces: `cmd_reindex(services, *, all_images: bool) -> int` (the exit code), `cmd_seed_demo(services, samples: Iterable[tuple[PIL.Image.Image, str]]) -> Counter`, `cifar_samples(data_dir: str, limit: int | None)`, and `main(argv=None) -> int`. Invoked as `python -m app.cli reindex [--all]` and `python -m app.cli seed-demo [--limit N] [--data-dir DIR]`.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_cli.py`:

```python
from PIL import Image

from app.cli import cmd_reindex, cmd_seed_demo
from app.services.images import ImageService
from tests.fakes import BrokenIndex
from tests.helpers import png_bytes


def test_reindex_catches_up_unindexed(services, sessions, storage, encoder, settings, index):
    broken = ImageService(sessions, storage, BrokenIndex(), encoder, settings)
    r = broken.upload("red.png", png_bytes(), tags=["goa"])
    assert services.images.get(r.id).indexed is False

    assert cmd_reindex(services, all_images=False) == 0
    assert services.images.get(r.id).indexed is True
    assert sorted(index.ids()) == sorted([f"{r.id}:img", f"{r.id}:txt"])


def test_reindex_all_rebuilds_from_s3(services, index):
    r = services.images.upload("red.png", png_bytes(), tags=[])
    index.recreate()  # simulate a wiped / new collection
    assert cmd_reindex(services, all_images=True) == 0
    assert index.ids() == [f"{r.id}:img"]


def test_seed_demo_is_idempotent(services):
    samples = [
        (Image.new("RGB", (32, 32), (200, 0, 0)), "automobile"),
        (Image.new("RGB", (32, 32), (0, 200, 0)), "frog"),
    ]
    first = cmd_seed_demo(services, samples)
    assert first["created"] == 2
    second = cmd_seed_demo(services, samples)
    assert second["duplicate"] == 2
    items, total = services.images.list_images(page=1, page_size=10, tags=["frog"], source="demo", sort="taken")
    assert total == 1 and items[0].filename == "cifar10-00002.png"
```

- [ ] **Step 2: Run to verify it fails**

Run: `../venv/Scripts/python -m pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.cli'`.

- [ ] **Step 3: Implement `backend/app/cli.py`**

```python
"""Maintenance commands.

  python -m app.cli reindex          # index images with indexed=false
  python -m app.cli reindex --all    # recreate the collection and re-embed everything from S3
  python -m app.cli seed-demo [--limit N] [--data-dir ../data]
"""
import argparse
import io
import logging
import sys
from collections import Counter
from collections.abc import Iterable, Iterator

from PIL import Image

from app.config import get_settings
from app.services.container import Services, build_services
from app.tags import normalize_tag


def cmd_reindex(services: Services, *, all_images: bool) -> int:
    ids = services.images.unindexed_ids(include_all=all_images)
    print(f"Reindexing {len(ids)} image(s) ...")
    failed = 0
    for n, image_id in enumerate(ids, 1):
        if services.images.index_image(image_id) is not None:
            failed += 1
        if n % 100 == 0:
            print(f"  {n}/{len(ids)}")
    print(f"Done: {len(ids) - failed} indexed, {failed} failed")
    return 1 if failed else 0


def cmd_seed_demo(services: Services, samples: Iterable[tuple[Image.Image, str]]) -> Counter:
    counts: Counter = Counter()
    for i, (img, label) in enumerate(samples, 1):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = services.images.upload(
            f"cifar10-{i:05d}.png", buf.getvalue(), tags=[normalize_tag(label)], source="demo"
        )
        counts[result.status] += 1
        if i % 500 == 0:
            print(f"  {i} processed {dict(counts)}")
    print(f"Seed complete: {dict(counts)}")
    return counts


def cifar_samples(data_dir: str, limit: int | None) -> Iterator[tuple[Image.Image, str]]:
    from torchvision.datasets import CIFAR10

    ds = CIFAR10(root=data_dir, train=False, download=True)
    n = len(ds) if limit is None else min(limit, len(ds))
    for i in range(n):
        img, label = ds[i]
        yield img, ds.classes[label]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    p_re = sub.add_parser("reindex", help="Rebuild missing (or all) vectors in ChromaDB")
    p_re.add_argument("--all", action="store_true", dest="all_images", help="Recreate collection, re-embed all")
    p_seed = sub.add_parser("seed-demo", help="Load the CIFAR-10 test set as demo images")
    p_seed.add_argument("--limit", type=int, default=None)
    p_seed.add_argument("--data-dir", default="../data")
    args = parser.parse_args(argv)

    services = build_services(get_settings())
    services.startup(recreate_index=args.command == "reindex" and args.all_images)
    if args.command == "reindex":
        return cmd_reindex(services, all_images=args.all_images)
    counts = cmd_seed_demo(services, cifar_samples(args.data_dir, args.limit))
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify it passes**

Run: `../venv/Scripts/python -m pytest tests/test_cli.py -v`
Expected: 3 passed.

- [ ] **Step 5: Seed the dev database for real**

Run: `../venv/Scripts/python -m app.cli seed-demo --limit 200`
Expected: `Seed complete: {'created': 200}`. Running it again gives `{'duplicate': 200}`. The full set (no `--limit`) takes a few minutes and can be run later.

- [ ] **Step 6: Commit**

```bash
git add backend/
git commit -m "feat(backend): reindex and seed-demo maintenance commands"
```

---

### Task 12: Frontend scaffold, API client and tag helpers

**Files:**
- Create: `frontend/package.json`, `frontend/index.html`, `frontend/tsconfig.json`, `frontend/vite.config.ts`, `frontend/src/vite-env.d.ts`, `frontend/src/main.tsx`, `frontend/src/index.css`, `frontend/src/App.tsx`, `frontend/src/api/types.ts`, `frontend/src/api/client.ts`, `frontend/src/lib/tags.ts`, `frontend/src/test/setup.ts`, `frontend/src/test/utils.tsx`
- Test: `frontend/src/api/client.test.ts`, `frontend/src/lib/tags.test.ts`

**Interfaces:**
- Consumes: the HTTP API from Task 10.
- Produces:
  - Types: `Source`, `ImageSummary`, `ImageDetail`, `ImagePage`, `SearchItem`, `SearchResponse`, `TagCount`, `UploadResult`, `ImagePatch`, `UpdateResponse`, `UploadMeta`, `ListParams`, `SearchParams`
  - `ApiError(status, code, message)`
  - `api.listImages(p: ListParams)`, `api.searchImages(p: SearchParams)`, `api.getImage(id)`, `api.updateImage(id, patch)`, `api.deleteImage(id)`, `api.listTags()`, `api.health()`, `api.uploadImage(file, meta, onProgress?)`
  - `normalizeTag(raw): string | null`
  - `renderWithClient(ui)`

- [ ] **Step 1: Create the project files**

`frontend/package.json`:

```json
{
  "name": "photo-retrieval-web",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "typecheck": "tsc",
    "test": "vitest run",
    "e2e": "playwright test"
  }
}
```

Run (from `frontend/`):

```bash
npm install react react-dom @tanstack/react-query sonner
```

```bash
npm install -D vite @vitejs/plugin-react typescript @types/react @types/react-dom tailwindcss @tailwindcss/vite vitest jsdom @testing-library/react @testing-library/user-event @testing-library/jest-dom @types/node @playwright/test
```

`frontend/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Photos</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

`frontend/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noEmit": true,
    "skipLibCheck": true,
    "types": ["node"]
  },
  "include": ["src", "e2e", "vite.config.ts", "playwright.config.ts"]
}
```

`frontend/vite.config.ts`:

```ts
/// <reference types="vitest/config" />
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: { "/api": process.env.VITE_API_PROXY ?? "http://localhost:8080" },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
```

`frontend/src/vite-env.d.ts`:

```ts
/// <reference types="vite/client" />
```

`frontend/src/index.css`:

```css
@import "tailwindcss";
```

`frontend/src/main.tsx`:

```tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 30_000, refetchOnWindowFocus: false } },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
);
```

`frontend/src/App.tsx` (temporary; replaced in Task 13):

```tsx
export default function App() {
  return <h1 className="p-4 text-lg font-semibold">Photos</h1>;
}
```

`frontend/src/test/setup.ts`:

```ts
import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => cleanup());
```

`frontend/src/test/utils.tsx`:

```tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import type { ReactElement } from "react";

export function renderWithClient(ui: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return { client, ...render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>) };
}
```

- [ ] **Step 2: Write `src/api/types.ts`**

```ts
export type Source = "upload" | "demo";

export interface ImageSummary {
  id: string;
  filename: string;
  title: string | null;
  description: string | null;
  tags: string[];
  width: number;
  height: number;
  taken_at: string | null;
  created_at: string;
  source: Source;
  indexed: boolean;
  thumb_url: string;
}

export interface ImageDetail extends ImageSummary {
  original_url: string;
  mime_type: string;
}

export interface ImagePage {
  items: ImageSummary[];
  page: number;
  page_size: number;
  total: number;
}

export interface SearchItem extends ImageSummary {
  score: number | null;
}

export interface SearchResponse {
  items: SearchItem[];
}

export interface TagCount {
  name: string;
  count: number;
}

export interface UploadResult {
  filename: string;
  status: "created" | "duplicate" | "error";
  id: string | null;
  message: string | null;
}

export interface ImagePatch {
  title?: string | null;
  description?: string | null;
  tags?: string[];
}

export interface UpdateResponse {
  image: ImageDetail;
  warning: string | null;
}

export interface UploadMeta {
  tags: string[];
  title: string;
  description: string;
}

export interface ListParams {
  page: number;
  page_size: number;
  tags?: string[];
  source?: Source;
  sort?: "taken" | "uploaded";
}

export interface SearchParams {
  q: string;
  tags?: string[];
  source?: Source;
  limit?: number;
}
```

- [ ] **Step 3: Write the failing tests**

`frontend/src/lib/tags.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { normalizeTag } from "./tags";

describe("normalizeTag", () => {
  it.each([
    ["Goa", "goa"],
    ["  Goa Trip ", "goa-trip"],
    ["new   year", "new-year"],
  ])("normalises %s", (raw, expected) => {
    expect(normalizeTag(raw)).toBe(expected);
  });

  it.each(["", "   ", "#fun!", "café", "x".repeat(41)])("rejects %s", (raw) => {
    expect(normalizeTag(raw)).toBeNull();
  });
});
```

`frontend/src/api/client.test.ts`:

```ts
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "./client";

function mockFetch(status: number, body: unknown) {
  const fn = vi.fn().mockResolvedValue(
    new Response(status === 204 ? null : JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    }),
  );
  vi.stubGlobal("fetch", fn);
  return fn;
}

afterEach(() => vi.unstubAllGlobals());

describe("api client", () => {
  it("builds list query with repeated tags and skips empty params", async () => {
    const fetchFn = mockFetch(200, { items: [], page: 1, page_size: 60, total: 0 });
    await api.listImages({ page: 1, page_size: 60, tags: ["goa", "beach"], source: undefined });
    expect(fetchFn).toHaveBeenCalledWith("/api/images?page=1&page_size=60&tags=goa&tags=beach", undefined);
  });

  it("sends PATCH as JSON", async () => {
    const fetchFn = mockFetch(200, { image: {}, warning: null });
    await api.updateImage("abc", { title: "x" });
    const [url, init] = fetchFn.mock.calls[0];
    expect(url).toBe("/api/images/abc");
    expect(init.method).toBe("PATCH");
    expect(JSON.parse(init.body)).toEqual({ title: "x" });
  });

  it("handles 204 on delete", async () => {
    mockFetch(204, null);
    await expect(api.deleteImage("abc")).resolves.toBeUndefined();
  });

  it("raises ApiError from the error envelope", async () => {
    mockFetch(400, { error: { code: "invalid_tag", message: "Invalid tag '#x'" } });
    const err = await api.updateImage("abc", { tags: ["#x"] }).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({ status: 400, code: "invalid_tag", message: "Invalid tag '#x'" });
  });
});
```

- [ ] **Step 4: Run to verify they fail**

Run: `npm test`
Expected: FAIL, because `./tags` and `./client` cannot be resolved.

- [ ] **Step 5: Implement `src/lib/tags.ts` and `src/api/client.ts`**

`frontend/src/lib/tags.ts`:

```ts
const VALID = /^[a-z0-9-]{1,40}$/;

/** Mirrors backend app/tags.py: trim, lowercase, whitespace -> '-'. Null if invalid. */
export function normalizeTag(raw: string): string | null {
  const name = raw.trim().toLowerCase().replace(/\s+/g, "-");
  return VALID.test(name) ? name : null;
}
```

`frontend/src/api/client.ts`:

```ts
import type {
  ImageDetail, ImagePage, ImagePatch, ListParams, SearchParams, SearchResponse, TagCount,
  UpdateResponse, UploadMeta, UploadResult,
} from "./types";

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

type QueryValue = string | number | undefined | string[];

function qs(params: Record<string, QueryValue>): string {
  const sp = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === "") continue;
    if (Array.isArray(value)) value.forEach((v) => sp.append(key, v));
    else sp.set(key, String(value));
  }
  const s = sp.toString();
  return s ? `?${s}` : "";
}

function toApiError(status: number, statusText: string, body: unknown): ApiError {
  const err = (body as { error?: { code?: string; message?: string } } | null)?.error;
  return new ApiError(status, err?.code ?? "http_error", err?.message ?? (statusText || "Request failed"));
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, init);
  if (!res.ok) {
    let body: unknown = null;
    try {
      body = await res.json();
    } catch {
      /* non-JSON error body */
    }
    throw toApiError(res.status, res.statusText, body);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const api = {
  listImages: (p: ListParams) =>
    request<ImagePage>(`/images${qs({ page: p.page, page_size: p.page_size, tags: p.tags, source: p.source, sort: p.sort })}`),

  searchImages: (p: SearchParams) =>
    request<SearchResponse>(`/search${qs({ q: p.q, tags: p.tags, source: p.source, limit: p.limit })}`),

  getImage: (id: string) => request<ImageDetail>(`/images/${id}`),

  updateImage: (id: string, patch: ImagePatch) =>
    request<UpdateResponse>(`/images/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }),

  deleteImage: (id: string) => request<void>(`/images/${id}`, { method: "DELETE" }),

  listTags: () => request<TagCount[]>("/tags"),

  health: () => request<{ status: string }>("/health"),

  /** Upload a single file (the dialog uploads files one by one for per-file progress). */
  uploadImage: (file: File, meta: UploadMeta, onProgress?: (fraction: number) => void) =>
    new Promise<UploadResult>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/images");
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress?.(e.loaded / e.total);
      };
      xhr.onload = () => {
        let body: unknown = null;
        try {
          body = JSON.parse(xhr.responseText);
        } catch {
          /* ignore */
        }
        if (xhr.status >= 200 && xhr.status < 300) resolve((body as UploadResult[])[0]);
        else reject(toApiError(xhr.status, xhr.statusText, body));
      };
      xhr.onerror = () => reject(new ApiError(0, "network_error", "Network error"));
      const form = new FormData();
      form.append("files", file);
      if (meta.tags.length) form.append("tags", meta.tags.join(","));
      if (meta.title.trim()) form.append("title", meta.title.trim());
      if (meta.description.trim()) form.append("description", meta.description.trim());
      xhr.send(form);
    }),
};
```

- [ ] **Step 6: Run the tests, the typecheck and the dev server**

Run: `npm test && npm run typecheck`
Expected: all tests pass, with no type errors.

Run: `npm run dev`, open http://localhost:5173, and confirm "Photos" renders in a styled font (which shows Tailwind is active). Then stop the dev server.

- [ ] **Step 7: Commit**

```bash
git add frontend/
git commit -m "feat(frontend): scaffold Vite React app with typed API client"
```

---

### Task 13: Gallery, search bar and tag filter

**Files:**
- Create: `frontend/src/hooks/useSearchState.ts`, `frontend/src/components/TagInput.tsx`, `frontend/src/components/SearchBar.tsx`, `frontend/src/components/ImageCard.tsx`, `frontend/src/components/Gallery.tsx`, `frontend/src/components/HealthBanner.tsx`
- Modify: `frontend/src/App.tsx` (replace)
- Test: `frontend/src/components/TagInput.test.tsx`, `frontend/src/components/Gallery.test.tsx`

**Interfaces:**
- Consumes: `api`, the types, `normalizeTag`.
- Produces:
  - `TagInput({ label, value, onChange, suggestions?, placeholder? })`. The input's accessible name is `label`. Remove buttons are labelled `Remove tag <name>`.
  - `useSearchState(): [{ q: string; tags: string[] }, (next: Partial<...>) => void]`, which syncs with `?q=&tags=`
  - `SearchBar({ q, tags, suggestions, onChange })`, whose search input is labelled "Search photos"
  - `Gallery({ q, tags, source, onOpen(id), onUpload() })`
  - `ImageCard({ image, score?, onOpen })`, a button labelled with `title || filename`
  - `HealthBanner()`
  - Query keys: `["images", ...]`, `["search", ...]`, `["tags"]`, `["image", id]`, `["health"]`

- [ ] **Step 1: Write the failing tests**

`frontend/src/components/TagInput.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { TagInput } from "./TagInput";

function Harness({ initial = [] as string[], suggestions = [] as string[] }) {
  const [tags, setTags] = useState(initial);
  return (
    <>
      <TagInput label="Tags" value={tags} onChange={setTags} suggestions={suggestions} />
      <output data-testid="value">{tags.join("|")}</output>
    </>
  );
}

describe("TagInput", () => {
  it("adds normalised tags on Enter and comma, ignoring duplicates", async () => {
    render(<Harness />);
    const input = screen.getByLabelText("Tags");
    await userEvent.type(input, "Goa Trip{Enter}family,goa trip{Enter}");
    expect(screen.getByTestId("value")).toHaveTextContent("goa-trip|family");
    expect(input).toHaveValue("");
  });

  it("shows an error for invalid tags", async () => {
    render(<Harness />);
    await userEvent.type(screen.getByLabelText("Tags"), "#fun!{Enter}");
    expect(screen.getByRole("alert")).toHaveTextContent(/letters, digits/);
    expect(screen.getByTestId("value")).toHaveTextContent("");
  });

  it("removes with the chip button and with Backspace on empty input", async () => {
    render(<Harness initial={["a", "b", "c"]} />);
    await userEvent.click(screen.getByRole("button", { name: "Remove tag b" }));
    expect(screen.getByTestId("value")).toHaveTextContent("a|c");
    await userEvent.type(screen.getByLabelText("Tags"), "{Backspace}");
    expect(screen.getByTestId("value")).toHaveTextContent("a");
  });

  it("suggests matching existing tags", async () => {
    render(<Harness suggestions={["goa", "goal", "beach"]} initial={["goal"]} />);
    await userEvent.type(screen.getByLabelText("Tags"), "go");
    const options = screen.getAllByRole("option");
    expect(options.map((o) => o.textContent)).toEqual(["goa"]);
    await userEvent.click(options[0]);
    expect(screen.getByTestId("value")).toHaveTextContent("goal|goa");
  });
});
```

`frontend/src/components/Gallery.test.tsx`:

```tsx
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import type { ImageSummary } from "../api/types";
import { renderWithClient } from "../test/utils";
import { Gallery } from "./Gallery";

vi.mock("../api/client", () => ({
  api: { listImages: vi.fn(), searchImages: vi.fn() },
  ApiError: class extends Error {},
}));

const img = (id: string, title: string | null = null): ImageSummary => ({
  id, filename: `${id}.png`, title, description: null, tags: [], width: 10, height: 10,
  taken_at: null, created_at: "2026-01-01T00:00:00Z", source: "upload", indexed: true,
  thumb_url: `http://s3/${id}.webp`,
});

beforeEach(() => vi.resetAllMocks());

describe("Gallery", () => {
  it("lists library images and opens one on click", async () => {
    vi.mocked(api.listImages).mockResolvedValue({ items: [img("a", "Beach"), img("b")], page: 1, page_size: 60, total: 2 });
    const onOpen = vi.fn();
    renderWithClient(<Gallery q="" tags={[]} source={undefined} onOpen={onOpen} onUpload={vi.fn()} />);
    await userEvent.click(await screen.findByRole("button", { name: "Beach" }));
    expect(onOpen).toHaveBeenCalledWith("a");
    expect(screen.getByRole("button", { name: "b.png" })).toBeInTheDocument();
  });

  it("shows search results with scores when q is set", async () => {
    vi.mocked(api.searchImages).mockResolvedValue({ items: [{ ...img("a"), score: 0.87 }] });
    renderWithClient(<Gallery q="red car" tags={["goa"]} source="upload" onOpen={vi.fn()} onUpload={vi.fn()} />);
    expect(await screen.findByText("87%")).toBeInTheDocument();
    expect(api.searchImages).toHaveBeenCalledWith({ q: "red car", tags: ["goa"], source: "upload", limit: 60 });
  });

  it("shows the empty library state with an upload action", async () => {
    vi.mocked(api.listImages).mockResolvedValue({ items: [], page: 1, page_size: 60, total: 0 });
    const onUpload = vi.fn();
    renderWithClient(<Gallery q="" tags={[]} source={undefined} onOpen={vi.fn()} onUpload={onUpload} />);
    await userEvent.click(await screen.findByRole("button", { name: "Upload photos" }));
    expect(onUpload).toHaveBeenCalled();
  });

  it("shows no-matches state for an empty search", async () => {
    vi.mocked(api.searchImages).mockResolvedValue({ items: [] });
    renderWithClient(<Gallery q="unicorn" tags={[]} source={undefined} onOpen={vi.fn()} onUpload={vi.fn()} />);
    expect(await screen.findByText("No matching photos")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run to verify they fail**

Run: `npm test`
Expected: FAIL, because `./TagInput` and `./Gallery` cannot be resolved.

- [ ] **Step 3: Implement the components**

`frontend/src/components/TagInput.tsx`:

```tsx
import { useId, useState, type KeyboardEvent } from "react";
import { normalizeTag } from "../lib/tags";

interface TagInputProps {
  label: string;
  value: string[];
  onChange: (tags: string[]) => void;
  suggestions?: string[];
  placeholder?: string;
}

export function TagInput({ label, value, onChange, suggestions = [], placeholder }: TagInputProps) {
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [focused, setFocused] = useState(false);
  const listId = useId();

  const prefix = text.trim().toLowerCase();
  const matches = prefix
    ? suggestions.filter((s) => s.startsWith(prefix) && !value.includes(s)).slice(0, 8)
    : [];

  function add(raw: string) {
    const tag = normalizeTag(raw);
    if (!tag) {
      setError("Tags use letters, digits and hyphens (max 40 characters)");
      return;
    }
    setError(null);
    if (!value.includes(tag)) onChange([...value, tag]);
    setText("");
  }

  function onKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if ((e.key === "Enter" || e.key === ",") && text.trim()) {
      e.preventDefault();
      add(text);
    } else if (e.key === "Backspace" && !text && value.length) {
      onChange(value.slice(0, -1));
    }
  }

  return (
    <div className="relative">
      <div className="flex flex-wrap items-center gap-1 rounded-lg border border-stone-300 bg-white px-2 py-1 focus-within:ring-2 focus-within:ring-amber-400">
        {value.map((tag) => (
          <span key={tag} className="flex items-center gap-1 rounded-full bg-amber-100 px-2 py-0.5 text-sm text-amber-900">
            {tag}
            <button
              type="button"
              aria-label={`Remove tag ${tag}`}
              className="text-amber-700 hover:text-amber-950"
              onClick={() => onChange(value.filter((t) => t !== tag))}
            >
              ×
            </button>
          </span>
        ))}
        <input
          aria-label={label}
          aria-controls={listId}
          className="min-w-24 flex-1 bg-transparent py-1 text-sm outline-none"
          value={text}
          placeholder={value.length ? "" : placeholder}
          onChange={(e) => {
            setText(e.target.value.replace(",", ""));
            setError(null);
          }}
          onKeyDown={onKeyDown}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
        />
      </div>
      {error && (
        <p role="alert" className="mt-1 text-xs text-red-600">
          {error}
        </p>
      )}
      {focused && matches.length > 0 && (
        <ul id={listId} role="listbox" className="absolute z-30 mt-1 w-full rounded-lg border bg-white py-1 shadow-lg">
          {matches.map((s) => (
            <li
              key={s}
              role="option"
              aria-selected={false}
              className="cursor-pointer px-3 py-1 text-sm hover:bg-amber-50"
              onMouseDown={(e) => {
                e.preventDefault();
                add(s);
              }}
            >
              {s}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
```

> Note: `userEvent.click` on an option fires `mousedown`, so the test's `click(options[0])` exercises `onMouseDown`.

`frontend/src/hooks/useSearchState.ts`:

```ts
import { useCallback, useEffect, useState } from "react";

export interface SearchState {
  q: string;
  tags: string[];
}

function read(search: string): SearchState {
  const sp = new URLSearchParams(search);
  return { q: sp.get("q") ?? "", tags: sp.getAll("tags") };
}

function toUrl(state: SearchState): string {
  const sp = new URLSearchParams();
  if (state.q) sp.set("q", state.q);
  state.tags.forEach((t) => sp.append("tags", t));
  const s = sp.toString();
  return s ? `?${s}` : window.location.pathname;
}

export function useSearchState() {
  const [state, setState] = useState<SearchState>(() => read(window.location.search));

  useEffect(() => {
    const onPop = () => setState(read(window.location.search));
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  useEffect(() => {
    window.history.replaceState(null, "", toUrl(state));
  }, [state]);

  const update = useCallback((next: Partial<SearchState>) => setState((prev) => ({ ...prev, ...next })), []);
  return [state, update] as const;
}
```

`frontend/src/components/SearchBar.tsx`:

```tsx
import { useEffect, useState } from "react";
import type { SearchState } from "../hooks/useSearchState";
import { TagInput } from "./TagInput";

interface SearchBarProps {
  q: string;
  tags: string[];
  suggestions: string[];
  onChange: (next: Partial<SearchState>) => void;
}

export function SearchBar({ q, tags, suggestions, onChange }: SearchBarProps) {
  const [text, setText] = useState(q);

  useEffect(() => setText(q), [q]);

  useEffect(() => {
    if (text.trim() === q) return;
    const timer = setTimeout(() => onChange({ q: text.trim() }), 400);
    return () => clearTimeout(timer);
  }, [text, q, onChange]);

  return (
    <form
      role="search"
      className="flex flex-1 flex-wrap items-start gap-2"
      onSubmit={(e) => {
        e.preventDefault();
        onChange({ q: text.trim() });
      }}
    >
      <input
        aria-label="Search photos"
        type="search"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder='Describe a photo, e.g. "red car at night"'
        className="min-w-64 flex-1 rounded-lg border border-stone-300 bg-white px-4 py-2 outline-none focus:ring-2 focus:ring-amber-400"
      />
      <div className="w-72">
        <TagInput
          label="Filter by tag"
          placeholder="Filter by tag"
          value={tags}
          suggestions={suggestions}
          onChange={(next) => onChange({ tags: next })}
        />
      </div>
    </form>
  );
}
```

`frontend/src/components/ImageCard.tsx`:

```tsx
import type { ImageSummary } from "../api/types";

interface ImageCardProps {
  image: ImageSummary;
  score?: number | null;
  onOpen: (id: string) => void;
}

export function ImageCard({ image, score, onOpen }: ImageCardProps) {
  const name = image.title || image.filename;
  return (
    <button
      type="button"
      aria-label={name}
      onClick={() => onOpen(image.id)}
      className="group relative aspect-square overflow-hidden rounded-lg bg-stone-200 focus:outline-none focus:ring-2 focus:ring-amber-400"
    >
      <img src={image.thumb_url} alt={name} loading="lazy" className="h-full w-full object-cover transition group-hover:scale-105" />
      {score != null && (
        <span className="absolute right-1 top-1 rounded bg-black/60 px-1.5 py-0.5 text-xs text-white opacity-0 group-hover:opacity-100">
          {Math.round(score * 100)}%
        </span>
      )}
      {image.tags.length > 0 && (
        <span className="absolute inset-x-0 bottom-0 truncate bg-gradient-to-t from-black/60 px-2 pb-1 pt-4 text-left text-xs text-white opacity-0 group-hover:opacity-100">
          {image.tags.join(" · ")}
        </span>
      )}
    </button>
  );
}
```

`frontend/src/components/Gallery.tsx`:

```tsx
import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { api } from "../api/client";
import type { Source } from "../api/types";
import { ImageCard } from "./ImageCard";

const PAGE_SIZE = 60;

interface GalleryProps {
  q: string;
  tags: string[];
  source: Source | undefined;
  onOpen: (id: string) => void;
  onUpload: () => void;
}

const grid = "grid grid-cols-2 gap-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6";

export function Gallery({ q, tags, source, onOpen, onUpload }: GalleryProps) {
  const searching = q.trim().length > 0;

  const search = useQuery({
    queryKey: ["search", q, tags, source],
    queryFn: () => api.searchImages({ q, tags, source, limit: PAGE_SIZE }),
    enabled: searching,
  });

  const library = useInfiniteQuery({
    queryKey: ["images", "list", tags, source],
    queryFn: ({ pageParam }) => api.listImages({ page: pageParam, page_size: PAGE_SIZE, tags, source }),
    initialPageParam: 1,
    getNextPageParam: (last) => (last.page * last.page_size < last.total ? last.page + 1 : undefined),
    enabled: !searching,
  });

  const sentinel = useRef<HTMLDivElement>(null);
  const { hasNextPage, fetchNextPage, isFetchingNextPage } = library;
  useEffect(() => {
    const el = sentinel.current;
    if (!el || !("IntersectionObserver" in window)) return;
    const observer = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting && hasNextPage && !isFetchingNextPage) fetchNextPage();
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [hasNextPage, fetchNextPage, isFetchingNextPage]);

  const active = searching ? search : library;
  if (active.isPending) return <p className="py-16 text-center text-stone-500">Loading…</p>;
  if (active.isError) return <p className="py-16 text-center text-red-600">{active.error.message}</p>;

  if (searching) {
    const items = search.data?.items ?? [];
    if (!items.length) return <p className="py-16 text-center text-stone-500">No matching photos</p>;
    return (
      <div className={grid}>
        {items.map((item) => (
          <ImageCard key={item.id} image={item} score={item.score} onOpen={onOpen} />
        ))}
      </div>
    );
  }

  const items = library.data?.pages.flatMap((p) => p.items) ?? [];
  if (!items.length) {
    if (tags.length) return <p className="py-16 text-center text-stone-500">No matching photos</p>;
    return (
      <div className="py-16 text-center">
        <p className="mb-4 text-stone-500">No photos yet.</p>
        <button type="button" onClick={onUpload} className="rounded-lg bg-amber-500 px-4 py-2 font-medium text-white hover:bg-amber-600">
          Upload photos
        </button>
      </div>
    );
  }
  return (
    <>
      <div className={grid}>
        {items.map((item) => (
          <ImageCard key={item.id} image={item} onOpen={onOpen} />
        ))}
      </div>
      <div ref={sentinel} className="h-8" />
      {isFetchingNextPage && <p className="py-4 text-center text-stone-500">Loading more…</p>}
    </>
  );
}
```

`frontend/src/components/HealthBanner.tsx`:

```tsx
import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";

export function HealthBanner() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 30_000, retry: false });
  if (!health.isError) return null;
  return (
    <div role="alert" className="bg-red-600 px-4 py-2 text-center text-sm text-white">
      Backend services unavailable — is <code>docker compose up -d</code> running and the API started?
    </div>
  );
}
```

`frontend/src/App.tsx` (replace):

```tsx
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Toaster } from "sonner";
import { api } from "./api/client";
import type { Source } from "./api/types";
import { Gallery } from "./components/Gallery";
import { HealthBanner } from "./components/HealthBanner";
import { SearchBar } from "./components/SearchBar";
import { useSearchState } from "./hooks/useSearchState";

export default function App() {
  const [search, setSearch] = useSearchState();
  const [, setOpenId] = useState<string | null>(null);
  const [, setUploadOpen] = useState(false);
  const [demoChoice, setDemoChoice] = useState<boolean | null>(null);

  const tags = useQuery({ queryKey: ["tags"], queryFn: api.listTags });
  const ownCount = useQuery({
    queryKey: ["images", "own-count"],
    queryFn: () => api.listImages({ page: 1, page_size: 1, source: "upload" }),
  });
  // Show demo images by default until the user has uploaded their own.
  const showDemo = demoChoice ?? (ownCount.data ? ownCount.data.total === 0 : true);
  const source: Source | undefined = showDemo ? undefined : "upload";

  return (
    <div className="min-h-screen bg-stone-50 text-stone-900">
      <HealthBanner />
      <header className="sticky top-0 z-10 border-b border-stone-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-start gap-3 px-4 py-3">
          <h1 className="py-2 text-lg font-semibold">Photos</h1>
          <SearchBar q={search.q} tags={search.tags} suggestions={tags.data?.map((t) => t.name) ?? []} onChange={setSearch} />
          <label className="flex items-center gap-2 py-2 text-sm text-stone-600">
            <input type="checkbox" checked={showDemo} onChange={(e) => setDemoChoice(e.target.checked)} />
            Show demo images
          </label>
          <button
            type="button"
            onClick={() => setUploadOpen(true)}
            className="rounded-lg bg-amber-500 px-4 py-2 font-medium text-white hover:bg-amber-600"
          >
            Upload
          </button>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-6">
        <Gallery q={search.q} tags={search.tags} source={source} onOpen={setOpenId} onUpload={() => setUploadOpen(true)} />
      </main>
      <Toaster position="bottom-right" richColors />
    </div>
  );
}
```

- [ ] **Step 4: Run tests and the typecheck**

Run: `npm test && npm run typecheck`
Expected: all pass.

- [ ] **Step 5: Check manually in the browser**

With infra up, the API running (`../venv/Scripts/python -m uvicorn app.main:app --port 8080` from `backend/`) and demo data seeded (Task 11), run `npm run dev` and open http://localhost:5173.
Expected: the CIFAR thumbnails show in a grid. Typing "truck" shows ranked trucks with % badges on hover. Adding the `frog` filter narrows the results, and the URL updates to `?q=truck&tags=frog`.

- [ ] **Step 6: Commit**

```bash
git add frontend/
git commit -m "feat(frontend): gallery with infinite scroll, semantic search bar and tag filter"
```

---

### Task 14: Image detail drawer (edit and delete)

**Files:**
- Create: `frontend/src/components/DetailDrawer.tsx`
- Modify: `frontend/src/App.tsx`
- Test: `frontend/src/components/DetailDrawer.test.tsx`

**Interfaces:**
- Consumes: `api.getImage`, `api.updateImage`, `api.deleteImage`, `api.listTags`, `TagInput`.
- Produces: `DetailDrawer({ imageId: string | null, onClose })`, which renders `role="dialog"` named "Photo details", the fields "Title", "Description" and "Tags", and the buttons "Save" and "Delete".

- [ ] **Step 1: Write the failing test**

`frontend/src/components/DetailDrawer.test.tsx`:

```tsx
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import type { ImageDetail } from "../api/types";
import { renderWithClient } from "../test/utils";
import { DetailDrawer } from "./DetailDrawer";

vi.mock("../api/client", () => ({
  api: { getImage: vi.fn(), updateImage: vi.fn(), deleteImage: vi.fn(), listTags: vi.fn() },
  ApiError: class extends Error {},
}));

const detail: ImageDetail = {
  id: "a", filename: "a.jpg", title: "Old", description: "desc", tags: ["goa"], width: 4000, height: 3000,
  taken_at: "2024-05-01T10:20:30Z", created_at: "2026-01-01T00:00:00Z", source: "upload", indexed: true,
  thumb_url: "http://s3/t.webp", original_url: "http://s3/o.jpg", mime_type: "image/jpeg",
};

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.getImage).mockResolvedValue(detail);
  vi.mocked(api.listTags).mockResolvedValue([{ name: "goa", count: 1 }]);
});

describe("DetailDrawer", () => {
  it("renders nothing without an image id", () => {
    renderWithClient(<DetailDrawer imageId={null} onClose={vi.fn()} />);
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("shows details and saves only changed fields", async () => {
    vi.mocked(api.updateImage).mockResolvedValue({ image: { ...detail, title: "New" }, warning: null });
    renderWithClient(<DetailDrawer imageId="a" onClose={vi.fn()} />);
    const title = await screen.findByLabelText("Title");
    expect(title).toHaveValue("Old");
    expect(screen.getByText("4000 × 3000")).toBeInTheDocument();
    await userEvent.clear(title);
    await userEvent.type(title, "New");
    await userEvent.type(screen.getByLabelText("Tags"), "Beach Day{Enter}");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(api.updateImage).toHaveBeenCalledWith("a", { title: "New", tags: ["goa", "beach-day"] });
  });

  it("deletes after confirmation and closes", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    vi.mocked(api.deleteImage).mockResolvedValue(undefined);
    const onClose = vi.fn();
    renderWithClient(<DetailDrawer imageId="a" onClose={onClose} />);
    await userEvent.click(await screen.findByRole("button", { name: "Delete" }));
    expect(api.deleteImage).toHaveBeenCalledWith("a");
    await vi.waitFor(() => expect(onClose).toHaveBeenCalled());
  });

  it("does not delete when confirmation is cancelled", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(false);
    renderWithClient(<DetailDrawer imageId="a" onClose={vi.fn()} />);
    await userEvent.click(await screen.findByRole("button", { name: "Delete" }));
    expect(api.deleteImage).not.toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npm test -- DetailDrawer`
Expected: FAIL, because `./DetailDrawer` cannot be resolved.

- [ ] **Step 3: Implement `frontend/src/components/DetailDrawer.tsx`**

```tsx
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { api } from "../api/client";
import type { ImageDetail, ImagePatch } from "../api/types";
import { TagInput } from "./TagInput";

interface DetailDrawerProps {
  imageId: string | null;
  onClose: () => void;
}

export function DetailDrawer({ imageId, onClose }: DetailDrawerProps) {
  const query = useQuery({
    queryKey: ["image", imageId],
    queryFn: () => api.getImage(imageId!),
    enabled: imageId !== null,
  });

  useEffect(() => {
    if (!imageId) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [imageId, onClose]);

  if (!imageId) return null;
  return (
    <div className="fixed inset-0 z-20 flex justify-end">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <aside role="dialog" aria-label="Photo details" className="relative flex h-full w-full max-w-xl flex-col overflow-y-auto bg-white shadow-xl">
        <button type="button" aria-label="Close" onClick={onClose} className="absolute right-3 top-3 z-10 rounded-full bg-white/80 px-2 text-xl">
          ×
        </button>
        {query.isPending && <p className="p-6 text-stone-500">Loading…</p>}
        {query.isError && <p className="p-6 text-red-600">{query.error.message}</p>}
        {query.data && <DetailForm key={query.data.id} image={query.data} onClose={onClose} />}
      </aside>
    </div>
  );
}

function sameTags(a: string[], b: string[]) {
  return a.length === b.length && a.every((t, i) => t === b[i]);
}

function DetailForm({ image, onClose }: { image: ImageDetail; onClose: () => void }) {
  const qc = useQueryClient();
  const [title, setTitle] = useState(image.title ?? "");
  const [description, setDescription] = useState(image.description ?? "");
  const [tags, setTags] = useState(image.tags);
  const tagList = useQuery({ queryKey: ["tags"], queryFn: api.listTags });

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ["images"] });
    qc.invalidateQueries({ queryKey: ["search"] });
    qc.invalidateQueries({ queryKey: ["tags"] });
  };

  const save = useMutation({
    mutationFn: (patch: ImagePatch) => api.updateImage(image.id, patch),
    onSuccess: (res) => {
      qc.setQueryData(["image", image.id], res.image);
      invalidate();
      if (res.warning) toast.warning(res.warning);
      else toast.success("Saved");
    },
    onError: (e) => toast.error(e.message),
  });

  const remove = useMutation({
    mutationFn: () => api.deleteImage(image.id),
    onSuccess: () => {
      qc.removeQueries({ queryKey: ["image", image.id] });
      invalidate();
      toast.success("Photo deleted");
      onClose();
    },
    onError: (e) => toast.error(e.message),
  });

  function onSave() {
    const patch: ImagePatch = {};
    if (title.trim() !== (image.title ?? "")) patch.title = title.trim() || null;
    if (description.trim() !== (image.description ?? "")) patch.description = description.trim() || null;
    if (!sameTags(tags, image.tags)) patch.tags = tags;
    if (Object.keys(patch).length === 0) {
      toast("Nothing to save");
      return;
    }
    save.mutate(patch);
  }

  function onDelete() {
    if (window.confirm("Delete this photo permanently?")) remove.mutate();
  }

  const field = "w-full rounded-lg border border-stone-300 px-3 py-2 outline-none focus:ring-2 focus:ring-amber-400";
  return (
    <div className="flex flex-col">
      <a href={image.original_url} target="_blank" rel="noreferrer" className="bg-stone-900">
        <img src={image.original_url} alt={image.title || image.filename} className="max-h-[60vh] w-full object-contain" />
      </a>
      <div className="space-y-4 p-6">
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm text-stone-600">
          <dt>File</dt>
          <dd className="truncate">{image.filename}</dd>
          <dt>Size</dt>
          <dd>
            {image.width} × {image.height}
          </dd>
          <dt>Taken</dt>
          <dd>{image.taken_at ? new Date(image.taken_at).toLocaleString() : "—"}</dd>
          {!image.indexed && (
            <>
              <dt>Search</dt>
              <dd className="text-amber-700">Not indexed yet</dd>
            </>
          )}
        </dl>
        <label className="block text-sm font-medium">
          Title
          <input className={`${field} mt-1`} value={title} maxLength={200} onChange={(e) => setTitle(e.target.value)} />
        </label>
        <label className="block text-sm font-medium">
          Description
          <textarea className={`${field} mt-1`} rows={3} value={description} maxLength={2000} onChange={(e) => setDescription(e.target.value)} />
        </label>
        <div className="text-sm font-medium">
          <span className="mb-1 block">Tags</span>
          <TagInput label="Tags" value={tags} onChange={setTags} suggestions={tagList.data?.map((t) => t.name) ?? []} placeholder="Add a tag and press Enter" />
        </div>
        <div className="flex justify-between pt-2">
          <button type="button" onClick={onDelete} disabled={remove.isPending} className="rounded-lg px-4 py-2 text-red-600 hover:bg-red-50">
            Delete
          </button>
          <button type="button" onClick={onSave} disabled={save.isPending} className="rounded-lg bg-amber-500 px-4 py-2 font-medium text-white hover:bg-amber-600 disabled:opacity-50">
            Save
          </button>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Wire the drawer into `App.tsx`**

In `frontend/src/App.tsx`:
- Add the import `import { DetailDrawer } from "./components/DetailDrawer";`
- Change `const [, setOpenId] = useState<string | null>(null);` to `const [openId, setOpenId] = useState<string | null>(null);`
- Insert `<DetailDrawer imageId={openId} onClose={() => setOpenId(null)} />` immediately before `<Toaster ... />`.

- [ ] **Step 5: Run tests and the typecheck**

Run: `npm test && npm run typecheck`
Expected: all pass.

- [ ] **Step 6: Check manually in the browser**

Click a thumbnail, give it the title "Sunset" and the tag `goa-trip`, then save. Search for "goa trip".
Expected: the image now ranks at or near the top. Deleting it removes it from the grid.

- [ ] **Step 7: Commit**

```bash
git add frontend/
git commit -m "feat(frontend): detail drawer to edit title, description, tags and delete"
```

---

### Task 15: Upload dialog

**Files:**
- Create: `frontend/src/components/UploadDialog.tsx`
- Modify: `frontend/src/App.tsx`
- Test: `frontend/src/components/UploadDialog.test.tsx`

**Interfaces:**
- Consumes: `api.uploadImage`, `api.listTags`, `TagInput`.
- Produces: `UploadDialog({ open, onClose })`, with a file input labelled "Choose files", a tag input labelled "Tags for upload", the fields "Title" and "Description", a submit button "Upload N file(s)", and per-file status text "Added", "Duplicate" or "Failed".

- [ ] **Step 1: Write the failing test**

`frontend/src/components/UploadDialog.test.tsx`:

```tsx
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import { renderWithClient } from "../test/utils";
import { UploadDialog } from "./UploadDialog";

vi.mock("../api/client", () => ({
  api: { uploadImage: vi.fn(), listTags: vi.fn() },
  ApiError: class extends Error {},
}));

const file = (name: string) => new File([new Uint8Array([1, 2, 3])], name, { type: "image/png" });

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.listTags).mockResolvedValue([]);
});

describe("UploadDialog", () => {
  it("renders nothing when closed", () => {
    renderWithClient(<UploadDialog open={false} onClose={vi.fn()} />);
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("uploads each file with shared metadata and shows per-file results", async () => {
    vi.mocked(api.uploadImage)
      .mockResolvedValueOnce({ filename: "a.png", status: "created", id: "1", message: null })
      .mockResolvedValueOnce({ filename: "b.png", status: "duplicate", id: "2", message: "Already in your library" })
      .mockRejectedValueOnce(new Error("Network error"));
    renderWithClient(<UploadDialog open onClose={vi.fn()} />);

    await userEvent.upload(screen.getByLabelText("Choose files"), [file("a.png"), file("b.png"), file("c.png")]);
    await userEvent.type(screen.getByLabelText("Tags for upload"), "Goa Trip{Enter}");
    await userEvent.type(screen.getByLabelText("Title"), "Holiday");
    await userEvent.click(screen.getByRole("button", { name: "Upload 3 files" }));

    expect(await screen.findByText("Added")).toBeInTheDocument();
    expect(await screen.findByText("Duplicate")).toBeInTheDocument();
    expect(await screen.findByText("Failed")).toBeInTheDocument();
    expect(screen.getByText("1 added · 1 duplicate · 1 failed")).toBeInTheDocument();
    expect(api.uploadImage).toHaveBeenCalledTimes(3);
    expect(vi.mocked(api.uploadImage).mock.calls[0][1]).toEqual({ tags: ["goa-trip"], title: "Holiday", description: "" });
  });

  it("disables upload with no files", () => {
    renderWithClient(<UploadDialog open onClose={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Upload 0 files" })).toBeDisabled();
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npm test -- UploadDialog`
Expected: FAIL, because `./UploadDialog` cannot be resolved.

- [ ] **Step 3: Implement `frontend/src/components/UploadDialog.tsx`**

```tsx
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { TagInput } from "./TagInput";

type Status = "pending" | "uploading" | "created" | "duplicate" | "error";

interface Entry {
  file: File;
  preview: string;
  status: Status;
  progress: number;
  message: string | null;
}

const STATUS_LABEL: Record<Status, string> = {
  pending: "Ready",
  uploading: "Uploading…",
  created: "Added",
  duplicate: "Duplicate",
  error: "Failed",
};

interface UploadDialogProps {
  open: boolean;
  onClose: () => void;
}

export function UploadDialog({ open, onClose }: UploadDialogProps) {
  const qc = useQueryClient();
  const [entries, setEntries] = useState<Entry[]>([]);
  const [tags, setTags] = useState<string[]>([]);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const tagList = useQuery({ queryKey: ["tags"], queryFn: api.listTags, enabled: open });
  const previews = useRef<string[]>([]);

  // Revoke preview object URLs only on unmount (close() revokes them too).
  useEffect(() => () => previews.current.forEach((url) => URL.revokeObjectURL(url)), []);

  if (!open) return null;

  function addFiles(files: FileList | File[]) {
    const next = Array.from(files).map((file) => ({
      file,
      preview: typeof URL.createObjectURL === "function" ? URL.createObjectURL(file) : "",
      status: "pending" as Status,
      progress: 0,
      message: null,
    }));
    previews.current.push(...next.map((e) => e.preview).filter(Boolean));
    setEntries((prev) => [...prev, ...next]);
  }

  function patch(index: number, changes: Partial<Entry>) {
    setEntries((prev) => prev.map((e, i) => (i === index ? { ...e, ...changes } : e)));
  }

  async function uploadAll() {
    setBusy(true);
    const meta = { tags, title, description };
    for (let i = 0; i < entries.length; i++) {
      if (entries[i].status !== "pending") continue;
      patch(i, { status: "uploading" });
      try {
        const res = await api.uploadImage(entries[i].file, meta, (p) => patch(i, { progress: p }));
        patch(i, { status: res.status, progress: 1, message: res.message });
      } catch (e) {
        patch(i, { status: "error", message: (e as Error).message });
      }
    }
    setBusy(false);
    qc.invalidateQueries({ queryKey: ["images"] });
    qc.invalidateQueries({ queryKey: ["search"] });
    qc.invalidateQueries({ queryKey: ["tags"] });
  }

  function close() {
    if (busy) return;
    previews.current.forEach((url) => URL.revokeObjectURL(url));
    previews.current = [];
    setEntries([]);
    setTags([]);
    setTitle("");
    setDescription("");
    onClose();
  }

  const pending = entries.filter((e) => e.status === "pending").length;
  const done = entries.filter((e) => ["created", "duplicate", "error"].includes(e.status));
  const count = (s: Status) => entries.filter((e) => e.status === s).length;
  const field = "mt-1 w-full rounded-lg border border-stone-300 px-3 py-2 outline-none focus:ring-2 focus:ring-amber-400";

  return (
    <div className="fixed inset-0 z-30 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/40" onClick={close} />
      <div role="dialog" aria-label="Upload photos" className="relative max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-xl bg-white p-6 shadow-xl">
        <h2 className="mb-4 text-lg font-semibold">Upload photos</h2>

        <label
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            addFiles(e.dataTransfer.files);
          }}
          className={`flex cursor-pointer flex-col items-center rounded-lg border-2 border-dashed p-8 text-stone-500 ${dragging ? "border-amber-400 bg-amber-50" : "border-stone-300"}`}
        >
          Drag photos here or click to choose
          <input
            type="file"
            multiple
            accept="image/jpeg,image/png,image/webp,image/heic,.heic"
            aria-label="Choose files"
            className="sr-only"
            onChange={(e) => e.target.files && addFiles(e.target.files)}
          />
        </label>

        {entries.length > 0 && (
          <ul className="mt-4 max-h-60 space-y-2 overflow-y-auto">
            {entries.map((e, i) => (
              <li key={i} className="flex items-center gap-3 text-sm">
                {e.preview ? <img src={e.preview} alt="" className="h-10 w-10 rounded object-cover" /> : <div className="h-10 w-10 rounded bg-stone-200" />}
                <div className="min-w-0 flex-1">
                  <p className="truncate">{e.file.name}</p>
                  {e.status === "uploading" && (
                    <div className="h-1 rounded bg-stone-200">
                      <div className="h-1 rounded bg-amber-500" style={{ width: `${Math.round(e.progress * 100)}%` }} />
                    </div>
                  )}
                  {e.message && <p className="truncate text-xs text-stone-500">{e.message}</p>}
                </div>
                <span className={e.status === "error" ? "text-red-600" : e.status === "created" ? "text-green-700" : "text-stone-600"}>
                  {STATUS_LABEL[e.status]}
                </span>
              </li>
            ))}
          </ul>
        )}

        <div className="mt-4 space-y-3">
          <div className="text-sm font-medium">
            <span className="mb-1 block">Tags (applied to all)</span>
            <TagInput label="Tags for upload" value={tags} onChange={setTags} suggestions={tagList.data?.map((t) => t.name) ?? []} placeholder="Add a tag and press Enter" />
          </div>
          <label className="block text-sm font-medium">
            Title
            <input className={field} value={title} maxLength={200} onChange={(e) => setTitle(e.target.value)} />
          </label>
          <label className="block text-sm font-medium">
            Description
            <textarea className={field} rows={2} value={description} maxLength={2000} onChange={(e) => setDescription(e.target.value)} />
          </label>
        </div>

        <div className="mt-6 flex items-center justify-between">
          <p className="text-sm text-stone-600">
            {done.length > 0 && `${count("created")} added · ${count("duplicate")} duplicate · ${count("error")} failed`}
          </p>
          <div className="flex gap-2">
            <button type="button" onClick={close} disabled={busy} className="rounded-lg px-4 py-2 hover:bg-stone-100">
              {done.length ? "Done" : "Cancel"}
            </button>
            <button
              type="button"
              onClick={uploadAll}
              disabled={busy || pending === 0}
              className="rounded-lg bg-amber-500 px-4 py-2 font-medium text-white hover:bg-amber-600 disabled:opacity-50"
            >
              {`Upload ${pending} file${pending === 1 ? "" : "s"}`}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Wire the dialog into `App.tsx`**

In `frontend/src/App.tsx`:
- Add the import `import { UploadDialog } from "./components/UploadDialog";`
- Change `const [, setUploadOpen] = useState(false);` to `const [uploadOpen, setUploadOpen] = useState(false);`
- Insert `<UploadDialog open={uploadOpen} onClose={() => setUploadOpen(false)} />` immediately before `<Toaster ... />`.

- [ ] **Step 5: Run tests and the typecheck**

Run: `npm test && npm run typecheck`
Expected: all pass.

- [ ] **Step 6: Check manually in the browser**

Upload two of your own photos (one from a phone if you can) with the tag `test-upload`. Then reupload one of them.
Expected: the results show "Added" for the new photos and "Duplicate" for the reupload. The "Show demo images" toggle now defaults to off, and your photos show with the correct orientation.

- [ ] **Step 7: Commit**

```bash
git add frontend/
git commit -m "feat(frontend): drag-and-drop upload dialog with shared tags and per-file progress"
```

---

### Task 16: End-to-end smoke test (Playwright)

**Files:**
- Create: `frontend/playwright.config.ts`, `frontend/e2e/smoke.spec.ts`, `frontend/e2e/fixtures/red.png`

**Interfaces:**
- Consumes: the running API on 8080 (with real CLIP), the infra, and the labels from Tasks 13–15.

- [ ] **Step 1: Create the fixture and the config**

Run (from the repo root): `venv/Scripts/python -c "from PIL import Image; Image.new('RGB',(256,256),(220,20,20)).save('frontend/e2e/fixtures/red.png')"`. Create `frontend/e2e/fixtures/` first if it doesn't exist.

`frontend/playwright.config.ts`:

```ts
import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "e2e",
  timeout: 60_000,
  use: { baseURL: "http://localhost:5173" },
  webServer: {
    command: "npm run dev",
    url: "http://localhost:5173",
    reuseExistingServer: true,
  },
});
```

Run: `npx playwright install chromium`

- [ ] **Step 2: Write the smoke test**

`frontend/e2e/smoke.spec.ts`:

```ts
import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import path from "node:path";

// Requires: docker compose up -d, and the API running on :8080 (see README).
test("upload → tag → search finds it", async ({ page, request }) => {
  const tag = `smoke-${Date.now()}`;
  // Trailing bytes after IEND keep the PNG valid but make the hash unique per run.
  const png = Buffer.concat([readFileSync(path.join(import.meta.dirname, "fixtures/red.png")), Buffer.from(tag)]);

  await page.goto("/");
  await page.getByRole("button", { name: "Upload", exact: true }).click();
  await page.getByLabel("Choose files").setInputFiles({ name: "red.png", mimeType: "image/png", buffer: png });
  const tagBox = page.getByLabel("Tags for upload");
  await tagBox.fill(tag);
  await tagBox.press("Enter");
  await page.getByRole("button", { name: "Upload 1 file" }).click();
  await expect(page.getByText("Added")).toBeVisible();
  await page.getByRole("button", { name: "Done" }).click();

  const filter = page.getByLabel("Filter by tag");
  await filter.fill(tag);
  await filter.press("Enter");
  await page.getByLabel("Search photos").fill("a red square");
  await page.getByLabel("Search photos").press("Enter");
  await expect(page.getByRole("button", { name: "red.png" })).toBeVisible();

  // Cleanup
  const list = await (await request.get(`/api/images?tags=${tag}`)).json();
  for (const item of list.items) await request.delete(`/api/images/${item.id}`);
});
```

- [ ] **Step 3: Run it against the live stack**

Prerequisites: `docker compose up -d`, and the API running from `backend/` (`../venv/Scripts/python -m uvicorn app.main:app --port 8080`).
Run (from `frontend/`): `npm run e2e`
Expected: 1 passed.

- [ ] **Step 4: Commit**

```bash
git add frontend/
git commit -m "test(frontend): Playwright smoke test for upload, tag and search"
```

---

### Task 17: Hosting Dockerfiles, README and legacy cleanup

**Files:**
- Create: `backend/Dockerfile`, `backend/.dockerignore`, `frontend/Dockerfile`, `frontend/.dockerignore`, `frontend/nginx.conf.template`, `README.md`
- Delete: `app.py`, `build_dataset.py`, `build_index.py`, `clip_image_search.py`, `search.py`, `search_LEGACY.py`, `utils.py`

**Interfaces:**
- Produces: buildable images `photo-api` (port 8080, which runs migrations on start) and `photo-web` (port 80, which proxies `/api` to `$API_UPSTREAM`).

- [ ] **Step 1: Write the backend image**

`backend/Dockerfile`:

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# CPU build of torch for portability; swap the index URL for a CUDA build on GPU hosts.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY alembic.ini .
COPY alembic ./alembic
COPY app ./app

EXPOSE 8080
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8080"]
```

`backend/.dockerignore`:

```
tests/
__pycache__/
.pytest_cache/
*.pyc
.env
```

- [ ] **Step 2: Write the frontend image**

`frontend/nginx.conf.template`:

```nginx
server {
    listen 80;
    client_max_body_size 30m;
    root /usr/share/nginx/html;

    location /api/ {
        proxy_pass ${API_UPSTREAM};
        proxy_set_header Host $host;
        proxy_read_timeout 120s;
    }

    location / {
        try_files $uri /index.html;
    }
}
```

`frontend/Dockerfile`:

```dockerfile
FROM node:24-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
ENV API_UPSTREAM=http://api:8080
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
```

`frontend/.dockerignore`:

```
node_modules/
dist/
e2e/
test-results/
playwright-report/
```

- [ ] **Step 3: Build both images**

Run: `docker build -t photo-web frontend`
Expected: the build succeeds.

Run: `docker build -t photo-api backend`
Expected: the build succeeds. It is large because of torch, so the first build takes several minutes.

- [ ] **Step 4: Remove the legacy scripts**

Run: `git rm app.py build_dataset.py build_index.py clip_image_search.py search.py search_LEGACY.py utils.py`

Run: `grep -rn "from search import\|from utils import\|search_LEGACY" --include=*.py . | grep -v venv`
Expected: no output.

- [ ] **Step 5: Write `README.md`**

````markdown
# Photo Retrieval

Search your photos by describing them ("red car at night"). CLIP embeds every image; your
tags, titles and descriptions are embedded too and blended into the ranking.

## Architecture

- **Docker (infra):** floci (S3 emulator, :4566), Postgres 16 (:5432), ChromaDB (:8000)
- **Native:** FastAPI backend (`backend/`, :8080, CLIP on GPU if available) and React dashboard (`frontend/`, :5173)
- Postgres is the source of truth; ChromaDB is a rebuildable index (`python -m app.cli reindex --all`).

## Prerequisites

Docker Desktop, Python 3.12+ with a CUDA or CPU build of torch + torchvision, Node 24.

## First-time setup

```bash
cp .env.example .env
docker compose up -d
cd backend
../venv/Scripts/python -m pip install -r requirements-dev.txt   # torch must already be installed
../venv/Scripts/python -m alembic upgrade head
../venv/Scripts/python -m app.cli seed-demo --limit 1000        # optional CIFAR-10 demo photos
cd ../frontend && npm install
```

## Run

```bash
docker compose up -d
cd backend && ../venv/Scripts/python -m uvicorn app.main:app --reload --port 8080
cd frontend && npm run dev     # http://localhost:5173
```

API docs: http://localhost:8080/docs

## Maintenance

| Command (from `backend/`) | Purpose |
|---|---|
| `python -m app.cli reindex` | Index photos saved while ChromaDB was unreachable |
| `python -m app.cli reindex --all` | Rebuild the whole index (e.g. after changing `CLIP_MODEL`) |
| `python -m app.cli seed-demo [--limit N]` | Load CIFAR-10 test images as demo data |

## Tests

```bash
cd backend && ../venv/Scripts/python -m pytest              # needs docker compose up
cd backend && ../venv/Scripts/python -m pytest -m slow      # real CLIP model
cd frontend && npm test
cd frontend && npm run e2e                                  # needs API running
```

## Configuration

See `.env.example`. Key settings: `SEARCH_IMG_WEIGHT` (0–1, weight of the visual match vs. your
metadata), `MAX_UPLOAD_MB`, `CLIP_MODEL`.

## Deploying

1. Build images: `docker build -t photo-api backend` and `docker build -t photo-web frontend`.
2. Provision S3 (two buckets), Postgres (e.g. RDS) and a ChromaDB server.
3. Run `photo-api` with env vars from `.env.example`, **unset** `S3_ENDPOINT_URL`, real AWS
   credentials (or an IAM role), `DATABASE_URL` pointing at Postgres, `CHROMA_HOST`/`CHROMA_PORT`,
   and `CORS_ORIGINS` set to your web origin. Migrations run on start.
4. Run `photo-web` with `API_UPSTREAM=http://<api-host>:8080`.
5. Index existing data on the new host if needed: `python -m app.cli reindex --all`.
````

- [ ] **Step 6: Run the full verification**

Run: `cd backend && ../venv/Scripts/python -m pytest -q && cd ../frontend && npm test && npm run typecheck && npm run build`
Expected: everything passes, and `vite build` produces `dist/`.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: add hosting Dockerfiles and README; remove legacy scripts"
```
