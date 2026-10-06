# Photo Retrieval Dashboard — Design Spec

**Date:** 2026-10-06
**Status:** Approved in brainstorming, pending written-spec review
**Branch:** `phase-1`

## 1. Goal

Turn the existing CLIP + ChromaDB proof of concept into a complete, single-user photo
dashboard where the user can:

- browse all their images in a grid,
- upload images (single or batch) with optional tags / title / description,
- edit an image's title, description and tags, and delete images,
- search with a free-text prompt (e.g. "red car"), where **user metadata (tags, title,
  description) participates in ranking**, and optionally narrow by tag filters.

The project is built for real personal photos (full-resolution JPEG/PNG/WebP/HEIC), and ships
with the CIFAR-10 test set as optional demo data.

Not hosted now, but structured so hosting later is "build two images, set env vars".

### Non-goals (YAGNI)

Authentication / multi-user, pixel editing (crop/filters), albums, video, background job
queues, face recognition.

## 2. Current state (starting point)

| File | Role | Fate |
|---|---|---|
| `build_dataset.py` | Dumps CIFAR-10 test set to PNGs | Replaced by `cli seed-demo` |
| `build_index.py` | CLIP-encodes a folder into ChromaDB | Replaced by `cli reindex` |
| `search.py` | Text → CLIP → Chroma top-k | Logic moves to `backend/app/services/search.py` |
| `utils.py` | CLIP loader, Chroma client | Moves to `backend/app/services/` |
| `app.py` | Flask `/search`, `/images/<file>` | Replaced by FastAPI app |
| `clip_image_search.py` | CLI contact sheet | Dropped (dashboard supersedes it) |
| `search_LEGACY.py` | Dead code | Deleted |

Known issues fixed by this work: CLIP model reloaded per request; image IDs are local file
paths; Chroma not provisioned anywhere; no `requirements.txt` / README.

## 3. Architecture

```
┌──────────── docker compose (infra only) ─────────────┐
│   floci (S3) :4566    postgres :5432   chromadb :8000 │
└───────▲──────────────────▲─────────────────▲─────────┘
        │                  │                 │
        │      api (FastAPI + CLIP, native venv, GPU) :8080
        │                  ▲
        │ presigned GETs   │ JSON /api
        └──────── web (React/Vite dev server, native) :5173
```

### Local runtime split (user requirement)

- **Docker (`docker compose up -d`)**: floci, Postgres, ChromaDB — each with a named volume
  so data survives restarts.
- **Native**: FastAPI app from the project venv (CLIP on the local RTX GPU, CPU fallback),
  and the React app via `npm run dev` (Vite proxies `/api` to the API).
- **Hosting-ready extras (not used locally)**: `backend/Dockerfile` (CPU torch by default)
  and `frontend/Dockerfile` (multi-stage build → nginx serving static files and proxying
  `/api`). README has a "Deploying" section.

### Components

- **floci** (`floci/floci`, MIT, AWS emulator) — S3 only. `FLOCI_STORAGE_MODE=persistent`,
  `FLOCI_STORAGE_PERSISTENT_PATH` mounted on a named volume. Buckets `photos-originals` and
  `photos-thumbs`, created idempotently by the API at startup. Accessed with `boto3`,
  path-style addressing. Browser reads images via short-lived pre-signed GET URLs; the API
  never streams image bytes.
- **Postgres 16** — source of truth for image records and tags.
- **ChromaDB** (server image, pinned to a version compatible with the Python client) —
  derived vector index; always rebuildable from Postgres + S3.
- **api** — FastAPI, SQLAlchemy 2.0, Alembic, Pydantic settings. Loads CLIP once at startup
  (lifespan). The only writer to S3, Postgres and Chroma.
- **web** — React + TypeScript + Vite, TanStack Query, Tailwind CSS.

### Configuration

All via env vars (`.env`, with `.env.example` committed):

| Var | Example (local) |
|---|---|
| `DATABASE_URL` | `postgresql+psycopg://photos:photos@localhost:5432/photos` |
| `S3_ENDPOINT_URL` | `http://localhost:4566` (unset for real AWS) |
| `S3_PUBLIC_ENDPOINT_URL` | endpoint used when signing browser URLs (defaults to `S3_ENDPOINT_URL`) |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION` | `test` / `test` / `us-east-1` |
| `S3_BUCKET_ORIGINALS` / `S3_BUCKET_THUMBS` | `photos-originals` / `photos-thumbs` |
| `CHROMA_HOST` / `CHROMA_PORT` | `localhost` / `8000` |
| `CLIP_MODEL` | `ViT-B/32` |
| `SEARCH_IMG_WEIGHT` | `0.7` (text weight = 1 − this) |
| `MAX_UPLOAD_MB` | `25` |
| `CORS_ORIGINS` | `http://localhost:5173` |

Moving to AWS = unset `S3_ENDPOINT_URL`, point `DATABASE_URL` at RDS, `CHROMA_HOST` at a
hosted Chroma. No code changes.

### Repository layout

```
docker-compose.yml          # floci, postgres, chromadb
.env.example
README.md
backend/
  pyproject.toml / requirements.txt
  Dockerfile
  alembic/ , alembic.ini
  app/
    main.py                 # FastAPI app, lifespan, routers, error handlers
    config.py               # pydantic-settings
    db.py, models.py        # SQLAlchemy
    schemas.py              # Pydantic request/response models
    routers/ images.py, search.py, tags.py, health.py
    services/
      clip_model.py         # load once; encode_images, encode_text
      storage.py            # S3: put, delete, presign, ensure buckets
      vector_index.py       # Chroma: upsert/delete/query, metadata mapping
      images.py             # upload pipeline, edit, delete orchestration
      search.py             # blended scoring
      imaging.py            # EXIF, orientation, thumbnail, hashing
    cli.py                  # reindex, seed-demo
  tests/
frontend/
  package.json, vite.config.ts, Dockerfile, nginx.conf
  src/ api/, components/, pages/
```

## 4. Data model

### Postgres

```
images
  id            uuid PK
  s3_key        text not null            -- originals/<id>.<ext>
  thumb_key     text not null            -- thumbs/<id>.webp (longest side 512px)
  filename      text not null            -- original upload name
  mime_type     text not null
  title         text null
  description   text null
  width         int not null
  height        int not null
  content_hash  text not null unique     -- sha256 of original bytes
  taken_at      timestamptz null         -- EXIF DateTimeOriginal
  source        text not null            -- 'upload' | 'demo'
  indexed       bool not null default false
  created_at    timestamptz not null default now()
  updated_at    timestamptz not null default now()

tags        id serial PK, name text unique   -- trimmed, lowercased, spaces → '-'
image_tags  image_id uuid FK → images ON DELETE CASCADE,
            tag_id int FK → tags ON DELETE CASCADE,
            PK (image_id, tag_id)
```

Tags with zero images are left in place (cheap) but excluded from `/tags` output.
Tag names: 1–40 chars after normalisation, `[a-z0-9-]` only.

### ChromaDB

Collection `photos_v1`, cosine space, collection metadata records `clip_model`.
Up to two vectors per image:

| id | vector | metadata |
|---|---|---|
| `<uuid>:img` | CLIP image embedding (L2-normalised) | `image_id`, `kind="img"`, `source`, `tag_<name>=true` per tag |
| `<uuid>:txt` | CLIP text embedding of `"{title}. {description}. tags: {t1}, {t2}"` (empty parts omitted; truncated to CLIP's 77-token limit) | same, `kind="txt"` |

`:txt` exists only if the image has a title, description or at least one tag. Tag filters
use `where={"$and": [{"tag_family": True}, ...]}` (all selected tags must match).

If `CLIP_MODEL` differs from the collection's recorded model, the API refuses to start and
tells the user to run `cli reindex --all`.

## 5. Flows

### Upload — `POST /api/images` (multipart)

Fields: `files[]` (1–50), optional `tags` (comma-separated), `title`, `description` —
applied to every file in the batch. Per file, synchronously:

1. Validate MIME (sniffed via Pillow, not trusted from the client): JPEG, PNG, WebP, HEIC
   (via `pillow-heif`). Reject > `MAX_UPLOAD_MB`.
2. sha256 → if `content_hash` exists, result `duplicate` with the existing id.
3. Read EXIF `DateTimeOriginal`; apply EXIF orientation; produce WebP thumbnail.
4. PUT original + thumb to S3.
5. Insert row + tags (single transaction).
6. CLIP-encode image; upsert `:img` (and `:txt` if metadata) to Chroma; set `indexed=true`.

Failure handling: failure in 1–5 → that file is `error`, any objects already written to S3
for it are deleted. Failure in 6 → row kept with `indexed=false`, result `created` with a
warning; `cli reindex` catches it up. Response: `[{filename, status, id?, message?}]`, HTTP
200 even when individual files fail (per-file status), 400 only for malformed requests.

### Edit — `PATCH /api/images/{id}`

Body: any of `title`, `description`, `tags` (full replacement list). Update Postgres, then
re-embed `:txt` (or delete it if no text metadata remains) and rewrite tag metadata on both
vectors. If the Chroma step fails, set `indexed=false` and return 200 with a warning.

### Delete — `DELETE /api/images/{id}`

Delete Chroma vectors → S3 objects → Postgres row. Missing pieces are ignored (idempotent).

### Search — `GET /api/search?q=&tags=&source=&limit=`

- `limit` default 40, max 200.
- If `q` is non-empty:
  1. Encode `q` with CLIP text encoder.
  2. Chroma query, `n_results = limit × 3`, `where` from tags/source.
  3. Convert distance → similarity `s = 1 − d/2`. Group by `image_id`:
     `score = w·img_sim + (1−w)·txt_sim`; if `txt_sim` missing, `score = img_sim`;
     if only `txt_sim` returned, fetch that image's `:img` similarity via
     `collection.get(..., include=["embeddings"])` and dot product.
  4. Sort desc, take `limit`, load rows from Postgres, attach presigned thumb URLs.
- If `q` is empty: behaves like `GET /api/images` with the same filters (date order).

### Listing — `GET /api/images?page=&page_size=&tags=&source=&sort=`

`page_size` default 60, max 200. `sort`: `taken` (coalesce(taken_at, created_at) desc,
default) or `uploaded`. Response `{items, page, page_size, total}`; each item has
`id, title, filename, tags, width, height, thumb_url, taken_at, source`.

### Other endpoints

- `GET /api/images/{id}` — full record + `original_url` + `thumb_url`.
- `GET /api/tags` — `[{name, count}]`, count > 0, sorted by count desc.
- `GET /api/health` — status of S3, Postgres, Chroma, model; 503 if any is down.

Presigned URL TTL: 1 hour.

### CLI — `python -m app.cli`

- `reindex [--all]` — re-embed images with `indexed=false` (or all, recreating the
  collection with `--all`), reading originals from S3.
- `seed-demo [--limit N]` — download CIFAR-10 test set via torchvision, ingest through the
  same upload service with `source='demo'` and the class name as tag. Idempotent via hash.

## 6. Dashboard (frontend)

Single page, no router needed beyond query-string state (`?q=&tags=`).

- **Header**: large search input (submit on Enter, debounced), tag-filter chips with
  autocomplete from `/api/tags`, "Upload" button, "Show demo images" toggle (default on
  until the user has ≥1 own upload, then off).
- **Gallery**: responsive CSS grid of thumbnails, infinite scroll (TanStack
  `useInfiniteQuery`). In search mode: results in rank order, score badge on hover, a
  "Clear search" action. Empty states for "no images yet" (with upload CTA) and "no matches".
- **Detail drawer** (click a thumbnail): large preview (original URL), metadata (filename,
  dimensions, taken date), editable title, description, tag chip input with autocomplete;
  Save (PATCH, invalidates list/search queries) and Delete (confirm dialog).
- **Upload dialog**: drag-and-drop + file picker, previews, shared tags/title/description,
  per-file progress (XHR upload progress), result summary listing created / duplicate /
  error per file.
- **Errors**: API error envelope surfaced as toasts; `/api/health` failure shows a banner
  "Backend services unavailable — is `docker compose up` running?".

## 7. Error handling

- API error envelope: `{"error": {"code": "not_found", "message": "..."}}` with correct HTTP
  status (400 validation, 404, 413 too large, 415 unsupported type, 503 dependency down).
- Startup: lifespan checks Postgres, S3 (ensures buckets), Chroma and model consistency;
  fails fast with an actionable message.
- Consistency: Postgres is authoritative. Chroma drift is repaired with `cli reindex`.
  S3 orphans from a crash between steps are tolerated (no GC in scope).

## 8. Testing

- **Backend (pytest)**: runs against the real compose services using a separate database
  (`photos_test`), bucket prefix and Chroma collection (`photos_test`). CLIP replaced by a
  deterministic fake encoder via dependency override in most tests; one `@pytest.mark.slow`
  test exercises the real model end-to-end ("red car" ranks a red-car image above a cat).
  Coverage targets: upload pipeline (incl. duplicate, bad type, oversize), edit re-indexing,
  delete idempotency, blended scoring math, tag filtering, listing pagination.
- **Frontend**: Vitest + Testing Library for the tag input, upload dialog and drawer; one
  Playwright smoke test: upload → tag → search finds it.

## 9. Deliverables checklist

- `docker-compose.yml` (floci, postgres, chromadb, volumes, healthchecks)
- Backend app, Alembic initial migration, CLI, tests, `requirements.txt`, `Dockerfile`
- Frontend app, tests, `Dockerfile` + `nginx.conf`
- `.env.example`, updated `.gitignore`, `README.md` (setup, run, seed, test, deploy)
- Old top-level scripts removed after their logic is ported
