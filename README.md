# Photo Retrieval

A personal photo dashboard where you find photos by describing them ("red car at night",
"goa trip"). Browse everything in a grid, upload photos with tags, and edit titles, descriptions and
tags.

- **Visual search:** [CLIP](https://github.com/mlfoundations/open_clip) turns every photo and every
  search into vectors, and ChromaDB finds the closest photos.
- **Your metadata counts:** titles, descriptions and tags are matched by keyword and blended into
  the ranking, so a photo tagged `goa-trip` is found by "goa trip" even though CLIP can't see "Goa".
- **Tag filters:** chips in the search bar narrow results to photos that carry all selected tags.

To run it on a real server, see **[docs/deployment.md](docs/deployment.md)**.

---

## Contents

1. [How it fits together](#1-how-it-fits-together)
2. [Prerequisites](#2-prerequisites)
3. [First-time setup (fresh machine)](#3-first-time-setup-fresh-machine)
4. [Daily workflow](#4-daily-workflow)
5. [Using the dashboard](#5-using-the-dashboard)
6. [Maintenance commands](#6-maintenance-commands)
7. [Running the tests](#7-running-the-tests)
8. [Configuration reference](#8-configuration-reference)
9. [Project layout](#9-project-layout)
10. [How search ranking works](#10-how-search-ranking-works)
11. [Changing the database schema](#11-changing-the-database-schema)
12. [Troubleshooting](#12-troubleshooting)
13. [Resetting / wiping data](#13-resetting--wiping-data)

---

## 1. How it fits together

```
 Browser ── http://localhost:5173 ──▶ Vite dev server (frontend/, React)
    │                                    │ proxies /api/* to :8080
    │                                    ▼
    │                            FastAPI (backend/, CLIP on your GPU)       ◀── runs natively
    │                              │            │             │
    │  images via presigned URLs   ▼            ▼             ▼
    └────────────────────────▶ floci (S3)   Postgres 16    ChromaDB          ◀── run in Docker
                               :4566        :5432          :8000
```

| Piece | Role | Where its data lives |
|---|---|---|
| **Postgres** | Source of truth: photo records, titles, descriptions, tags | Docker volume `pg-data` |
| **floci** | Local AWS S3 emulator: original files and WebP thumbnails | Docker volume `floci-data` |
| **ChromaDB** | One CLIP vector per photo, for visual search. **Derived**: can be rebuilt from Postgres + S3 at any time | Docker volume `chroma-data` |
| **API** (`backend/`) | FastAPI. The only thing that writes to the three stores. Loads CLIP once at startup | — |
| **Dashboard** (`frontend/`) | React + Vite + TanStack Query + Tailwind | — |

Only the infrastructure runs in Docker. The API and dashboard run natively so CLIP can use your GPU
and both reload instantly while you edit code.

---

## 2. Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Docker Desktop | any recent | Must be **running** (whale icon in the tray) before `docker compose` |
| Python | 3.12+ | Developed on 3.14 |
| Node.js | 24+ | Includes `npm` |
| NVIDIA GPU + driver | optional | CLIP runs on CPU too, just slower (≈ 1 s/photo vs ≈ 20 ms) |
| Git | any | |

Disk: about 2 GB for Python packages (torch), 600 MB for CLIP weights, and 170 MB for the
optional CIFAR-10 demo set.

**Shell used in this guide:** Git Bash on Windows. Python lives at `venv/Scripts/python` on Windows
and at `venv/bin/python` on macOS/Linux. Set this once per terminal:

```bash
# Windows (Git Bash)
PY=../venv/Scripts/python
# macOS / Linux
PY=../venv/bin/python
```

All backend commands below run **from `backend/`** and use `$PY`.

---

## 3. First-time setup (fresh machine)

### 3.1 Clone and create the env file

```bash
git clone git@github.com:Ayush-Ghiya/photo-retrieval.git
cd photo-retrieval
cp .env.example .env          # local defaults work as-is
```

### 3.2 Start the infrastructure

```bash
docker compose up -d
docker compose ps             # wait until all three show "Up"; postgres shows "(healthy)"
```

The first start pulls images. **floci needs about 15 s to start listening.** Until then the API
reports `Startup check failed`.

On its very first start, Postgres also creates the `photos_test` database used by the test suite
(`docker/postgres/init.sql`).

### 3.3 Python environment

```bash
python -m venv venv
```

Install **torch first**, picking the build for your hardware:

```bash
# NVIDIA GPU (CUDA 12.8 build; required for RTX 50-series, fine for older cards)
venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# CPU only
venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

(On macOS/Linux use `venv/bin/python`. On a Mac, use plain `pip install torch torchvision`.)

Check that the GPU is visible (should print `True` on an NVIDIA machine):

```bash
venv/Scripts/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Then install the rest. `requirements.txt` deliberately leaves out torch so it never replaces your
GPU build:

```bash
cd backend
$PY -m pip install -r requirements-dev.txt
```

### 3.4 Create the database schema

```bash
$PY -m alembic upgrade head          # from backend/
```

### 3.5 (Optional) Load demo photos

This downloads CIFAR-10 (about 170 MB, into `data/`) and imports its 10,000 test images as demo
photos, tagged by class (`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`,
`ship`, `truck`):

```bash
$PY -m app.cli seed-demo --limit 1000    # quick: 1,000 photos (~1 min on GPU)
$PY -m app.cli seed-demo                 # all 10,000 (~10 min on GPU)
```

It is safe to re-run: photos already imported are skipped as duplicates. Demo images are 32×32
pixels, so they look blurry; that's expected.

### 3.6 Frontend dependencies

```bash
cd ../frontend
npm install
```

### 3.7 Start everything

Use two terminals:

```bash
# Terminal 1: API (from backend/)
$PY -m uvicorn app.main:app --reload --port 8080
```

```bash
# Terminal 2: dashboard (from frontend/)
npm run dev
```

Open **http://localhost:5173**. The interactive API docs are at http://localhost:8080/docs.

The **first API start downloads the CLIP weights** (about 600 MB, cached in
`~/.cache/huggingface`). Later starts take a few seconds.

Check that everything is connected:

```bash
curl -s localhost:8080/api/health
# {"status":"ok","checks":{"postgres":true,"s3":true,"chroma":true,"model":true}}
```

---

## 4. Daily workflow

```bash
docker compose up -d                                                     # repo root
cd backend  && $PY -m uvicorn app.main:app --reload --port 8080          # terminal 1
cd frontend && npm run dev                                               # terminal 2
```

To stop: `Ctrl+C` both terminals, then `docker compose stop` (or leave the containers running;
they're light). **`docker compose stop` / `down` keep your data. Only `down -v` deletes it.**

---

## 5. Using the dashboard

| Action | How |
|---|---|
| Search | Type a description in the search bar ("dog on a beach"). Results update as you type; the % badge on hover is the match score |
| Filter by tag | Type a tag in "Filter by tag" and press Enter. Several chips = photos must have **all** of them |
| Upload | **Upload** button → drag files in or click to choose. Tags, title and description apply to every file in the batch. Accepted: JPEG (incl. multi-picture phone JPEGs), PNG, WebP, HEIC; max 25 MB each |
| Edit | Click a photo → change title, description and tags → **Save**. Search reflects the change immediately |
| Delete | Click a photo → **Delete** → confirm. Removes it from S3, Postgres and the index |
| Demo images | "Show demo images" toggle. It starts on, and defaults to off once you've uploaded your own photos |
| Share a search | The URL holds the query and tags (`?q=red+car&tags=family`) |

**Tags** are normalised: lowercased, spaces become `-`, and only `a-z 0-9 -` is allowed
(max 40 characters). `Goa Trip` becomes `goa-trip`. Text typed in a tag box counts even if you
don't press Enter.

**Duplicates:** an identical file (same bytes) is detected and reported as "Duplicate"; it isn't
stored twice. Its existing tags are left unchanged.

---

## 6. Maintenance commands

Run from `backend/`:

| Command | When to use it |
|---|---|
| `$PY -m app.cli reindex` | Photos were saved while ChromaDB was down (the API says "Saved, but search indexing failed"), or a `reindex --all` was interrupted. Indexes only what's missing; safe to re-run |
| `$PY -m app.cli reindex --all` | You changed `CLIP_MODEL`, or the ChromaDB volume was lost. Rebuilds the whole index from S3. If interrupted, finish with plain `reindex` |
| `$PY -m app.cli seed-demo [--limit N]` | Import CIFAR-10 demo photos (§3.5) |

Rough timing for `reindex --all`: about 10–20 min per 10k photos on a GPU (photos are embedded one at a time). The running API keeps
working during a reindex (photos not yet re-indexed are temporarily missing from visual search).

---

## 7. Running the tests

The backend tests use the **real** Postgres, floci and ChromaDB from `docker compose`. They use a
separate database (`photos_test`), separate buckets (`test-photos-*`) and a separate collection
(`photos_test`), so your own photos are never touched.

```bash
docker compose up -d                       # must be running

cd backend
$PY -m pytest                              # ~100 tests, ~1 min; uses a fake CLIP
$PY -m pytest -m slow                      # also runs the real CLIP model (~10 s after weights are cached)

cd ../frontend
npm test                                   # Vitest component tests
npm run typecheck                          # TypeScript
npm run e2e                                # Playwright: needs the API running on :8080
                                           # (first time: npx playwright install chromium)
```

The e2e test uploads a uniquely tagged image, searches for it, and deletes it again.

---

## 8. Configuration reference

All settings come from environment variables, read from `.env` at the repo root (see
`.env.example`).

| Variable | Default (local) | Meaning |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg://photos:photos@localhost:5432/photos` | Postgres connection (SQLAlchemy URL) |
| `S3_ENDPOINT_URL` | `http://localhost:4566` | S3 endpoint the **API** uses. Empty = real AWS |
| `S3_PUBLIC_ENDPOINT_URL` | *(empty → same as above)* | Endpoint put into the image URLs given to the **browser**. Only differs when the browser reaches S3 at a different address than the API does |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | `test` / `test` | floci accepts anything. Real AWS: real keys, or omit to use an IAM role |
| `AWS_REGION` | `us-east-1` | Bucket region |
| `S3_BUCKET_ORIGINALS` / `S3_BUCKET_THUMBS` | `photos-originals` / `photos-thumbs` | Created automatically if missing |
| `CHROMA_HOST` / `CHROMA_PORT` | `localhost` / `8000` | ChromaDB server |
| `CLIP_MODEL` | `ViT-B/32` | Changing it requires `reindex --all` (the API refuses to start until you do) |
| `SEARCH_IMG_WEIGHT` | `0.7` | 0–1. Weight of visual similarity vs. your metadata in the ranking (§10) |
| `MAX_UPLOAD_MB` | `25` | Per-file upload limit |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated browser origins allowed to call the API directly |

Collection name (`photos_v1`), presigned URL lifetime (1 h), batch size (50 files) and
thumbnail size (512 px, WebP) are code defaults in `backend/app/config.py`.

---

## 9. Project layout

```
docker-compose.yml          Local infra: floci, postgres, chromadb
docker/postgres/init.sql    Creates the photos_test DB on first Postgres start
.env.example                Local config template (copy to .env)
deploy/                     Production compose file + env template (docs/deployment.md)
docs/
  deployment.md             How to run this on a real server
  superpowers/specs/        Design spec (incl. §10 search amendment)
  superpowers/plans/        Implementation plan
backend/
  app/main.py               FastAPI app factory; startup checks (lifespan)
  app/config.py             Settings (env vars)
  app/models.py, db.py      SQLAlchemy models (images, tags, image_tags)
  app/routers/              HTTP endpoints: images, search, tags, health
  app/services/
    imaging.py              Validate image, EXIF date/orientation, WebP thumbnail
    storage.py              S3 (boto3): put/get/delete/presign
    clip_model.py           CLIP encoder (loaded once)
    vector_index.py         ChromaDB: one vector per photo + tag/source filter metadata
    images.py               Upload / edit / delete / list orchestration
    text_match.py           Keyword matching of queries against metadata
    search.py               Blended ranking
    container.py            Wires services together; startup checks
  app/cli.py                reindex, seed-demo
  alembic/                  Database migrations
  tests/                    pytest (real infra, fake CLIP)
  Dockerfile                Production API image (CPU torch)
frontend/
  src/App.tsx               Page layout, demo toggle
  src/components/           Gallery, SearchBar, TagInput, DetailDrawer, UploadDialog, …
  src/api/                  Typed API client
  e2e/                      Playwright smoke test
  Dockerfile, nginx.conf.template   Production web image (static files + /api proxy)
```

---

## 10. How search ranking works

For a query `q` with optional tag filters:

1. **Visual candidates:** CLIP encodes `q`; ChromaDB returns the `limit × 3` most similar photos
   (respecting tag filters). Visual similarity `img = 1 − cosine_distance/2`, which in practice
   falls between about 0.55 and 0.70.
2. **Metadata candidates:** up to 500 photos whose title, description or tags contain a word
   starting with a query word (stopwords like "a", "the", "photo" are ignored; plurals fold:
   "cars" → "car"). Photos matching more words come first.
3. **Score:** `score = w · img + (1 − w) · coverage`, where `coverage` is the fraction of query
   words found in the photo's metadata (0–1) and `w = SEARCH_IMG_WEIGHT` (0.7).
4. An **empty query** just lists photos (newest first) with the tag filters applied.

Why keywords instead of CLIP for metadata: CLIP's text-to-text similarity is nearly identical for
any short phrase ("goa trip" scores higher against "tags: ship" than against a photo actually
tagged `goa-trip`). Details are in spec §10.

Tuning: raise `SEARCH_IMG_WEIGHT` toward 1.0 to rely more on what's in the picture, or lower it to
make your tags dominate.

---

## 11. Changing the database schema

1. Edit `backend/app/models.py`.
2. Generate a migration: `$PY -m alembic revision --autogenerate -m "describe change"`.
3. Review the file in `backend/alembic/versions/`.
4. Apply: `$PY -m alembic upgrade head`.
5. Check models and migrations agree: `$PY -m alembic check` → `No new upgrade operations detected.`

Tests build their schema from the models directly, so they don't need migrations. Production
runs `alembic upgrade head` automatically when the API container starts.

---

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `failed to connect to the docker API … dockerDesktopLinuxEngine` | Docker Desktop isn't running. Start it and wait for the whale icon to settle |
| API exits with `Startup check failed: … Is docker compose up -d running` | Infra isn't up, or floci is still starting (≈ 15 s). Check `docker compose ps`, wait, retry |
| API exits with `Collection 'photos_v1' was built with CLIP model …` | You changed `CLIP_MODEL`. Run `$PY -m app.cli reindex --all` |
| Red banner "Backend services unavailable" in the dashboard | The API isn't running on :8080, or `/api/health` reports a failing dependency. Run `curl localhost:8080/api/health` |
| Some photos never show up in visual search | They aren't indexed (an upload during a Chroma outage, or an interrupted reindex). Run `$PY -m app.cli reindex` |
| Tests fail with `database "photos_test" does not exist` | The Postgres volume existed before `init.sql` was added. Run: `docker compose exec postgres psql -U photos -c "CREATE DATABASE photos_test OWNER photos;"` |
| `port is already allocated` on 5432 / 8000 / 4566 | Another service uses the port (e.g. a locally installed Postgres). Change the left side of the port mapping in `docker-compose.yml` (e.g. `"5433:5432"`) and the matching value in `.env` |
| Upload says "Unsupported image type" | Only JPEG, PNG, WebP and HEIC are accepted. Convert GIF/TIFF/BMP first |
| Upload says "Image is too large" | Over Pillow's ~179-megapixel safety limit |
| First API start hangs for a while | It's downloading the CLIP weights (≈ 600 MB). Later starts are fast |
| `torch.cuda.is_available()` is `False` on an NVIDIA machine | You installed the CPU build. Reinstall torch with the `cu128` index URL (§3.3) |
| `LF will be replaced by CRLF` warnings on commit | Harmless on Windows |

Logs: the API logs to its terminal. Infra logs: `docker compose logs -f floci` (or `postgres`,
`chromadb`).

---

## 13. Resetting / wiping data

```bash
# Rebuild only the search index (safe; nothing is lost)
cd backend && $PY -m app.cli reindex --all

# DANGER: delete ALL photos, tags and the index (Docker volumes)
docker compose down -v
docker compose up -d
cd backend && $PY -m alembic upgrade head
```

Inspecting data directly:

```bash
docker compose exec postgres psql -U photos -d photos -c "select filename, title, indexed from images limit 10;"
```
