# Photo Retrieval

Search your photos by describing them ("red car at night"). CLIP matches the visual content;
your own tags, titles and descriptions are matched by keyword and blended into the ranking, so a
photo tagged `goa-trip` is found by "goa trip" even though CLIP can't see "Goa".

## Architecture

- **Docker (infra):** floci (S3 emulator, :4566), Postgres 16 (:5432), ChromaDB (:8000)
- **Native:** FastAPI backend (`backend/`, :8080, CLIP on GPU if available) and React dashboard (`frontend/`, :5173)
- Postgres is the source of truth; ChromaDB is a rebuildable index (`python -m app.cli reindex --all`).
- Ranking: `score = SEARCH_IMG_WEIGHT × visual similarity + (1 − SEARCH_IMG_WEIGHT) × metadata keyword match`.

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

> **There is no authentication.** Anyone who can reach the API can upload, edit and delete photos.
> Keep it on a private network (or behind an authenticating proxy) until auth is added.

1. Build images: `docker build -t photo-api backend` and `docker build -t photo-web frontend`.
2. Provision S3 (two buckets), Postgres (e.g. RDS) and a ChromaDB server.
3. Run `photo-api` with env vars from `.env.example`, **unset** `S3_ENDPOINT_URL`, real AWS
   credentials (or an IAM role), `DATABASE_URL` pointing at Postgres, `CHROMA_HOST`/`CHROMA_PORT`,
   and `CORS_ORIGINS` set to your web origin. Migrations run on start.
4. Run `photo-web` with `API_UPSTREAM=http://<api-host>:8080`.
5. Index existing data on the new host if needed: `python -m app.cli reindex --all`.
