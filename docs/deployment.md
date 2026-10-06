# Deploying Photo Retrieval

This guide takes the app from your laptop to a real server, with photos in **real AWS S3**.

> ⚠️ **There is no login.** Anyone who can reach the site can view, upload, edit and **delete**
> every photo. Never put it on the open internet as-is. Use one of the protections in
> [§6](#6-restrict-access-required).

**Contents**

1. [What runs where](#1-what-runs-where)
2. [Pick a host](#2-pick-a-host)
3. [AWS setup: S3 buckets and permissions](#3-aws-setup-s3-buckets-and-permissions)
4. [Server setup](#4-server-setup)
5. [Deploy with docker compose](#5-deploy-with-docker-compose)
6. [Restrict access (required)](#6-restrict-access-required)
7. [HTTPS and a domain](#7-https-and-a-domain)
8. [Moving your local photos to the server](#8-moving-your-local-photos-to-the-server)
9. [Updating to a new version](#9-updating-to-a-new-version)
10. [Backups and restore](#10-backups-and-restore)
11. [Operations cheat-sheet](#11-operations-cheat-sheet)
12. [Alternative: managed services (RDS, ECS)](#12-alternative-managed-services-rds-ecs)
13. [Pre-launch checklist](#13-pre-launch-checklist)

---

## 1. What runs where

The recommended setup is **one Linux VM running Docker**, plus S3 for the photo files:

```
             Internet / VPN
                   │  :443 (Caddy / ALB / Cloudflare Tunnel, §6–7)
                   ▼
  ┌──────────────── VM: docker compose (deploy/docker-compose.prod.yml) ──────────┐
  │  web  (nginx: dashboard + /api proxy)  ── :80 ──▶  api (FastAPI + CLIP, CPU)    │
  │                                                    │            │              │
  │                                              postgres 16    chromadb           │
  │                                              (volume)       (volume)           │
  └─────────────────────────────────────────────────────┼──────────────────────────┘
                                                        ▼
                                              AWS S3: originals + thumbs
                  browser loads images directly from S3 via presigned URLs (1 h)
```

| Component | Production choice | Why |
|---|---|---|
| Photo files | **AWS S3** (2 private buckets) | Durable, cheap; the API code is the same as locally |
| Database | Postgres container on the VM (or RDS, §12) | Small data; one VM is simplest |
| Vector index | ChromaDB container on the VM | There is no managed Chroma on AWS. It can be rebuilt from Postgres + S3 at any time |
| API | `backend/Dockerfile` image (CPU torch) | CLIP on CPU: about 0.3–1 s per uploaded photo, about 50 ms per search |
| Web | `frontend/Dockerfile` image (nginx) | Serves the built dashboard and proxies `/api` to the API |

What's in the repo for this:

- `deploy/docker-compose.prod.yml`: the stack above. It has been tested end to end: build, start,
  migrate, upload, search, presigned image fetch, and `reindex` inside the container.
- `deploy/.env.prod.example`: the template for the server's config.
- `backend/Dockerfile`: runs `alembic upgrade head` on every start, then the API on :8080.
- `frontend/Dockerfile` + `nginx.conf.template`: builds the dashboard; `API_UPSTREAM` sets where
  `/api` goes.

---

## 2. Pick a host

Minimum: **2 vCPU, 4 GB RAM, 30 GB disk**. CLIP plus torch uses about 1.5 GB of RAM. Recommended:
4 GB+ RAM.

| Option | Example | Approx. cost/month |
|---|---|---|
| AWS EC2 | `t3.medium` (2 vCPU / 4 GB), 30 GB gp3 | ~$30–35 + S3 |
| Lightsail | 4 GB plan | ~$24 + S3 |
| Any VPS (Hetzner, DigitalOcean…) | 2–4 vCPU / 4–8 GB | $7–25 + S3 |
| Home server / spare PC | anything with Docker | free + S3 |

S3 cost for a personal library: about $0.023/GB-month. 50 GB of photos is roughly $1.15/month,
plus small request and transfer costs.

Put the VM in **the same AWS region as the buckets** if it's on AWS.

---

## 3. AWS setup: S3 buckets and permissions

Use the AWS Console or the AWS CLI. The examples use `ap-south-1` (Mumbai); pick your region and
**globally unique** bucket names.

### 3.1 Create two private buckets

```bash
REGION=ap-south-1
aws s3api create-bucket --bucket yourname-photos-originals --region $REGION \
  --create-bucket-configuration LocationConstraint=$REGION
aws s3api create-bucket --bucket yourname-photos-thumbs --region $REGION \
  --create-bucket-configuration LocationConstraint=$REGION
```

(For `us-east-1`, leave out `--create-bucket-configuration`.)

Keep **Block Public Access ON** (the default). The browser gets images through presigned URLs,
so the buckets never need to be public, and **no CORS configuration is needed** because images
load through `<img>` tags.

Recommended: turn on versioning for the originals bucket, so an accidental delete is recoverable:

```bash
aws s3api put-bucket-versioning --bucket yourname-photos-originals \
  --versioning-configuration Status=Enabled
```

(The API also creates the buckets on startup if they're missing. Creating them yourself lets you
choose the settings.)

### 3.2 Give the API permission to use them

Create an IAM policy, for example `photo-retrieval-s3`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
      "Resource": [
        "arn:aws:s3:::yourname-photos-originals",
        "arn:aws:s3:::yourname-photos-thumbs"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": [
        "arn:aws:s3:::yourname-photos-originals/*",
        "arn:aws:s3:::yourname-photos-thumbs/*"
      ]
    }
  ]
}
```

Then attach it in **one** of these ways:

- **EC2 (preferred):** create an IAM **role** for EC2 with this policy and attach it to the
  instance. You then put **no keys** in the env file; boto3 picks up the role automatically.
- **Any other host:** create an IAM **user** with only this policy, create an access key, and put
  `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` in `deploy/.env.prod`.

---

## 4. Server setup

On a fresh Ubuntu 24.04 VM:

```bash
# Docker Engine + compose plugin
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker

# The code
git clone https://github.com/Ayush-Ghiya/photo-retrieval.git
cd photo-retrieval
```

Firewall: allow **22** (SSH) and **443** (and 80 if using Caddy for HTTPS). Postgres, Chroma and
the API are **not** published by the compose file. Only `web` is (on `WEB_PORT`).

---

## 5. Deploy with docker compose

### 5.1 Configure

```bash
cp deploy/.env.prod.example deploy/.env.prod
nano deploy/.env.prod
```

| Variable | What to set |
|---|---|
| `POSTGRES_PASSWORD` | A long random string: `openssl rand -hex 24`. Use letters and digits only (it's placed inside the database URL) |
| `S3_ENDPOINT_URL`, `S3_PUBLIC_ENDPOINT_URL` | **Leave empty** for real AWS |
| `AWS_REGION` | Your bucket region, e.g. `ap-south-1` |
| `S3_BUCKET_ORIGINALS`, `S3_BUCKET_THUMBS` | Your bucket names from §3 |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Only if not using an EC2 role. With a role, **delete both lines** |
| `CLIP_MODEL` | Keep `ViT-B/32` unless you know you want another (bigger models need more RAM) |
| `SEARCH_IMG_WEIGHT` | `0.7` (see README §10) |
| `CORS_ORIGINS` | Your site's URL, e.g. `https://photos.example.com` |
| `WEB_PORT` | `80`, or e.g. `8080` if Caddy/another proxy takes 80/443 on the same machine |

`deploy/.env.prod` is git-ignored. Never commit it.

### 5.2 Build and start

```bash
docker compose -f deploy/docker-compose.prod.yml --env-file deploy/.env.prod up -d --build
```

The first build takes 5–10 minutes (torch is large, and the API image is about 1.8 GB). The first
API start downloads the CLIP weights (about 600 MB) into the `model-cache` volume, so later
restarts are quick.

Tip: define an alias so you don't have to type that every time:

```bash
alias dc='docker compose -f deploy/docker-compose.prod.yml --env-file deploy/.env.prod'
```

(The rest of this guide uses `dc`.)

### 5.3 Verify

```bash
dc ps                                   # all 4 services Up
dc logs -f api                          # wait for "Uvicorn running on http://0.0.0.0:8080"
curl -s localhost:${WEB_PORT:-80}/api/health
# {"status":"ok","checks":{"postgres":true,"s3":true,"chroma":true,"model":true}}
```

If `"s3": false`, the bucket names, region or credentials/role are wrong. `dc logs api` shows
the boto error.

Then open the site, upload a photo and search for it. If the thumbnail appears, presigned URLs to
S3 work.

---

## 6. Restrict access (required)

Pick one:

| Option | Effort | Notes |
|---|---|---|
| **Tailscale** (recommended for personal use) | 10 min | Install Tailscale on the VM and on your devices. Don't open 80/443 publicly; browse to `http://<vm-name>:80` over the tailnet. Free for personal use |
| **Cloudflare Tunnel + Access** | 20 min | No open ports at all. Cloudflare Access puts an email/OTP login in front. Free tier is enough |
| **Caddy with basic auth** | 10 min | Simple username/password at the proxy (config in §7). Fine for one user; use a strong password |
| **AWS security group** | 2 min | Allow 80/443 only from your home IP. Breaks when your IP changes |

Even with these, keep S3 Block Public Access on. Presigned URLs expire after 1 hour, so leaked
image links stop working.

---

## 7. HTTPS and a domain

Skip this if you use Tailscale or Cloudflare Tunnel (both give you HTTPS or private access).

With your own domain: point an `A` record (e.g. `photos.example.com`) at the VM's public IP, set
`WEB_PORT=8080` in `deploy/.env.prod`, restart (`dc up -d`), and run **Caddy** on the host for
automatic Let's Encrypt certificates:

```bash
sudo apt install -y caddy
sudo tee /etc/caddy/Caddyfile >/dev/null <<'EOF'
photos.example.com {
    # Optional login — generate the hash with:  caddy hash-password
    basic_auth {
        ayush <paste-bcrypt-hash-here>
    }
    request_body {
        max_size 30MB
    }
    reverse_proxy localhost:8080
}
EOF
sudo systemctl reload caddy
```

Open ports 80 and 443 in the firewall or security group (Caddy needs 80 for the certificate
challenge). Set `CORS_ORIGINS=https://photos.example.com`.

---

## 8. Moving your local photos to the server

Your local photos live in three places: floci (files), Postgres (records and tags) and Chroma (the
index). Move the first two; **rebuild** the index on the server.

**1. Copy files from floci to S3.** Run on your laptop with infra up and AWS credentials
configured:

```bash
mkdir -p /tmp/photos-export
aws s3 sync s3://photos-originals /tmp/photos-export/originals --endpoint-url http://localhost:4566
aws s3 sync s3://photos-thumbs    /tmp/photos-export/thumbs    --endpoint-url http://localhost:4566
aws s3 sync /tmp/photos-export/originals s3://yourname-photos-originals
aws s3 sync /tmp/photos-export/thumbs    s3://yourname-photos-thumbs
```

(Object keys stay the same, e.g. `originals/<uuid>.jpg`, so the database rows still point at the
right files.)

**2. Copy the database.**

```bash
# laptop
docker compose exec -T postgres pg_dump -U photos -d photos --clean --if-exists > photos.sql
scp photos.sql you@server:~/photo-retrieval/

# server (stack running)
dc exec -T postgres psql -U photos -d photos < photos.sql
```

**3. Rebuild the vector index on the server.**

```bash
dc exec api python -m app.cli reindex --all
```

On CPU this takes about 0.3–1 s per photo (10k photos ≈ 1–3 hours). Run it in `tmux` or `screen`.
If it gets interrupted, `dc exec api python -m app.cli reindex` continues where it stopped.

If you want demo photos on the server instead, run
`dc exec api python -m app.cli seed-demo --limit 1000 --data-dir /tmp/cifar` (it downloads
CIFAR-10 into the container).

---

## 9. Updating to a new version

```bash
cd ~/photo-retrieval
git pull
dc up -d --build          # rebuilds images; the API runs new migrations on start
dc logs -f api            # check it comes up healthy
```

If the update changes `CLIP_MODEL`, the API refuses to start until you rebuild the index. Run
`dc run --rm api python -m app.cli reindex --all`, then `dc up -d`.

---

## 10. Backups and restore

| Data | Backed up how | Restore |
|---|---|---|
| Photo files (S3) | Durable by design; versioning (§3.1) protects against deletes. Optional: S3 replication to another region | — |
| Postgres (titles, tags, which file is which) | **You must back this up.** Nightly `pg_dump` to S3 (below) | `psql < dump.sql` |
| ChromaDB index | Not needed: rebuild with `reindex --all` | `reindex --all` |
| CLIP weights | Not needed: re-downloaded automatically | — |

Nightly database backup to the originals bucket (crontab on the server, `crontab -e`):

```cron
15 3 * * * cd ~/photo-retrieval && docker compose -f deploy/docker-compose.prod.yml --env-file deploy/.env.prod exec -T postgres pg_dump -U photos -d photos | gzip | aws s3 cp - s3://yourname-photos-originals/backups/photos-$(date +\%F).sql.gz
```

(This needs the AWS CLI on the host, plus `s3:PutObject` on `backups/*`, which the §3.2 policy
already allows.)

Restore drill (do it once so you know it works):

```bash
aws s3 cp s3://yourname-photos-originals/backups/photos-2026-10-06.sql.gz - | gunzip \
  | dc exec -T postgres psql -U photos -d photos
dc exec api python -m app.cli reindex --all
```

---

## 11. Operations cheat-sheet

| Task | Command |
|---|---|
| Status | `dc ps` |
| Logs | `dc logs -f api` (or `web`, `postgres`, `chromadb`) |
| Health | `curl -s localhost:${WEB_PORT:-80}/api/health` |
| Restart the API | `dc restart api` |
| Index missed photos | `dc exec api python -m app.cli reindex` |
| Rebuild the whole index | `dc exec api python -m app.cli reindex --all` |
| SQL shell | `dc exec postgres psql -U photos -d photos` |
| Stop everything (keeps data) | `dc down` |
| ⚠️ Delete all server data (DB + index; S3 untouched) | `dc down -v` |

---

## 12. Alternative: managed services (RDS, ECS)

If you outgrow one VM, or want AWS to run the database:

- **Postgres → RDS** (PostgreSQL 16, `db.t4g.micro` is plenty). Remove the `postgres` service
  and the `DATABASE_URL` override from the compose file and set
  `DATABASE_URL=postgresql+psycopg://USER:PASS@<rds-endpoint>:5432/photos?sslmode=require` in
  `deploy/.env.prod`. Allow the VM's security group in the RDS security group.
- **API/web → ECS Fargate:** push both images to ECR. The API task needs 1 vCPU / 3 GB, env vars
  from §5.1 (secrets in SSM Parameter Store), and an EFS volume at `/root/.cache` (or accept the
  weight download on each new task). Put an ALB in front of `web`, or serve the `frontend` build
  from S3 + CloudFront with `/api/*` routed to the API.
- **ChromaDB** must still run somewhere with persistent disk (an ECS service with EFS, or a small
  EC2). It's only an index, so losing it costs a `reindex --all`, not data.
- **GPU** is optional. For faster bulk reindexing, temporarily run the CLI on a GPU instance
  (change the torch index URL in `backend/Dockerfile` to `cu128`).

---

## 13. Pre-launch checklist

- [ ] Buckets are private (Block Public Access on); versioning on for originals
- [ ] IAM: least-privilege policy (§3.2); EC2 role instead of keys where possible
- [ ] `deploy/.env.prod`: strong `POSTGRES_PASSWORD`, S3 endpoints **empty**, correct region and bucket names
- [ ] Access is restricted (§6). The site is **not** openly reachable
- [ ] HTTPS in place (Caddy, Cloudflare or Tailscale)
- [ ] `/api/health` returns `"status":"ok"`; an upload shows its thumbnail
- [ ] Nightly `pg_dump` to S3 is scheduled, and a restore was tested once
- [ ] Local photos migrated (§8) and `reindex --all` has finished
