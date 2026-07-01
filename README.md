# Chachkalica

A unified deployment of the two Chachkalica projects:

| Subfolder | What it is |
|-----------|------------|
| [`chachkalica/`](./chachkalica) | Django **annotator-fleet manager** — admin UI, annotation webhook receiver, RQ worker that provisions per-annotator Label Studio containers, and a `training/` app that drives the trainer over HTTP. |
| [`friendy_chachkalica/`](./friendy_chachkalica) | FastAPI + PyTorch **training service** — config-driven object-detection training/eval (RetinaNet, RT-DETR, RF-DETR, YOLOX). |

Each subproject keeps its own `Dockerfile` and `docker-compose.yml` and can still be
run on its own. The top-level `docker-compose.yml` here is the **one that unites
them** into a single stack on a shared network and shared data volume.

## Run the whole stack

```bash
cp .env.example .env      # optional — defaults work out of the box
docker compose up -d
```

This starts:

| Service    | Role                                   | Host port |
|------------|----------------------------------------|-----------|
| `web`      | Django admin + webhook (Gunicorn)      | 9000      |
| `worker`   | RQ background jobs (fleet provisioning)| —         |
| `trainer`  | friendy_chachkalica FastAPI trainer    | 8200      |
| `postgres` | Fleet state database                   | 5432      |
| `redis`    | RQ job queue                           | 6379      |

- Admin UI: <http://localhost:9000/admin/> ("Chachkalica Fleet")
- Trainer health: <http://localhost:8200/health>

`web` runs migrations and `collectstatic` on start, so first boot is ready to use.

## How the two halves connect

- **HTTP:** the Django `training/` app calls the trainer at `http://trainer:8200`
  (injected via `TRAINING_SERVICE_URL` in the compose file — no admin edit needed).
- **Shared filesystem:** Django writes training-config YAMLs with absolute
  `/app/data/...` paths; the trainer opens them as-is. All app containers therefore
  bind-mount `./chachkalica/data` at `/app/data`, so those paths resolve identically
  everywhere. See the note at the top of `docker-compose.yml`.

## GPU

The trainer runs CPU-only by default (fine for the synthetic smoke test). For real
training, uncomment the `deploy:` block on the `trainer` service in
`docker-compose.yml` (requires the NVIDIA Container Toolkit on the host).

## Standalone use

Run either half by itself from its own folder, e.g.:

```bash
cd chachkalica && docker compose up -d          # fleet manager only
cd friendy_chachkalica && docker compose up -d   # trainer only
```
