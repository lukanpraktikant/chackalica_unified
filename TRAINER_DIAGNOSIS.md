# Trainer diagnosis & fix handoff

Context for a fresh agent: this is the unified Chachkalica stack (Django fleet/training app in
`chachkalica/`, FastAPI trainer service in `friendy_chachkalica/`, run via `docker-compose.yml`).
A training run (`TrainingRun` pk=2, experiment `PPE_v0.1`) showed as **running forever** in the
Django admin, and its **validation loss blew up**. Diagnosis below is confirmed from logs, the DB
row, and the trainer service. There are **two independent problems**. Fix both.

Paths below are relative to the repo root unless they start with `/app` (that's the path *inside*
the trainer container; on the host it maps to `chachkalica/data/...`).

---

## Problem 1 — Run stuck at "running" (orchestration bug, NOT a training crash)

### What actually happened
- The training **finished successfully** (~6h). Trainer service reports `GET /runs/2` →
  `status:"ok"`, `returncode:0`; `run_summary.yaml`, `metrics.csv`, `results.yaml` were written
  (`chachkalica/data/training/runs/PPE_v0.1-2/`). `/health` → `active:0`.
- The **RQ worker job** that supervises the run was killed by RQ's **900s (15-min) job timeout**:
  ```
  15:00:26  default: training.jobs.run_training(2)
  15:15:26  job ...: JobTimeoutException: Task exceeded maximum timeout value (900 seconds)
  ```
- Root cause chain:
  - `fleetsite/settings.py:122` → `RQ_QUEUES["default"]["DEFAULT_TIMEOUT"] = 900`.
  - `training/admin.py` enqueues with **no `job_timeout`** override
    (e.g. `training/admin.py:123`, `:225`; eval equivalents `:346`, `:430`).
  - `training/jobs.py:run_training` launches the trainer as a **decoupled subprocess** (in the
    trainer container) then sits in a `while waited < MAX_WAIT` (48h) poll loop
    (`training/jobs.py:42-61`). The poll loop is **not** wrapped in try/except (only
    `runner.launch` is). So at 15 min `JobTimeoutException` propagates straight out and
    `_mark(run, OK/ERROR)` is never called.
- Consequences: DB row frozen at `status='running'`, `finished_at=None`, `last_error=''`
  (confirmed). And because the job died before the `ingest.ingest_run()` step, **results were
  never ingested into Django** (`RunResult`/`TrainedModel` rows missing) and auto-eval never
  scheduled. This is **systemic** — run 1 died the identical way (13:56→14:11, same 900s timeout).

### Fixes (in `chachkalica/`)
1. **Set a job timeout that exceeds the poll budget.** At every `run_training` enqueue in
   `training/admin.py`, pass `job_timeout` ≥ `jobs.MAX_WAIT` (48h), e.g.
   `_queue().enqueue(jobs.run_training, run.pk, job_timeout=jobs.MAX_WAIT + 3600)`.
   Do the same for the `run_eval` enqueues. (Keep the queue `DEFAULT_TIMEOUT` for other short jobs,
   or raise it — but per-job override is safer.)
2. **Guard the poll loop so a killed/timed-out job still marks the row.** In
   `training/jobs.py:run_training` (and `run_eval`), wrap the polling loop in
   `try/except Exception` and on exception `_mark(run, ERROR, error=..., finished=True)` then
   re-raise. This prevents future silent "stuck running" rows.
3. **Add a reconcile/heal path** for rows already stuck (and for any future orphan): a management
   command or admin action that, for a `running` row, calls `runner.fetch_status(run)` and/or
   checks `ingest.is_complete(run.output_dir)` — if the trainer says `ok` or `run_summary.yaml`
   exists, run `ingest.ingest_run(run)` + `_mark(run, OK, finished=True)`. `runner.fetch_status`
   and `ingest.is_complete` already exist (`training/services/runner.py`, `training/services/ingest.py`).
4. **Heal run 2 right now** using that path (its results are complete on disk and just need
   ingesting), OR at minimum flip the stale row so the UI isn't lying.

Note: the trainer container clock is ~2h ahead of the web container (trainer reports
`started_at` 17:00 vs DB 15:00 for the same launch). Cosmetic, but don't trust cross-container
wall-clock math — it produced a bogus "66h" duration.

---

## Problem 2 — "Val loss blew up"

### ROOT CAUSE FOUND & FIXED (2026-07-06): eval labels never loaded (label-filename mismatch)

**This is the real cause of the rising val loss — it supersedes the class-mismatch theory in 2a
below, which was wrong.** The trainer *does* harmonize class spaces at runtime by name
(`friendy_chachkalica/train.py:_remap_targets_to_model_classes` for the loss,
`metrics.py:_remap_to_eval_classes` for mAP), so the 4-vs-8 class difference is handled and is NOT
the problem.

The actual bug: the val/test datasets loaded with **zero labels**. The loader log said it plainly:
```
[data] Dataset ready: images=9084 labels_found=8840 ... train_prototype_0   <- train OK
[data] Dataset ready: images=770  labels_found=0    ... test_dataset_0       <- val:  NO labels
[data] Dataset ready: images=770  labels_found=0    ... test_dataset_0       <- test: NO labels
```
Cause: label-file naming conventions differ between datasets, and
`friendy_chachkalica/data.py:_image_to_label_path` only handled one:
- **train** labels are `<stem>.txt` (e.g. `…rf.abcd.txt`) — matched by `.with_suffix(".txt")`.
- **val/test** labels are `<image filename>.txt` (e.g. `frame_000.jpg.txt`, double extension) —
  `.with_suffix(".txt")` turns `frame_000.jpg` into `frame_000.txt`, which does not exist → miss.

Consequences (all now explained):
- Every eval image loaded as background-only. In YOLOX the val loss then has `cls_loss=0.0` and
  `iou_loss=0.0` every epoch (no foreground assignments) and only objectness/`conf_loss`, which
  **rises** as the model grows more confident (every prediction is a false positive vs empty GT).
- The same empty-GT eval hit **every model** (shared loader) — RF-DETR showed the identical
  val-loss buildup.
- **val/test mAP is also invalid** (scored against empty ground truth), so no eval metric from
  runs 2 or 3 is trustworthy, and the "best" checkpoint (epoch 3) was selected on noise.

**Fixes applied (`friendy_chachkalica/data.py`):**
1. `_image_to_label_path` now tries both `<image>.txt` (appended) and `<stem>.txt` (stripped),
   preferring whichever exists on disk. Safe for train (its appended form doesn't exist).
2. `YoloDetectionDataset` takes `require_labels`; `build_eval_dataloader` passes `require_labels=True`
   so a val/test set with `labels_found=0` now **raises** instead of silently producing junk
   metrics. Train still warns-and-continues.

Verified live on **run 4 (`PPE_v0.1-4`)**: val/test now report `labels_found=770`.

---

### (superseded) 2a. Train/val class mismatch — NOT the cause
> Kept for history. This attributed the bad metrics to a 4-vs-8 class-count/index mismatch between
> `train_prototype_0` and `test_dataset_0`. In fact the trainer remaps classes by name at runtime,
> so the overlap (helmet/vest/no_helmet/no_vest) is handled correctly. The real cause was the
> label-path bug above (eval labels never loaded at all). No dataset re-split is required.

### 2b. YOLOX overfit + no early stopping
`train_loss` 5.66→0.29 while `val_loss` rose monotonically 68→237. Best checkpoint = **epoch 5**
(val 54.4); it then trained 95 more epochs of pure overfitting. Even ignoring 2a, there's **no
early stopping / patience** and 100 epochs is far too many for 9k images. Add early-stopping
(patience on val) and/or reduce epochs; consider stronger augmentation / regularization.

### 2c. RTDETR diverged to NaN on epoch 1 (no checkpoint)
Log: `boxes1 must be in [x0, y0, x1, y1] format, but got tensor([[nan, nan, nan, nan]...])`.
Predicted boxes went NaN immediately. `gradient_clip_norm: null` + `lr: 2e-4` AdamW + AMP is a
likely culprit. Try: enable gradient clipping (e.g. `gradient_clip_norm: 0.1`), lower LR for
rtdetr (e.g. 1e-4 with warmup), and/or disable AMP for this model to test. Config knobs are in the
experiment `training.optimizer` / `gradient_clip_norm` block.

### 2d. RFDETR config crash on epoch 1 (no checkpoint)
Log: `Backbone requires input shape to be divisible by 56, but got torch.Size([8, 3, 900, 900])`.
RF-DETR's DINOv2 backbone needs resolution divisible by 56; **900 is not** (900/56=16.07). Set
`rfdetr.resolution` to a multiple of 56 (e.g. **896** or 952). This is set per-model in the config
(`models[].params.resolution` — see resolved config line ~31 / 148); fix it at the source in the
`ExperimentModel` params or `config_gen.py`.

---

## Suggested order of work
1. Heal the stuck run 2 row (ingest existing on-disk results) — quick win, unblocks the UI.
2. Orchestration fixes 1+2+3 so this never silently happens again.
3. Fix the val dataset (2a) — nothing about val metrics is trustworthy until this is done.
4. RFDETR resolution (2d, one-line) and RTDETR stability (2c), then YOLOX early stopping (2b).
5. Re-run and confirm all 3 models produce `best.pt` and val loss is sane.

## Key files
- `chachkalica/training/jobs.py` — supervise/poll loop, `_mark`, `MAX_WAIT`.
- `chachkalica/training/admin.py` — enqueue sites (add `job_timeout`).
- `chachkalica/fleetsite/settings.py:119-122` — `RQ_QUEUES` / `DEFAULT_TIMEOUT`.
- `chachkalica/training/services/runner.py` — trainer HTTP client (`fetch_status`, `launch`).
- `chachkalica/training/services/ingest.py` — `is_complete`, `ingest_run`.
- `chachkalica/training/services/config_gen.py` — builds the run config from DB models/datasets.
- `friendy_chachkalica/service.py` — trainer FastAPI (launches `run.py`, in-memory job table).
- Evidence for run 2: `chachkalica/data/training/runs/PPE_v0.1-2/{service.log,config.resolved.yaml,run_summary.yaml,metrics.csv}`.
</content>
</invoke>
