# Pipeline audit prompt (for Fable 5)

Paste the block below to an agent to audit the full training pipeline for silent correctness bugs.

---

You are auditing a computer-vision object-detection training pipeline for CORRECTNESS bugs —
specifically the silent kind that don't crash but quietly corrupt training or produce meaningless
metrics. Bias toward "runs fine, numbers are garbage" failures over crashes.

Repo layout (monorepo, run via docker-compose.yml at root):
- friendy_chachkalica/  — the actual PyTorch trainer (config-driven, reads an experiment YAML).
    config.py           — parses the experiment YAML into dataclasses (classes as list OR dict,
                          num_classes: auto, dataset roles train/val/test).
    data.py             — YOLO-format dataset + dataloaders (image↔label pairing, label parsing:
                          normalized xywh / OBB / polygon, box validity, class ids).
    train.py            — training loop, per-epoch val-loss eval, best-checkpoint selection,
                          early stopping, resume; _remap_targets_to_model_classes (class
                          harmonization by NAME between the val set and the model's train classes).
    val.py, eval_checkpoint.py, metrics.py — post-train val/test eval and mAP; metrics.py also
                          remaps predictions/targets to an eval class space by name.
    adapters/{yolox,rtdetr,rfdetr,retinanet}.py — per-model loss/predict/checkpoint; note
                          resolution divisibility constraints, AMP, grad-clip, num_classes head
                          re-init when loading pretrained weights.
    run.py, service.py  — CLI entry + FastAPI service that launches run.py as a subprocess.
- chachkalica/          — Django app that GENERATES the trainer YAML and orchestrates runs.
    training/services/config_gen.py — builds the trainer YAML from DB rows (datasets, models,
                          class maps come from each dataset's classes.txt).
    training/jobs.py, services/runner.py — enqueue/launch/poll/ingest, status marking, timeouts.

CONTEXT — a bug of exactly the class I care about was just found and fixed, use it as the pattern
to hunt for more like it: train label files are named "<stem>.txt" but the val/test set used
"<image>.jpg.txt" (double extension). The loader's `.with_suffix(".txt")` only matched the train
convention, so ALL val/test labels silently failed to load (labels_found=0). Training "succeeded,"
but val loss was pure objectness (cls/iou loss = 0) and rose every epoch, and val/test mAP was
computed against empty ground truth — all metrics were meaningless, and the "best" checkpoint was
selected on noise. Nothing crashed.

Audit these areas and report concrete findings (file:line, what breaks, failing input, fix):

1. DATA INTEGRITY (data.py, config.py)
   - Image↔label pairing across naming conventions and subdirectories; silently-dropped labels.
   - Label parsing: normalized-vs-pixel coordinate assumptions, OBB (5-val) and polygon paths,
     degenerate/negative boxes, off-by-one class ids, header/comment lines, empty files.
   - Are train and eval images/boxes put through the SAME coordinate space? Any resize/letterbox
     applied to images but not boxes (or vice versa)?
   - classes.txt parsing → class list/dict; index alignment; what happens with a "# tools:" header.

2. CLASS HARMONIZATION (train.py remap, metrics.py remap, config_gen.py)
   - num_classes: auto resolution — does the model head size match what train/val/test feed it?
   - The by-name remap: does it correctly keep the intersection and drop the rest? What if a name
     exists in val but not train (and vice versa)? Is anything logged when classes are dropped?
   - Are TRAIN targets remapped consistently with VAL targets, or is there an asymmetry?

3. LOSS / METRIC VALIDITY
   - Any path where a loss or mAP is computed against empty/mismatched ground truth and reported
     as a real number. Any metric that can be silently 0 / NaN / undefined.
   - best-checkpoint selection criterion — val loss can diverge from mAP; is the selected "best"
     actually the best model? Should selection use mAP instead of/in addition to val loss?
   - Early-stopping patience logic, scheduler.step() timing, LR/warmup per model.

4. PER-MODEL ADAPTERS
   - Resolution/stride divisibility constraints (e.g. RF-DETR needs multiples of 56); AMP + NaN
     divergence (RT-DETR); grad-clip actually applied; pretrained-weight head re-init when class
     count differs; checkpoint save/load round-trips (num_classes, class names persisted?).

5. ORCHESTRATION (jobs.py, runner.py, service.py, run.py)
   - Does run.py swallow per-model failures and still exit 0 (making the service report "ok" when
     models failed with no checkpoint)? Surface every failure mode that looks like success.
   - Job timeouts vs poll budget; status marking on crash/timeout; results ingestion gaps;
     cross-container clock skew affecting any logic.

For each finding: severity (does it corrupt training silently, or just crash?), the exact
file:line, a minimal failing input, and the smallest correct fix. Prioritize silent-corruption
bugs. Do NOT propose stylistic changes. Read the code before claiming a bug — verify the data flow
end to end rather than pattern-matching.
