# Trainer service image: the friendy_chachkalica torch pipeline behind its FastAPI
# wrapper (service.py). Django talks to it over HTTP and shares files on disk
# (see docker-compose.yml for how the shared filesystem is wired).
FROM python:3.12-slim

# opencv-python needs these shared libs present at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install every architecture's deps. rtdetr and rfdetr each pull in the base
# requirements (-r requirements.txt) plus their own extras — note they are
# DIFFERENT models: rtdetr (RT-DETR, via transformers/scipy) and rfdetr (RF-DETR,
# the Apache-2.0 `rfdetr` package). retinanet is plain torchvision (base) and
# yolox is vendored under vendor/ (no package). The default torch wheels are
# CUDA-enabled on linux/x86_64 and fall back to CPU when no GPU is visible.
COPY requirements.txt requirements-rtdetr.txt requirements-rfdetr.txt requirements-yolox.txt ./
RUN pip install --no-cache-dir -r requirements-rtdetr.txt -r requirements-rfdetr.txt

# Trainer code: service.py, run.py, eval_checkpoint.py, adapters/, vendor/, ...
COPY . /app

# Listen on the port Django targets (TrainingSettings.service_base_url). The
# value is overridable at runtime via the FC_PORT env var; EXPOSE is only docs.
ENV FC_PORT=8200
EXPOSE 8200

# Shell form so $FC_PORT expands at container start. service.py defines `app`
# and launches run.py / eval_checkpoint.py as subprocesses from this WORKDIR.
CMD uvicorn service:app --host 0.0.0.0 --port ${FC_PORT}
