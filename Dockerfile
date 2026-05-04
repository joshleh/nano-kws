# Inference-only image: runs the bundled INT8 ONNX model and the benchmark.
# Training is intentionally out of scope for the container — it pulls
# torch/torchaudio which would inflate the image to multiple GB.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps: libsndfile for soundfile.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install runtime deps first (better layer caching).
COPY requirements.txt pyproject.toml README.md LICENSE ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the package and the bundled assets needed at runtime.
COPY nano_kws ./nano_kws
COPY app ./app
COPY assets ./assets

RUN pip install --no-deps -e .

CMD ["python", "-m", "nano_kws.benchmark", \
     "--int8", "assets/ds_cnn_small_int8.onnx", \
     "--output", "assets/benchmark_table.md"]
