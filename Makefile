# nano-kws — one-command developer UX.
#
# Conventions:
#   * Targets are POSIX-shell. On Windows use `make` from Git Bash, WSL, or
#     `nmake`-compatible wrappers; PowerShell users can also invoke each
#     command directly (see the body of each target).
#   * `make install` is editable + dev extras: training, quantization, tests.
#   * `make benchmark` and `make app` work on the bundled INT8 model — no
#     training required.

PYTHON ?= python
PIP    ?= $(PYTHON) -m pip

.PHONY: help install install-runtime lint format test cov \
        download-data train quantize export benchmark sweep app \
        docker docker-run clean

help:
	@echo "Targets:"
	@echo "  install         Editable install with dev extras (torch, pytest, ruff, ...)"
	@echo "  install-runtime Runtime-only install (inference + demo, no training)"
	@echo "  lint            Run ruff lint + format check"
	@echo "  format          Apply ruff formatting"
	@echo "  test            Run pytest"
	@echo "  cov             Run pytest with coverage report"
	@echo "  download-data   Fetch Speech Commands v2 to the local cache"
	@echo "  train           Train DS-CNN small (default config)"
	@echo "  quantize        Static PTQ + INT8 ONNX export"
	@echo "  export          fp32 ONNX export"
	@echo "  benchmark       fp32 vs INT8 latency / size / accuracy table"
	@echo "  sweep           Train tiny/small/medium and emit accuracy-vs-MACs plot"
	@echo "  app             Launch Streamlit live mic demo"
	@echo "  docker          Build the inference Docker image"
	@echo "  docker-run      Run the benchmark inside the Docker image"
	@echo "  clean           Remove build artefacts and caches"

# --- install --------------------------------------------------------------

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	pre-commit install || true

install-runtime:
	$(PIP) install --upgrade pip
	$(PIP) install -e .

# --- quality gates --------------------------------------------------------

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest

cov:
	pytest --cov=nano_kws --cov-report=term-missing --cov-report=xml

# --- pipeline -------------------------------------------------------------

download-data:
	$(PYTHON) -m scripts.download_data

train:
	$(PYTHON) -m nano_kws.train --width 0.5 --epochs 30

quantize:
	$(PYTHON) -m nano_kws.quantize \
	    --checkpoint assets/ds_cnn_w0p5.pt \
	    --output assets/ds_cnn_small_int8.onnx \
	    --fp32-output assets/ds_cnn_small_fp32.onnx

export:
	$(PYTHON) -m nano_kws.export_onnx \
	    --checkpoint assets/ds_cnn_w0p5.pt \
	    --output assets/ds_cnn_small_fp32.onnx

benchmark:
	$(PYTHON) -m nano_kws.benchmark \
	    --checkpoint assets/ds_cnn_w0p5.pt \
	    --fp32 assets/ds_cnn_small_fp32.onnx \
	    --int8 assets/ds_cnn_small_int8.onnx \
	    --output assets/benchmark_table.md \
	    --update-readme

sweep:
	$(PYTHON) -m scripts.sweep_sizes \
	    --widths 0.25 0.5 1.0 \
	    --epochs 30 \
	    --update-readme \
	    --publish-canonical

app:
	streamlit run app/streamlit_app.py

# --- docker ---------------------------------------------------------------

docker:
	docker build -t nano-kws:latest .

docker-run:
	docker run --rm -v "$(PWD)/assets:/app/assets" nano-kws:latest \
	    python -m nano_kws.benchmark \
	        --int8 assets/ds_cnn_small_int8.onnx \
	        --output assets/benchmark_table.md

# --- housekeeping ---------------------------------------------------------

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
