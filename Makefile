PYTHON ?= python3
export PYTHONPATH := src

.PHONY: help install install-train test lint dataset index train eval ablation serve chat notebook clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## CPU-only dependencies (retrieval, evaluation, serving)
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

install-train:  ## Add GPU fine-tuning dependencies
	$(PYTHON) -m pip install -r requirements-train.txt

test:  ## Run the unit test suite
	$(PYTHON) -m pytest tests

lint:  ## Static checks
	$(PYTHON) -m ruff check src tests scripts

dataset:  ## Build the QA dataset from raw NBR PDFs listed in data/raw/manifest.json
	$(PYTHON) scripts/prepare_dataset.py --config configs/default.yaml

index:  ## Build the retrieval index from processed segments
	$(PYTHON) scripts/build_index.py --config configs/default.yaml

train:  ## QLoRA fine-tuning (requires a CUDA GPU)
	$(PYTHON) -m banglafingpt.models.train --config configs/default.yaml --index-dir artifacts/index

eval:  ## Evaluate the full system on the held-out test set
	$(PYTHON) -m banglafingpt.eval.evaluate --config configs/default.yaml --tag banglafingpt

ablation:  ## Reproduce the ablation and significance tables
	$(PYTHON) -m banglafingpt.eval.ablation --config configs/default.yaml

chat:  ## Interactive bilingual CLI
	$(PYTHON) -m banglafingpt.app.cli --config configs/default.yaml

serve:  ## FastAPI service on :8000
	$(PYTHON) -m uvicorn banglafingpt.app.api:app --host 0.0.0.0 --port 8000

notebook:  ## Re-execute the reproduction notebook in place
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=1800 notebooks/BanglaFinGPT_reproduction.ipynb

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
