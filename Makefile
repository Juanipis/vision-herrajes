# Simple task runner for the vision-herrajes toolkit.
SHELL := /bin/bash
.ONESHELL:

VENV ?= .venv
PYTHON := $(VENV)/bin/python
UV := $(VENV)/bin/uv
REQ_PKGS := opencv-python pillow av tqdm scikit-image scikit-learn scipy joblib

.PHONY: help gui preview install-deps dataset train classify evaluate clean-pycache clean-ds

help:
	@echo "Available targets:"
	@echo "  make gui             # Launch the Tkinter tuning GUI"
	@echo "  make preview VIDEO=path [PRESET=name] [STEP=30] [OUTPUT=dir]"
	@echo "                      # Export preview frames for a video"
	@echo "  make install-deps    # Ensure GUI and scripts dependencies are installed"
	@echo "  make dataset PRESET=name   # Build balanced mask dataset"
	@echo "  make train           # Train the MLP classifier on mask features"
	@echo "  make classify        # Launch the inference GUI"
	@echo "  make evaluate        # Batch-evaluate videos in train/"
	@echo "  make clean-pycache   # Remove __pycache__ directories"
	@echo "  make clean-ds        # Remove .DS_Store files"

install-deps:
	@if [ ! -x "$(UV)" ]; then \
		echo "uv not found in $(UV). Ensure the virtual environment exists."; \
		exit 1; \
	fi
	UV_CACHE_DIR=$$(pwd)/.uv-cache \
	"$(UV)" pip install $(REQ_PKGS)

_gui-check:
	@if [ ! -x "$(PYTHON)" ]; then \
		echo "Python executable not found at $(PYTHON). Create the venv or adjust PYTHON."; \
		exit 1; \
	fi

# Launch the interactive Tkinter GUI for tuning filters.
gui: _gui-check
	"$(PYTHON)" -m src.gui.video_tuner

# Generate side-by-side previews from a video using the current filter pipeline.
preview: _gui-check
	@if [ -z "$(VIDEO)" ]; then \
		echo "ERROR: VIDEO=/path/to/file is required" >&2; \
		exit 1; \
	fi
	cmd=("$(PYTHON)" scripts/preview_filters.py "$(VIDEO)")
	if [ -n "$(OUTPUT)" ]; then
		cmd+=(--output "$(OUTPUT)")
	fi
	if [ -n "$(PRESET)" ]; then
		cmd+=(--preset "$(PRESET)")
	fi
	if [ -n "$(STEP)" ]; then
		cmd+=(--step "$(STEP)")
	fi
	if [ -n "$(MAX_WIDTH)" ]; then
		cmd+=(--max-width "$(MAX_WIDTH)")
	fi
	"$${cmd[@]}"

# Build a balanced, ROI-cropped dataset of frames using the configured preset.
dataset: _gui-check
	"$(PYTHON)" scripts/build_dataset.py$(if $(OUTPUT_DIR), --output "$(OUTPUT_DIR)")$(if $(TRAIN_DIR), --train "$(TRAIN_DIR)")$(if $(PRESET), --preset "$(PRESET)")$(if $(SEED), --seed "$(SEED)")

# Train the MLP classifier using extracted mask features.
train: _gui-check
	"$(PYTHON)" scripts/train_mlp.py$(if $(MANIFEST), --manifest "$(MANIFEST)")$(if $(SEED), --seed "$(SEED)")

# Launch the classification viewer GUI.
classify: _gui-check
	"$(PYTHON)" -m src.gui.model_viewer$(if $(MODEL), --model "$(MODEL)")$(if $(PRESET), --preset "$(PRESET)")

evaluate: _gui-check
	"$(PYTHON)" scripts/batch_classify.py$(if $(MODEL), --model "$(MODEL)")$(if $(PRESET), --preset "$(PRESET)")$(if $(TRAIN_DIR), --train-dir "$(TRAIN_DIR)")$(if $(WORKERS), --workers "$(WORKERS)")$(if $(CAPTURE_DIR), --capture-dir "$(CAPTURE_DIR)")

clean-pycache:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +

clean-ds:
	find . -name '.DS_Store' -delete
