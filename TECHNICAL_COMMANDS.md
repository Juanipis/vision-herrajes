# Technical Commands

## Main Dataset & Training
- `make dataset PRESET=<preset> [TRAIN_DIR=...] [OUTPUT_DIR=...] [SEED=...] [NO_SAVE_FRAMES=1] [PNG_COMPRESSION=0-9] [KEEP_EVERY=N] [SCAN_STEP=N]`
- `make submodel-base PRESET=<preset> [SOURCE_MANIFEST=...] [SOURCE_ROOT=...] [OUTPUT_DIR=...] [LIMIT=...] [DRY_RUN=1]`
- `make train [MANIFEST=path/to/manifest.json] [SEED=...]`

## Inference & Evaluation
- `make classify [MODEL=path/to/model.joblib] [PRESET=<main>] [SIZE_PRESET=<submodel>]`
- `make evaluate [MODEL=...] [PRESET=...] [TRAIN_DIR=...] [WORKERS=...] [CAPTURE_DIR=...]`

## Filter Tuning & Previews
- `make gui`
- `make preview VIDEO=path/to/video [PRESET=...] [STEP=...] [OUTPUT=...] [MAX_WIDTH=...]`

## Size Submodels
- `make size-dataset FAMILY=<family|*> [QUALITIES=BUENO|MALO|ALL...] [MANIFEST=...] [MASK_ROOT=...] [OUTPUT_DIR=...] [BALANCE=1] [SEED=...]`
- `make size-train FAMILY=<family|*> [QUALITIES=...] [DATASET=path/to/dataset.npz] [OUTPUT_DIR=...] [SEED=...] [HIDDEN="h1 h2 ..."]`

## Maintenance
- `make clean-pycache`
- `make clean-ds`
- `make install-deps`
