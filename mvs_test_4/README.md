# CLIP finetuning and data collection (mvs_test_4)

This folder contains dataset collection helpers and a CLIP finetuning script used for the intersection driving experiments.

Files of interest
- `clip_rldrive_data_collection_ambulance_ego_yielding_traffic.py` — CLIP finetuning trainer. Expects a folder with `images/` and `train.jsonl`, `val.jsonl`, `test.jsonl`.
- `prepare_dataset_for_clip.py` — helper to convert `frame.json` or `frames.jsonl` into `train.jsonl`/`val.jsonl`/`test.jsonl` splits inside `data/`.
- `clip-rl-hybrid_2.py` — enhanced pipeline (collection + finetune + RL) used elsewhere in the repo.

Quick start
1. Create (or activate) a Python virtualenv and install dependencies (see `requirements.txt`). For CUDA/GPU support, install a matching `torch` wheel from https://pytorch.org/get-started/locally/ before installing other deps.

2. Prepare the dataset splits (this will read `data/frame.json` or `data/frames.jsonl` if present):

```bash
python3 prepare_dataset_for_clip.py --data-dir ./data --seed 0
```

3. Run a quick dry-run finetune (small batch/epochs):

```bash
python3 clip_rldrive_data_collection_ambulance_ego_yielding_traffic.py \
  --root ./data --out ./runs/first_training --epochs 20 --bs 8 --bs-eval 16 --stack-frames 4
```

Notes on dependencies
- The trainer uses the MLFoundations `open_clip` implementation. Install via git:

```bash
pip install git+https://github.com/mlfoundations/open_clip.git
```

- Alternatively, try `pip install open_clip_torch` if it exists for your environment, but installing from the git repo is recommended.

- Install `torch` first using the official wheel for your CUDA version (see https://pytorch.org/get-started/locally/). Example CPU-only (linux):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Troubleshooting
- Image paths: the dataset expects `image` fields in JSONL to be relative paths that resolve under `--root`. If you see FileNotFoundError for an index, list `data/images/` and inspect names.
- JSON vs JSONL: the helper supports both a JSON array (`frame.json`) and JSONL (`frames.jsonl`).
- If torch raises pickle/unpickling errors loading checkpoints, ensure the trainer saves only plain builtins (the code converts defaultdicts to plain dicts).

License & credits
- Code adapted from the project repo; see the top-level README for license details.
