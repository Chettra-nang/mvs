# clip-rl-hybrid-3.py — Quick start

This README explains how to set up a Python environment and run the `clip-rl-hybrid-3.py` script located in this folder. The script supports data collection, dataset preparation, CLIP fine-tuning, RL training (DQN/PPO), and evaluation.

## Requirements
- Python 3.8+ (3.8–3.11 recommended)
- A GPU with CUDA is optional but recommended for training.

Python packages used by the script (not exhaustive):
- torch, torchvision
- open-clip-torch (imported as `open_clip`) and the OpenAI `clip` package
- scikit-learn, matplotlib, tqdm, pillow
- gymnasium, highway-env
- stable-baselines3
- wandb (optional)
- peft, ollama (optional)
- numpy

## Create a virtual environment (recommended)
Run these commands from this folder (Bash):

```bash
python3 -m venv mvs_venv
source mvs_venv/bin/activate
pip install -U pip
```

## Install dependencies (example)
Install a CPU-only PyTorch wheel or the CUDA wheel from pytorch.org depending on your system. Example (CPU-only):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install open-clip-torch git+https://github.com/openai/CLIP.git
pip install scikit-learn matplotlib tqdm pillow gymnasium highway-env stable-baselines3[extra] wandb
pip install peft ollama
```

Notes:
- If you have CUDA, install the matching PyTorch wheel from https://pytorch.org instead of the CPU wheel above.
- `open-clip-torch` is used via `import open_clip`. The repository also imports `clip` (OpenAI CLIP); installing both is recommended.
- `peft` and `ollama` are optional — the script checks availability and will work without them.

## Where the script stores data and models
- Data: `data/` (images in `data/images/`, JSONL dataset files)
- Fine-tuned CLIP checkpoints: saved under output directory you provide (default `runs/improved_clip`)
- RL models: `models/` (default `models/enhanced_dqn_clip` or `models/enhanced_ppo_clip`)

## Usage
Run the script with the `--mode` flag. Example command format:

```bash
python clip-rl-hybrid-3.py --mode <MODE> [options]
```

Modes and examples:
- collect — collect data and save to `data/` (creates `data/images/` and `data/frames.jsonl`):

```bash
python clip-rl-hybrid-3.py --mode collect
```

- prepare — split an existing dataset in `data/` into train/val/test JSONL files:

```bash
python clip-rl-hybrid-3.py --mode prepare --root data
```

- finetune — fine-tune CLIP using `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`:

```bash
python clip-rl-hybrid-3.py --mode finetune --root data --out runs/improved_clip --epochs 100 --bs 64
```

- train_dqn — train a DQN RL agent (uses CLIP reward if `--clip-path` points to a checkpoint):

```bash
python clip-rl-hybrid-3.py --mode train_dqn --clip-path robust_clip_finetuned.pt --model-path models/enhanced_dqn_clip
```

- train_ppo — train a PPO RL agent:

```bash
python clip-rl-hybrid-3.py --mode train_ppo --clip-path robust_clip_finetuned.pt --model-path models/enhanced_ppo_clip
```

- evaluate — evaluate a saved RL model (DQN by default). If you used PPO, adjust the loader inside the script or load the model manually.

```bash
python clip-rl-hybrid-3.py --mode evaluate --clip-path robust_clip_finetuned.pt --model-path models/enhanced_dqn_clip
```

## Useful CLI flags available in the script
- `--root` : data dir (default `data`)
- `--out`  : output dir for finetune (default `runs/improved_clip`)
- `--clip-path` : path to CLIP checkpoint
- `--model-path` : RL model save/load path
- `--labels` : list custom labels (space separated)
- `--stack-frames` : number of stacked frames (default 8)
- `--epochs`, `--bs`, `--lr`, `--seed`, `--use-paraphrases`, `--tune-mode`, `--balanced-sampler`, `--oversample-factor`, `--device`, `--use-wandb`

## Troubleshooting
- Import errors: install the missing package shown in the traceback (pip install <pkg>).
- `torch`/CUDA mismatch: install the correct torch wheel for your CUDA version. Check with:

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)
PY
```

- `highway-env` issues: ensure `highway-env` and its dependencies (Box2D, etc.) are installed. See highway-env docs if rendering/backends fail.
- No data for fine-tune: ensure `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl` exist and `data/images/` contains corresponding images.
- OOM on GPU: reduce `--bs` or `--stack-frames`, or use gradient accumulation / smaller image sizes.

## Quick test suggestion
To verify things run end-to-end quickly, consider modifying `collect_improved_data` call to use a small `n_episodes` (e.g., 5) or run only `prepare`/`evaluate` on a very small dataset.

## Next improvements (optional)
- Add CLI flags for `collect_improved_data` like `--n-episodes`.
- Provide separate small test preset script to run a short end-to-end smoke test.

If you'd like, I can add a `--n-episodes` flag to `collect_improved_data` and update the script; tell me and I'll implement it and run a quick smoke test.
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
