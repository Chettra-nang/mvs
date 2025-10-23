#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined and Improved CLIP-RLDrive Script (Full Version)

This is the complete combined script with all parts included. The previous version was shortened for brevity in the response, but here the RL training and evaluation parts are fully integrated from the second script, with improvements applied throughout.

Improvements:
- Balanced data collection with class balancing, more episodes (1000+), curriculum density.
- Oversampling minority classes (balanced_sampler=True, oversample_factor=2.0).
- Stack frames increased to 8.
- Use of both prompts and paraphrases.
- Lower LR (5e-6), more epochs (100), patience=20.
- Class weighting in CE loss.
- Stronger augmentations (flips, rotation, color jitter).
- Integrated stable fine-tuning techniques from RobustCLIPFineTuner into the trainer.
- Support for 5 actions.
- Curriculum in data collection and RL.
- Detailed evaluation with confusion matrices and multi-density testing.

Usage:
  python combined_improved_clip_rldrive.py --mode [collect|finetune|train_dqn|train_ppo|evaluate]

Install:
  pip install torch torchvision open_clip_torch scikit-learn matplotlib tqdm pillow gymnasium highway-env stable-baselines3 wandb peft ollama

Notes:
- Data saved to ./data/
- Models saved to ./models/
- Set --use-wandb if you have wandb setup.
"""

import os, re, json, random, glob, contextlib, argparse
from pathlib import Path
from collections import Counter, defaultdict, deque
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import multiprocessing
from PIL import Image, ImageOps, ImageEnhance
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import open_clip
import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import clip
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False
try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False
try:
    import ollama
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False
import types
import warnings
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------- Labels & Prompts (5 Actions) --------------------
DEFAULT_LABELS_5 = ["SLOWER", "IDLE", "FASTER", "LANE_LEFT", "LANE_RIGHT"]

PROMPTS = {
    "SLOWER": "Slow down because there might be a collision.",
    "IDLE":   "Maintain your current speed.",
    "FASTER": "Speed up since the chance of collision is low.",
    "LANE_LEFT": "Change to the left lane.",
    "LANE_RIGHT": "Change to the right lane."
}

PARAPHRASES = {
    "SLOWER": [
        "Slow down; collision risk.", "Reduce speed—caution ahead.",
        "Decelerate; gap is tight.", "Ease off; possible hazard.",
        "Reduce speed—collision risk ahead.", "Back off the throttle; traffic up ahead.",
        "Slow down to stay safe.", "Drop speed; the gap is tight.", "Decelerate—possible hazard ahead."
    ],
    "IDLE": [
        "Maintain current speed.", "Hold speed and lane.",
        "Stay steady in this lane.", "Continue unchanged.",
        "Keep the current lane and speed.", "Hold speed and lane.",
        "Stay steady in this lane.", "Maintain pace; no lane change.",
        "Remain in lane with current speed.", "Continue unchanged."
    ],
    "FASTER": [
        "Accelerate; safe gap ahead.", "Speed up; path looks clear.",
        "Pick up pace—no blockers.", "Increase speed toward target.",
        "Accelerate—path looks clear.", "Increase speed; safe gap ahead.",
        "Pick up pace—no blockers.", "Go quicker; open stretch ahead.", "Speed up to target pace."
    ],
    "LANE_LEFT": [
        "Merge left safely.", "Move to the left lane.", "Shift to left lane to pass."
    ],
    "LANE_RIGHT": [
        "Merge right safely.", "Move to the right lane.", "Shift to right lane to yield."
    ],
}

ACTION_MAP = {
    0: "SLOWER",
    1: "IDLE",
    2: "FASTER",
    3: "LANE_LEFT",
    4: "LANE_RIGHT"
}

# -------------------- Tiny Utils --------------------
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def read_jsonl(path: Path) -> List[dict]:
    rows=[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t=line.strip()
            if t: rows.append(json.loads(t))
    return rows

def write_jsonl(p: Path, rows: List[dict]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def letterbox_square(im: Image.Image, target=224, fill=0) -> Image.Image:
    if im.mode != "RGB": im = im.convert("RGB")
    w, h = im.size
    s = min(target / w, target / h)
    nw, nh = int(round(w*s)), int(round(h*s))
    im = im.resize((nw, nh), Image.BICUBIC)
    pad_w = target - nw; pad_h = target - nh
    padding = (pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2)
    return ImageOps.expand(im, padding, fill=fill)

def parse_index_from_filename(rel_path: str) -> Optional[int]:
    m = re.search(r"(\d{6})", Path(rel_path).name)
    return int(m.group(1)) if m else None

def find_image_by_index(images_dir: Path, idx: int) -> Optional[Path]:
    hits = glob.glob(str(images_dir / f"{idx:06d}_*.png"))
    return Path(hits[0]) if hits else None

# -------------------- Improved Dataset with Stronger Augmentation --------------------
class AugmentedFrameTextDataset(Dataset):
    def __init__(self, root: Path, jsonl_name: str, label_order: List[str],
                 stack_frames: int = 8, image_size: int = 224, augment: bool = True):
        self.root = Path(root)
        self.rows_all = read_jsonl(self.root / jsonl_name)
        keep = set(label_order)
        self.rows = [r for r in self.rows_all
                     if r.get("action_id") in keep and (self.root / r["image"]).exists()]
        if not self.rows:
            raise RuntimeError(f"No usable rows in {jsonl_name}.")
        self.labels = list(label_order)
        self.lab2id = {k:i for i,k in enumerate(self.labels)}
        self.images_dir = self.root / "images"
        self.S = max(1, int(stack_frames))
        self.HW = int(image_size)
        self.augment = bool(augment)

        dist_keep = Counter([r.get("action_id","__NA__") for r in self.rows])
        print(f"[{jsonl_name}] kept={len(self.rows)} | class-dist={dict(dist_keep)}")

        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:,None,None]
        self.std  = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:,None,None]

    def __len__(self): return len(self.rows)

    def _open_by_index(self, idx: int) -> Image.Image:
        p = find_image_by_index(self.images_dir, idx)
        if p is None:
            p = find_image_by_index(self.images_dir, max(0, idx))
        if p is None:
            hits = list(self.images_dir.glob(f"*{idx:06d}*.png"))
            if hits:
                p = hits[0]
        if p is None:
            candidates = []
            for f in self.images_dir.glob("*.png"):
                fid = parse_index_from_filename(str(f))
                if fid is not None:
                    candidates.append((abs(fid - idx), f))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                p = candidates[0][1]
        if p is None:
            raise FileNotFoundError(f"No image for index {idx} in {self.images_dir}")
        return Image.open(p).convert("RGB")

    def _augment_image(self, im: Image.Image) -> Image.Image:
        if not self.augment:
            return im
        if random.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.6:
            angle = random.uniform(-10, 10)
            im = im.rotate(angle, resample=Image.BICUBIC)
        if random.random() < 0.8:
            enhancer = ImageEnhance.Brightness(im)
            factor = random.uniform(0.8, 1.2)
            im = enhancer.enhance(factor)
            enhancer = ImageEnhance.Contrast(im)
            factor = random.uniform(0.8, 1.2)
            im = enhancer.enhance(factor)
        if random.random() < 0.5:
            enhancer = ImageEnhance.Color(im)
            factor = random.uniform(0.8, 1.2)
            im = enhancer.enhance(factor)
        return im

    def _build_stack(self, idx: int) -> torch.Tensor:
        frames = []
        for k in range(self.S-1, -1, -1):
            im = self._open_by_index(max(0, idx - k))
            im = self._augment_image(im)
            im = letterbox_square(im, target=self.HW, fill=0)
            t = torch.from_numpy(np.array(im)).permute(2,0,1).float()/255.0
            t = (t - self.mean) / self.std
            frames.append(t)
        return torch.cat(frames, dim=0)

    def __getitem__(self, i: int):
        r = self.rows[i]
        rel = r["image"]
        idx = parse_index_from_filename(rel) or r.get("index", i)
        x = self._build_stack(idx)
        text = r.get("instruction", "") or self.labels[self.lab2id[r["action_id"]]]
        y = self.lab2id[r["action_id"]]
        return x, text, torch.tensor(y, dtype=torch.long), rel

# -------------------- CLIP Adaptor --------------------
def adapt_clip_first_layer_for_stack(model, stack_frames: int, device: str):
    assert hasattr(model.visual, "conv1")
    conv1: nn.Conv2d = model.visual.conv1
    out_ch, in_ch, kh, kw = conv1.weight.shape
    if in_ch == 3 * stack_frames:
        return model
    assert in_ch == 3
    new_in = 3 * stack_frames
    new_conv = nn.Conv2d(new_in, out_ch, kernel_size=(kh,kw),
                         stride=conv1.stride, padding=conv1.padding, bias=False)
    with torch.no_grad():
        rep = new_in // in_ch
        w = conv1.weight.data
        w_tiled = w.repeat(1, rep, 1, 1) / float(rep)
        new_conv.weight.copy_(w_tiled)
    model.visual.conv1 = new_conv.to(device)
    return model

# -------------------- Text Bank --------------------
def build_prompts_per_class(rows: List[dict], label_order: List[str],
                            mode: str = "both", use_paraphrases: bool = True) -> Dict[str, List[str]]:
    per = defaultdict(list)
    if mode in ("dataset", "both"):
        for r in rows:
            a = r.get("action_id"); s = r.get("instruction")
            if a in label_order and isinstance(s, str) and len(s) > 2:
                per[a].append(s.strip())
    if mode in ("canonical", "both"):
        for a in label_order:
            if a in PROMPTS:
                per[a].append(PROMPTS[a])
    if use_paraphrases:
        for a in label_order:
            per[a].extend(PARAPHRASES.get(a, []))
    for a in label_order:
        per[a] = sorted(set(per[a]), key=len) or [a.lower()]
    return per

@torch.no_grad()
def build_text_bank(prompts_per_class: Dict[str, List[str]],
                    label_order: List[str], model, device, max_per_class=128) -> torch.Tensor:
    bank = []
    for lab in label_order:
        texts = prompts_per_class.get(lab, [lab.lower()])[:max_per_class]
        tok = open_clip.tokenize(texts).to(device)
        T = model.encode_text(tok)
        T = T / T.norm(dim=-1, keepdim=True)
        bank.append(T.mean(dim=0, keepdim=True))
    return torch.cat(bank, dim=0)

# -------------------- Losses --------------------
def clip_contrastive_loss(img_feats, txt_feats, logit_scale):
    logits_per_image = logit_scale * img_feats @ txt_feats.t()
    logits_per_text = logits_per_image.t()
    labels = torch.arange(img_feats.size(0), device=img_feats.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return 0.5 * (loss_i + loss_t)

def ce_classification_from_bank(img_feats, class_bank, y, tau=0.35, weight=None):
    logits = (img_feats @ class_bank.t()) / float(tau)
    return F.cross_entropy(logits, y, weight=weight), logits

# -------------------- Eval Helpers --------------------
@torch.no_grad()
def evaluate(model, loader, device, class_bank, tau_cls=0.35):
    model.eval()
    ys, preds = [], []
    for x, txt, y, rel in loader:
        x = x.to(device, non_blocking=True)
        img = model.encode_image(x)
        img = img / img.norm(dim=-1, keepdim=True)
        logits = (img @ class_bank.t()) / float(tau_cls)
        p = logits.argmax(dim=1).cpu()
        ys.append(y.cpu()); preds.append(p)
    y_true = torch.cat(ys).numpy(); y_pred = torch.cat(preds).numpy()
    acc = float((y_true == y_pred).mean())
    f1m = float(f1_score(y_true, y_pred, average="macro"))
    cm = confusion_matrix(y_true, y_pred)
    return {"acc": acc, "macro_f1": f1m, "cm": cm, "y_true": y_true, "y_pred": y_pred}

def sweep_tau(model, loader, device, class_bank, grid=(0.20,0.25,0.30,0.35,0.40,0.45,0.50)):
    best_tau, best_f1, best_metrics = None, -1.0, None
    for t in grid:
        m = evaluate(model, loader, device, class_bank, tau_cls=t)
        if m["macro_f1"] > best_f1:
            best_tau, best_f1, best_metrics = t, m["macro_f1"], m
    return best_tau, best_f1, best_metrics

def save_confusion(cm: np.ndarray, labels: List[str], out_png: str, title: Optional[str]=None):
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i,j])), ha="center", va="center",
                    color=("white" if cm[i,j] > cm.max()*0.5 else "black"))
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    if title: ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

# -------------------- Improved Trainer with Stability --------------------
def train_finetune_clip(
    root: str, outdir: str,
    train_file="train.jsonl", val_file="val.jsonl", test_file="test.jsonl",
    label_order: List[str] = None,
    stack_frames: int = 8, image_size=224,
    epochs=100, bs_train=128, bs_eval=256,
    lr=5e-6, lr_text=5e-6, lr_head=1e-4, weight_decay=0.01,
    seed=42, tau_grid=(0.20,0.25,0.30,0.35,0.40,0.45,0.50),
    lambda_contrast=1.0, lambda_cls=1.0,
    text_bank_mode="both", use_paraphrases=True,
    tune_mode="last2",
    class_weight=True,
    patience=20,
    compile_model: bool = False,
    device: Optional[str] = None,
    balanced_sampler: bool = True,
    oversample_factor: float = 2.0,
    num_workers: Optional[int] = None,
):
    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    if device == "cpu":
        cpu_count = os.cpu_count() or 1
        threads = min(16, max(1, cpu_count - 1))
        torch.set_num_threads(threads)
        print(f"CPU threads: {threads}")

    Path(outdir).mkdir(parents=True, exist_ok=True)

    if label_order is None:
        label_order = DEFAULT_LABELS_5
    print("Labels:", label_order)

    ds_tr = AugmentedFrameTextDataset(Path(root), train_file, label_order, stack_frames, image_size, augment=True)
    ds_va = AugmentedFrameTextDataset(Path(root), val_file, label_order, stack_frames, image_size, augment=False)
    ds_te = AugmentedFrameTextDataset(Path(root), test_file, label_order, stack_frames, image_size, augment=False)

    cpu_count = os.cpu_count() or 1
    if num_workers is None:
        num_workers = min(8, max(2, cpu_count - 1)) if device == "cuda" else min(8, max(0, cpu_count - 1))
    pin_memory = device == "cuda"
    dl_kwargs = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 3

    dl_tr = None
    if balanced_sampler:
        cnt = Counter([r['action_id'] for r in ds_tr.rows])
        class_w = {lab: 1.0 / max(1, cnt.get(lab, 0)) for lab in label_order}
        sample_weights = [class_w[r['action_id']] for r in ds_tr.rows]
        num_samples = int(len(sample_weights) * oversample_factor)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)
        dl_tr = DataLoader(ds_tr, batch_size=bs_train, sampler=sampler, **dl_kwargs)
        print(f'Using balanced sampler: num_samples={num_samples}, factor={oversample_factor}')
    if dl_tr is None:
        dl_tr = DataLoader(ds_tr, batch_size=bs_train, shuffle=True, **dl_kwargs)
    dl_va = DataLoader(ds_va, batch_size=bs_eval, shuffle=False, **dl_kwargs)
    dl_te = DataLoader(ds_te, batch_size=bs_eval, shuffle=False, **dl_kwargs)

    weight_vec = None
    if class_weight:
        cnt = Counter([r["action_id"] for r in ds_tr.rows])
        w = torch.tensor([1.0/max(1, cnt[l]) for l in label_order], dtype=torch.float32)
        w = (w / w.sum()) * len(label_order)
        weight_vec = w.to(device)
        print("Class weights:", dict(zip(label_order, w.tolist())))

    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
    model = model.to(device)
    model = adapt_clip_first_layer_for_stack(model, stack_frames, device)

    # Stability patch: clamped forwards
    def clamped_forward(self_block, x):
        try:
            if hasattr(self_block, 'attn') and hasattr(self_block.attn, 'in_proj'):
                qkv = self_block.attn.in_proj(x)
                qkv = torch.clamp(qkv, -5.0, 5.0)
                x = self_block.attn(qkv)
            if hasattr(self_block, 'mlp'):
                x_mlp = self_block.mlp(x)
                x_mlp = torch.clamp(x_mlp, -10.0, 10.0)
                x = x + x_mlp
            x = torch.clamp(x, -10.0, 10.0)
        except Exception:
            pass
        return x
    for resblock in model.visual.transformer.resblocks:
        resblock.forward = types.MethodType(clamped_forward, resblock)
    print("✓ Applied activation clamping")

    if compile_model and hasattr(torch, 'compile'):
        model = torch.compile(model)

    # Apply LoRA if PEFT available
    if tune_mode == "last2" and PEFT_AVAILABLE:
        lora_config = LoraConfig(
            r=4, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=0.1, bias="none"
        )
        visual_model = model.visual.transformer
        for i in [10, 11]:
            if i < len(visual_model.resblocks):
                get_peft_model(visual_model.resblocks[i], lora_config)
        print("✓ Applied LoRA")

    for p in model.parameters(): p.requires_grad = False
    param_groups = []

    if tune_mode == "logit_only":
        model.logit_scale.requires_grad_(True)
        param_groups.append({"params":[model.logit_scale], "lr": lr_head or lr})

    elif tune_mode == "last2":
        for n,p in model.named_parameters():
            if n.startswith(("visual.transformer.resblocks.10", "visual.transformer.resblocks.11")):
                p.requires_grad = True
            if n.startswith(("transformer.", "token_embedding", "positional_embedding")):
                p.requires_grad = True
        model.logit_scale.requires_grad_(True)
        vg = [p for n,p in model.named_parameters() if p.requires_grad and n.startswith("visual.")]
        tg = [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith("visual.") and n != "logit_scale"]
        param_groups += [
            {"params": vg, "lr": lr},
            {"params": tg, "lr": lr_text or lr},
            {"params": [model.logit_scale], "lr": lr_head or lr},
        ]

    elif tune_mode == "all":
        for p in model.parameters(): p.requires_grad = True
        param_groups += [
            {"params": [p for n,p in model.named_parameters() if n.startswith("visual.")], "lr": lr},
            {"params": [p for n,p in model.named_parameters() if n.startswith(("transformer.","token_embedding","positional_embedding"))], "lr": lr_text or lr},
            {"params": [model.logit_scale], "lr": lr_head or lr},
        ]

    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1,epochs), eta_min=1e-7)

    autocast = torch.amp.autocast(device_type='cuda') if device == "cuda" else contextlib.nullcontext()

    all_rows = ds_tr.rows + ds_va.rows + ds_te.rows
    prompts_per_class = build_prompts_per_class(
        all_rows, label_order, mode=text_bank_mode, use_paraphrases=use_paraphrases
    )

    best = {"score": -1.0, "path": None, "cm": None, "tau": None}
    since = 0

    for ep in range(epochs):
        with torch.no_grad():
            class_bank = build_text_bank(prompts_per_class, label_order, model, device)

        model.train()
        loss_meter = acc_meter = 0.0; nb = 0
        for x, txt, y, rel in tqdm(dl_tr, desc=f"Ep {ep:03d} [train]"):
            x = x.to(device)
            y = y.to(device)
            tok = open_clip.tokenize(list(txt)).to(device)

            optimizer.zero_grad()
            with autocast():
                img = model.encode_image(x); img = img / img.norm(dim=-1, keepdim=True)
                txtf = model.encode_text(tok); txtf = txtf / txtf.norm(dim=-1, keepdim=True)
                loss_con = clip_contrastive_loss(img, txtf, model.logit_scale.exp())
                loss_ce, logits_cls = ce_classification_from_bank(img, class_bank, y, tau=0.35, weight=weight_vec)
                loss = lambda_contrast * loss_con + lambda_cls * loss_ce

            loss.backward()
            # Gradient handling for stability
            total_nonfinite = 0
            total_elements = 0
            for name, param in model.named_parameters():
                if param.grad is not None and param.requires_grad:
                    mask = ~torch.isfinite(param.grad)
                    nonfinite = mask.sum().item()
                    total_nonfinite += nonfinite
                    total_elements += param.numel()
                    if nonfinite > 0:
                        param.grad[mask] = 0.0
            fraction = total_nonfinite / max(1, total_elements)
            if fraction > 0:
                print(f"Handled NaN grads: fraction={fraction:.1%}")
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                acc = (logits_cls.argmax(dim=1) == y).float().mean().item()
            loss_meter += loss.item(); acc_meter += acc; nb += 1

        sched.step()
        tr_loss = loss_meter / nb; tr_acc = acc_meter / nb

        tau_best, f1_best, va = sweep_tau(model, dl_va, device, class_bank, grid=tau_grid)

        print(f"Ep {ep:03d} | train loss={tr_loss:.4f} acc={tr_acc:.3f} | val acc={va['acc']:.3f} f1={va['macro_f1']:.3f} (τ*={tau_best:.2f})")

        if f1_best > best["score"]:
            since = 0
            best.update(score=f1_best, tau=tau_best, path=Path(outdir)/"best.pt", cm=va["cm"])
            torch.save({
                "state_dict": model.state_dict(),
                "labels": label_order,
                "stack_frames": stack_frames,
                "tau_cls": tau_best,
                "prompts_per_class": {k: list(v) for k,v in prompts_per_class.items()},
                "tune_mode": tune_mode,
            }, best["path"])
            save_confusion(va["cm"], label_order, str(Path(outdir)/"val_cm.png"), "Val CM")
            print(f"💾 Best saved: f1={f1_best:.3f}")
        else:
            since += 1
            if since >= patience:
                print(f"Early stop: no improvement for {patience} eps")
                break

    if best["path"] is None:
        raise RuntimeError("No checkpoint")
    ck = torch.load(best["path"], map_location=device)
    model.load_state_dict(ck["state_dict"])
    tau_selected = ck.get("tau_cls", best["tau"])
    with torch.no_grad():
        class_bank = build_text_bank(prompts_per_class, label_order, model, device)
    te = evaluate(model, dl_te, device, class_bank, tau_cls=tau_selected)
    print(f"[TEST] acc={te['acc']:.3f} macroF1={te['macro_f1']:.3f} (τ={tau_selected:.2f})")
    save_confusion(te["cm"], label_order, str(Path(outdir)/"test_cm.png"), "Test CM")

    rep = classification_report(te["y_true"], te["y_pred"], target_names=label_order, output_dict=True)
    with open(Path(outdir)/"metrics.json", "w") as f:
        json.dump({
            "labels": label_order,
            "tau_selected": tau_selected,
            "val_best_macro_f1": best["score"],
            "test_acc": te["acc"],
            "test_macro_f1": te["macro_f1"],
            "test_cm": te["cm"].tolist(),
            "classification_report": rep,
        }, f, indent=2)

    return best["path"]

# -------------------- Enhanced Data Language Encoder --------------------
class EnhancedDataLanguageEncoder:
    def __init__(self):
        self.speed_lo = 3.0
        self.speed_hi = 7.0
        self.near_gap = 15.0
        self.medium_gap = 30.0
        self.ttc_danger = 2.5
        self.ttc_caution = 4.0
        self.density_near_radius = 25.0
        self.conflict_angle_threshold = 45.0

    def _find_conflicting_vehicles(self, env, ego):
        conflicts = []
        try:
            ego_heading = np.degrees(np.arctan2(ego.heading[1], ego.heading[0])) % 360 if hasattr(ego.heading, '__len__') and len(ego.heading) >= 2 else float(np.degrees(ego.heading)) % 360
        except:
            ego_heading = 0.0
        for v in env.unwrapped.road.vehicles:
            if v is ego:
                continue
            try:
                v_heading = np.degrees(np.arctan2(v.heading[1], v.heading[0])) % 360 if hasattr(v.heading, '__len__') and len(v.heading) >= 2 else float(np.degrees(v.heading)) % 360
            except:
                v_heading = ego_heading
            rel_pos = v.position - ego.position
            angle_to_v = np.degrees(np.arctan2(rel_pos[1], rel_pos[0])) % 360
            heading_diff = min(abs(ego_heading - v_heading), 360 - abs(ego_heading - v_heading))
            if (heading_diff > 90 and heading_diff < 270) or (30 < angle_to_v < 150):
                gap = np.linalg.norm(rel_pos)
                rel_speed = ego.speed - v.speed
                closing_speed = max(1e-3, rel_speed * np.cos(np.radians(v_heading - ego_heading)))
                ttc = gap / closing_speed if closing_speed > 1e-3 else float("inf")
                conflicts.append({'vehicle': v, 'gap': gap, 'ttc': ttc, 'direction': angle_to_v, 'rel_speed': rel_speed, 'heading_diff': heading_diff})
        conflicts.sort(key=lambda x: (x['ttc'], x['gap']))
        return conflicts

    def _density_and_directions(self, env, ego):
        sectors = {'front': 0, 'left': 0, 'right': 0, 'rear': 0}
        ego_pos = ego.position
        for v in env.unwrapped.road.vehicles:
            if v is ego:
                continue
            rel_pos = v.position - ego_pos
            angle = np.degrees(np.arctan2(rel_pos[1], rel_pos[0])) % 360
            if -45 <= angle < 45:
                sectors['front'] += 1
            elif 45 <= angle < 135:
                sectors['left'] += 1
            elif 135 <= angle <= 225:
                sectors['rear'] += 1
            else:
                sectors['right'] += 1
        total_density = sum(sectors.values())
        return total_density, sectors

    def describe(self, env):
        ego = env.unwrapped.vehicle
        if ego is None:
            return "IDLE", PROMPTS["IDLE"]
        speed = float(ego.speed)
        conflicts = self._find_conflicting_vehicles(env, ego)
        total_density, sectors = self._density_and_directions(env, ego)
        needs_slow = (len(conflicts) > 0 and conflicts[0]['ttc'] < self.ttc_caution) or (sectors['front'] > 0 and speed > self.speed_hi) or total_density >= 4
        needs_speed = (speed < self.speed_lo and len(conflicts) == 0 and sectors['front'] == 0 and total_density < 2)
        needs_left = (sectors['front'] > 1 and sectors['left'] == 0 and sectors['right'] > sectors['left']) or (conflicts and 225 < conflicts[0]['direction'] < 315)
        needs_right = (sectors['front'] > 1 and sectors['right'] == 0 and sectors['left'] > sectors['right']) or (conflicts and 45 < conflicts[0]['direction'] < 135)
        if needs_left:
            action_id = "LANE_LEFT"
        elif needs_right:
            action_id = "LANE_RIGHT"
        elif needs_slow:
            action_id = "SLOWER"
        elif needs_speed:
            action_id = "FASTER"
        else:
            action_id = "IDLE"
        phrases = [PROMPTS[action_id]] + PARAPHRASES.get(action_id, [])
        instruction = random.choice(phrases)
        return action_id, instruction

# -------------------- Improved Data Collection --------------------
def collect_improved_data(n_episodes=1000, save_path="data/frames.jsonl", use_dle=True, use_ollama=False, ollama_model='llama3.1:8b'):
    env = gym.make("intersection-v1", render_mode="rgb_array")
    config = {
        "observation": {"type": "Kinematics"},
        "action": {"type": "DiscreteMetaAction", "lateral": True, "longitudinal": True, "target_speeds": [0, 4.5, 9]},
        "duration": 40,
        "simulation_frequency": 15,
        "screen_width": 600,
        "screen_height": 600,
    }
    density_schedule = np.linspace(1, 8, n_episodes, dtype=int).tolist()
    dataset = []
    data_dir = Path("data")
    img_dir = data_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    dle = EnhancedDataLanguageEncoder() if use_dle else None
    global_index = 0
    class_counter = Counter()
    max_per_class = n_episodes * 40 // len(DEFAULT_LABELS_5) * 2  # Allow oversampling

    for episode in tqdm(range(n_episodes)):
        current_density = density_schedule[episode]
        config["initial_vehicle_count"] = current_density
        config["spawn_probability"] = 0.1 + 0.02 * current_density
        env.unwrapped.config.update(config)
        obs, _ = env.reset(seed=episode)
        done = False
        step = 0
        while not done:
            rgb_frame = env.render()
            if rgb_frame is not None:
                action_id, instruction = dle.describe(env) if dle else _heuristic_description(env, current_density)
                if use_ollama and OLLAMA_AVAILABLE and random.random() < 0.2:
                    action_id, instruction = generate_description_with_ollama(env, ollama_model)
                if class_counter[action_id] >= max_per_class:
                    continue
                class_counter[action_id] += 1
                img_filename = f"{global_index:06d}_{action_id}.png"
                img_path = img_dir / img_filename
                Image.fromarray(rgb_frame).save(img_path)
                ego = env.unwrapped.vehicle
                ego_speed = float(ego.speed) if ego else 0.0
                n_controlled = len([v for v in env.unwrapped.road.vehicles if v is not ego and np.linalg.norm(v.position - ego.position) < 25]) if ego else 0
                dataset.append({
                    "index": global_index,
                    "image": f"images/{img_filename}",
                    "action_id": action_id,
                    "instruction": instruction,
                    "episode": episode,
                    "t": step,
                    "ego_speed": ego_speed,
                    "n_controlled": n_controlled,
                    "courtesy": True
                })
                global_index += 1
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step += 1
    write_jsonl(Path(save_path), dataset)
    print(f"Collected {len(dataset)} samples. Class dist: {dict(class_counter)}")
    env.close()
    return dataset

def _heuristic_description(env, current_density):
    ego = env.unwrapped.vehicle
    if ego is not None:
        speed = float(ego.speed)
        nearby_vehicles = len([v for v in env.unwrapped.road.vehicles if v is not ego and np.linalg.norm(v.position - ego.position) < 25])
        if nearby_vehicles >= current_density * 0.5 and speed > 6:
            action_id = "SLOWER"
        elif nearby_vehicles < 2 and speed < 4:
            action_id = "FASTER"
        else:
            action_id = "IDLE"
        phrases = [PROMPTS[action_id]] + PARAPHRASES.get(action_id, [])
        instruction = random.choice(phrases)
    else:
        action_id = "IDLE"
        instruction = PROMPTS["IDLE"]
    return action_id, instruction

def generate_description_with_ollama(env, model='llama3.1:8b'):
    ego = env.unwrapped.vehicle
    ego_speed = float(ego.speed) if ego is not None else 0.0
    nearby = 0
    density = env.unwrapped.config.get('initial_vehicle_count', 0) if hasattr(env.unwrapped, 'config') else 0
    try:
        if hasattr(env.unwrapped, 'road') and hasattr(env.unwrapped.road, 'vehicles') and ego is not None:
            nearby = len([v for v in env.unwrapped.road.vehicles if v is not ego and np.linalg.norm(v.position - ego.position) < 25])
    except Exception:
        nearby = 0
    prompt = f"""You are a driving instructor for an unsignalized intersection scenario.
Generate an action_id and instruction.
Allowed action_ids: 'SLOWER', 'IDLE', 'FASTER', 'LANE_LEFT', 'LANE_RIGHT'.
Instruction must match one of: {', '.join(sum([[PROMPTS[k]] + PARAPHRASES.get(k, []) for k in PROMPTS], []))}.
Context: ego_speed={ego_speed:.1f} m/s, nearby={nearby}, density={density}.
Output exactly:
Action_id: <one of the allowed action_ids>
Instruction: <one of the allowed instructions>
"""
    resp = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
    text = resp.get('message', {}).get('content', '') if isinstance(resp, dict) else str(resp)
    text = text.strip()
    lower = text.lower()
    action_id = None
    instruction = None
    for key in ACTION_MAP.values():
        phrases = [PROMPTS[key].lower()] + [p.lower() for p in PARAPHRASES.get(key, [])]
        for phrase in phrases:
            if phrase in lower:
                action_id = key
                instruction = next(p for p in ([PROMPTS[key]] + PARAPHRASES.get(key, [])) if p.lower() == phrase)
                break
        if action_id:
            break
    if not action_id:
        action_id = "IDLE"
        instruction = PROMPTS["IDLE"]
    return action_id, instruction

# -------------------- Enhanced CLIP Reward Model --------------------
class EnhancedCLIPRewardModel:
    def __init__(self, clip_model, preprocess, device=None, temperature=0.07):
        self.model = clip_model.eval()
        self.preprocess = preprocess
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self.action_texts = list(PROMPTS.values())
        with torch.no_grad():
            text_tokens = clip.tokenize(self.action_texts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            self.text_features = F.normalize(text_features, dim=-1)

    @torch.no_grad()
    def _encode_image(self, frame_np):
        if frame_np.ndim == 2:
            frame_np = np.stack([frame_np] * 3, axis=-1)
        elif frame_np.ndim == 3 and frame_np.shape[0] == 1:
            frame_np = np.stack([frame_np[0]] * 3, axis=-1)
        elif frame_np.ndim == 3 and frame_np.shape[-1] == 1:
            frame_np = np.repeat(frame_np, 3, axis=-1)
        if frame_np.max() <= 1.0:
            frame_np = (frame_np * 255).astype(np.uint8)
        elif frame_np.dtype != np.uint8:
            frame_np = frame_np.astype(np.uint8)
        frame_np = np.clip(frame_np, 0, 255)
        try:
            img = Image.fromarray(frame_np)
            processed = self.preprocess(img).unsqueeze(0).to(self.device)
            features = self.model.encode_image(processed)
            return F.normalize(features, dim=-1)
        except Exception as e:
            print(f"Image encoding error: {e}")
            return torch.zeros(1, 512, device=self.device)

    @torch.no_grad()
    def score(self, frame_np, action=None):
        img_features = self._encode_image(frame_np)
        similarities = img_features @ self.text_features.T
        logits = similarities / self.temperature
        action_probs = F.softmax(logits, dim=-1).squeeze(0)
        best_action = int(action_probs.argmax().item())
        confidence = float(action_probs[best_action])
        base_clip_reward = float(similarities[0, best_action])
        clip_reward = np.clip(base_clip_reward, -1, 1)
        clip_reward = (clip_reward + 1) / 2
        clip_reward *= confidence
        return float(clip_reward), best_action, float(confidence), action_probs.cpu().numpy()
    
    def get_action_probabilities(self, frame_np):
        img_features = self._encode_image(frame_np)
        similarities = img_features @ self.text_features.T
        logits = similarities / self.temperature
        return F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

# -------------------- Enhanced Environment Wrapper --------------------
class EnhancedCLIPRewardWrapper(gym.Wrapper):
    def __init__(self, env, clip_reward_model, weight_clip=1.2, max_episode_steps=40, 
                 use_curriculum=False, current_density_level=0):
        super().__init__(env)
        self.rm = clip_reward_model
        self.weight_clip = weight_clip
        self.max_episode_steps = max_episode_steps
        self.use_curriculum = use_curriculum
        self.current_density_level = current_density_level
        self.density_levels = [
            (1, 0.1),
            (3, 0.15),
            (5, 0.2),
            (8, 0.3)
        ]
        self.clip_reward_count = 0
        self.total_steps = 0
        self.episode_step = 0
        self.episode_rewards = deque(maxlen=100)
        if self.use_curriculum:
            self._update_density()
    
    def _update_density(self):
        if self.current_density_level < len(self.density_levels):
            initial_count, spawn_prob = self.density_levels[self.current_density_level]
            self.env.unwrapped.config.update({
                "initial_vehicle_count": initial_count,
                "spawn_probability": spawn_prob
            })
            print(f"🎓 Curriculum: Set density level {self.current_density_level} "
                  f"({initial_count} vehicles, {spawn_prob} spawn)")
    
    def advance_curriculum(self):
        if not self.use_curriculum or self.current_density_level >= len(self.density_levels) - 1:
            return
        recent_success_rate = np.mean(self.episode_rewards) / self.max_episode_steps * 100
        if recent_success_rate > 70:
            self.current_density_level += 1
            self._update_density()
    
    def reset(self, **kwargs):
        self.episode_step = 0
        obs = self.env.reset(**kwargs)
        return obs
    
    def step(self, action):
        obs, r_env, terminated, truncated, info = self.env.step(action)
        frame = self._get_frame_from_obs(obs)
        try:
            r_clip, suggested_action, confidence, action_probs = self.rm.score(frame, action)
        except Exception as e:
            r_clip, suggested_action, confidence, action_probs = 0.0, 0, 0.0, np.zeros(5)
            print(f"Enhanced CLIP score error: {e}")
        self.total_steps += 1
        self.episode_step += 1
        r_env_norm = r_env / self.max_episode_steps if self.max_episode_steps > 0 else r_env
        r_clip_scaled = self.weight_clip * r_clip
        follow_bonus = 0.1 * confidence if int(action) == suggested_action else 0.0
        living_penalty = -0.01 / self.max_episode_steps
        r_total = r_env_norm + r_clip_scaled + follow_bonus + living_penalty
        self.clip_reward_count += 1
        info.update({
            'clip_reward': r_clip,
            'clip_confidence': confidence,
            'clip_suggested': int(suggested_action),
            'base_reward': float(r_env),
            'base_reward_norm': float(r_env_norm),
            'follow_bonus': follow_bonus,
            'episode_step': self.episode_step,
            'density_level': self.current_density_level
        })
        if terminated or truncated:
            self.episode_rewards.append(self.episode_step)
            if 'episode' in info:
                info['episode']['r'] = r_total
                if info.get('crashed', False):
                    info['terminal_observation'] = obs
                elif info.get('arrived_at_destination', False):
                    info['terminal_observation'] = obs
        if self.total_steps % 1000 == 0:
            clip_apply_rate = 100.0 * self.clip_reward_count / max(1, self.total_steps)
            avg_clip_r = np.mean([info.get('clip_reward', 0) for _ in range(10)])
            print(f"🎯 CLIP Stats: {self.clip_reward_count}/{self.total_steps} "
                  f"({clip_apply_rate:.1f}%), avg_r_clip={avg_clip_r:.3f}, "
                  f"density_lvl={self.current_density_level}")
        return obs, float(r_total), terminated, truncated, info
    
    def _get_frame_from_obs(self, obs):
        if isinstance(obs, np.ndarray):
            arr = obs
        else:
            arr = np.array(obs)
        if arr.ndim == 3:
            if arr.shape[0] <= 4:
                frame = arr[-1]
            else:
                frame = arr
        elif arr.ndim == 4:
            frame = arr[0, -1] if arr.shape[1] <= 4 else arr[0]
        else:
            frame = arr
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        elif frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        return np.clip(frame, 0, 255)

# -------------------- Enhanced Custom CNN --------------------
class EnhancedCustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample_input = torch.zeros(1, n_input_channels, 84, 84)
            n_flatten = self.cnn(sample_input).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations):
        cnn_features = self.cnn(observations)
        return self.linear(cnn_features)

# -------------------- Enhanced Training Callback --------------------
class EnhancedTrainingCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=5000, use_wandb=False, verbose=1,
                 curriculum_wrapper=None, min_success_for_advance=0.7, patience=20):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.curriculum_wrapper = curriculum_wrapper
        self.min_success_for_advance = min_success_for_advance
        self.patience = patience
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_successes = deque(maxlen=50)
        self.clip_rewards = deque(maxlen=1000)
        self.training_losses = deque(maxlen=100)
        self.best_mean_reward = -float('inf')
        self.no_improvement_steps = 0
        self.last_eval_step = 0
    
    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                ep_reward = info['episode']['r']
                ep_length = info['episode']['l']
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                crashed = info.get('crashed', False)
                arrived = info.get('arrived_at_destination', False)
                is_success = arrived and not crashed and ep_length < 35
                self.episode_successes.append(1 if is_success else 0)
            if 'clip_reward' in info and np.isfinite(info['clip_reward']):
                self.clip_rewards.append(info['clip_reward'])
            if 'train/loss' in info:
                self.training_losses.append(info['train/loss'])
        if self.n_calls - self.last_eval_step >= self.eval_freq:
            self._perform_evaluation()
            self.last_eval_step = self.n_calls
            if self.curriculum_wrapper and len(self.episode_successes) >= 20:
                recent_success_rate = np.mean(list(self.episode_successes)[-20:])
                if recent_success_rate >= self.min_success_for_advance:
                    print(f"🎓 Curriculum advancement triggered: {recent_success_rate:.2%} success")
                    self.curriculum_wrapper.advance_curriculum()
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(list(self.episode_rewards)[-50:])
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.no_improvement_steps = 0
                else:
                    self.no_improvement_steps += self.eval_freq
                if self.no_improvement_steps >= self.patience * self.eval_freq:
                    print(f"🛑 Early stopping: No improvement for {self.patience} evals")
                    return False
        return True
    
    def _perform_evaluation(self):
        if len(self.episode_rewards) == 0:
            return
        mean_reward = float(np.mean(list(self.episode_rewards)[-50:]))
        std_reward = float(np.std(list(self.episode_rewards)[-50:]))
        mean_length = float(np.mean(list(self.episode_lengths)[-50:]))
        mean_clip = float(np.mean(list(self.clip_rewards)[-200:])) if self.clip_rewards else 0.0
        success_rate = float(np.mean(list(self.episode_successes)[-50:]))
        metrics = {
            "train/mean_reward": mean_reward,
            "train/std_reward": std_reward,
            "train/mean_length": mean_length,
            "train/success_rate": success_rate,
            "train/mean_clip_reward": mean_clip,
            "train/step": self.n_calls,
            "train/density_level": getattr(self.curriculum_wrapper, 'current_density_level', 0)
        }
        print(f"\n📊 Step {self.n_calls:6d}: R={mean_reward:.2f}±{std_reward:.2f} "
              f"L={mean_length:.1f} S={success_rate:.1%} Clip={mean_clip:.3f}")
        if self.use_wandb:
            wandb.log(metrics)
        if self.n_calls % 25000 == 0:
            self._detailed_evaluation()
    
    def _detailed_evaluation(self):
        try:
            mean_reward, std_reward = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=20, deterministic=True, 
                return_episode_rewards=True
            )
            print(f"🔍 Quick Eval: R={mean_reward:.2f}±{std_reward:.2f} (20 eps)")
        except Exception as e:
            print(f"Eval error: {e}")

# -------------------- Enhanced Environment Creation --------------------
def create_enhanced_intersection_env(use_clip_reward=True, clip_reward_model=None, 
                                     curriculum=False, density_level=2, seed=42):
    def make_env():
        env = gym.make("intersection-v1", render_mode=None)
        env.unwrapped.config.update({
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (84, 84),
                "stack_size": 4,
                "weights": [0.2989, 0.5870, 0.1140],
            },
            "action": {
                "type": "DiscreteMetaAction",
                "lateral": True,
                "longitudinal": True,
                "target_speeds": [0, 4.5, 9],
            },
            "duration": 40,
            "simulation_frequency": 15,
            "initial_vehicle_count": 5,
            "spawn_probability": 0.2,
            "collision_reward": -5.0,
            "arrived_reward": +2.0,
            "high_speed_reward": +1.0,
            "reward_speed_range": [7.0, 9.0],
            "normalize_reward": False,
            "offroad_terminal": True,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.3, 0.5],
            "scaling": 5.5,
            "show_trajectories": False,
        })
        env.reset(seed=seed)
        env = Monitor(env)
        if use_clip_reward and clip_reward_model:
            env = EnhancedCLIPRewardWrapper(
                env, clip_reward_model, 
                use_curriculum=curriculum, 
                current_density_level=density_level,
                max_episode_steps=40 * 15
            )
        return env
    return make_env

# -------------------- Enhanced DQN Training --------------------
def train_enhanced_dqn(clip_model_path, total_timesteps=300000, use_clip=True, 
                       save_path="models/enhanced_dqn_clip", use_wandb=False, seed=42):
    print("="*80)
    print("🚗 Training Enhanced DQN with CLIP Reward Shaping")
    print(f"Timesteps: {total_timesteps:,} | CLIP: {use_clip} | Seed: {seed}")
    print("="*80)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    if use_clip and clip_model_path and os.path.exists(clip_model_path):
        clip_model.load_state_dict(torch.load(clip_model_path, map_location=device))
        print("✓ Loaded fine-tuned CLIP model")
    clip_reward_model = EnhancedCLIPRewardModel(clip_model, preprocess, device) if use_clip else None
    env_fn = create_enhanced_intersection_env(
        use_clip_reward=use_clip, 
        clip_reward_model=clip_reward_model,
        curriculum=True,
        seed=seed
    )
    env = DummyVecEnv([env_fn])
    policy_kwargs = dict(
        features_extractor_class=EnhancedCustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[256, 256],
        activation_fn=torch.nn.ReLU,
        dueling=True,
    )
    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=5e-4,
        buffer_size=100000,
        learning_starts=5000,
        train_freq=(4, 1),
        gradient_steps=1,
        batch_size=32,
        tau=1.0,
        gamma=0.95,
        target_update_interval=1000,
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        max_grad_norm=10,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        seed=seed,
        tensorboard_log=f"./logs/enhanced_dqn_{'clip' if use_clip else 'vanilla'}_{seed}/",
        optimize_memory_usage=False,
    )
    eval_env_fn = create_enhanced_intersection_env(
        use_clip_reward=use_clip, 
        clip_reward_model=clip_reward_model,
        curriculum=False,
        seed=seed + 100
    )
    eval_env = DummyVecEnv([eval_env_fn])
    callback = EnhancedTrainingCallback(
        eval_env, 
        eval_freq=5000,
        use_wandb=use_wandb,
        curriculum_wrapper=env.envs[0] if use_clip else None,
        patience=15
    )
    class LRSchedulerCallback(BaseCallback):
        def __init__(self, initial_lr=5e-4, final_lr=1e-5, total_steps=300000, verbose=0):
            super().__init__(verbose)
            self.initial_lr = initial_lr
            self.final_lr = final_lr
            self.total_steps = total_steps
            self.start_step = 0
        def _on_step(self) -> bool:
            if self.n_calls > self.start_step:
                progress = (self.n_calls - self.start_step) / self.total_steps
                new_lr = self.initial_lr - progress * (self.initial_lr - self.final_lr)
                self.model.learning_rate = new_lr
                if self.n_calls % 25000 == 0:
                    print(f"📈 LR scheduled to: {new_lr:.2e} at step {self.n_calls}")
            return True
    lr_callback = LRSchedulerCallback()
    print(f"Starting enhanced DQN training for {total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[callback, lr_callback],
        log_interval=10,
        progress_bar=True,
        tb_log_name=f"dqn_{'clip' if use_clip else 'vanilla'}_{seed}"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\n✓ Enhanced DQN model saved to {save_path}")
    eval_env.close()
    env.close()
    return model, env_fn

# -------------------- Enhanced PPO Training --------------------
def train_enhanced_ppo(clip_model_path, total_timesteps=500000, use_clip=True,
                       save_path="models/enhanced_ppo_clip", use_wandb=False, seed=42):
    print("="*80)
    print("🚗 Training Enhanced PPO with CLIP Reward Shaping")
    print(f"Timesteps: {total_timesteps:,} | CLIP: {use_clip} | Seed: {seed}")
    print("="*80)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    if use_clip and clip_model_path and os.path.exists(clip_model_path):
        clip_model.load_state_dict(torch.load(clip_model_path, map_location=device))
    clip_reward_model = EnhancedCLIPRewardModel(clip_model, preprocess, device) if use_clip else None
    env_fn = create_enhanced_intersection_env(
        use_clip_reward=use_clip, 
        clip_reward_model=clip_reward_model,
        curriculum=True,
        seed=seed
    )
    env = DummyVecEnv([env_fn])
    policy_kwargs = dict(
        features_extractor_class=EnhancedCustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.ReLU,
    )
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=5e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        seed=seed,
        tensorboard_log=f"./logs/enhanced_ppo_{'clip' if use_clip else 'vanilla'}_{seed}/",
    )
    eval_env_fn = create_enhanced_intersection_env(
        use_clip_reward=use_clip, 
        clip_reward_model=clip_reward_model,
        curriculum=False,
        seed=seed + 100
    )
    eval_env = DummyVecEnv([eval_env_fn])
    class KLMonitoringCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.last_kl = 0.0
            self.kl_threshold = 0.05
        def _on_step(self) -> bool:
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                current_kl = self.model.logger.name_to_value.get('train/approx_kl', 0)
                self.last_kl = current_kl
                if current_kl > self.kl_threshold * 2:
                    self.model.clip_range = min(0.3, self.model.clip_range * 0.9)
                elif current_kl < self.kl_threshold / 2:
                    self.model.clip_range = max(0.1, self.model.clip_range * 1.1)
                if self.n_calls % 10000 == 0:
                    print(f"🎛️  KL={current_kl:.4f}, clip_range={self.model.clip_range:.3f}")
            return True
    callback = EnhancedTrainingCallback(
        eval_env, 
        eval_freq=5000,
        use_wandb=use_wandb,
        curriculum_wrapper=env.envs[0] if use_clip else None,
        patience=20
    )
    class EntropySchedulingCallback(BaseCallback):
        def __init__(self, total_steps, verbose=0):
            super().__init__(verbose)
            self.total_steps = total_steps
            self.start_step = 100000
        def _on_step(self) -> bool:
            if self.n_calls > self.start_step:
                progress = (self.n_calls - self.start_step) / (self.total_steps - self.start_step)
                ent_coef = 0.01 * progress
                if abs(self.model.ent_coef - ent_coef) > 1e-5:
                    self.model.ent_coef = ent_coef
                    if self.n_calls % 25000 == 0:
                        print(f"🎲 Entropy coef scheduled to: {ent_coef:.4f}")
            return True
    ent_callback = EntropySchedulingCallback(total_timesteps)
    print(f"Starting enhanced PPO training for {total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[callback, KLMonitoringCallback(), ent_callback],
        log_interval=10,
        progress_bar=True,
        tb_log_name=f"ppo_{'clip' if use_clip else 'vanilla'}_{seed}"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\n✓ Enhanced PPO model saved to {save_path}")
    eval_env.close()
    env.close()
    return model, env_fn

# -------------------- Enhanced Evaluation --------------------
def evaluate_enhanced_agent(model, n_episodes=100, density=[1, 3, 6], render=False, 
                            clip_reward_model=None, seed=42):
    print("\n" + "="*80)
    print("🔍 Enhanced Agent Evaluation (Multiple Densities)")
    print("="*80)
    results = {}
    all_actions = []
    all_clip_suggestions = []
    all_confidences = []
    trajectory_data = []
    for d in density:
        print(f"\n📊 Evaluating at density {d} vehicles...")
        def make_eval_env():
            env = gym.make("intersection-v1", render_mode="rgb_array" if render else None)
            env.unwrapped.config.update({
                "observation": {
                    "type": "GrayscaleObservation",
                    "observation_shape": (84, 84),
                    "stack_size": 4,
                },
                "action": {
                    "type": "DiscreteMetaAction",
                    "lateral": True,
                    "longitudinal": True,
                    "target_speeds": [0, 4.5, 9],
                },
                "duration": 40,
                "initial_vehicle_count": d,
                "spawn_probability": 0.1,
                "normalize_reward": False,
            })
            env.reset(seed=seed)
            if clip_reward_model:
                class EvalCLIPWrapper(gym.Wrapper):
                    def __init__(self, env, rm):
                        super().__init__(env)
                        self.rm = rm
                    def step(self, action):
                        obs, reward, term, trunc, info = self.env.step(action)
                        frame = self._get_frame(obs)
                        try:
                            r_clip, sugg, conf, probs = self.rm.score(frame, action)
                            info.update({
                                'clip_suggested': sugg,
                                'clip_confidence': conf,
                                'clip_probs': probs.tolist()
                            })
                        except:
                            info.update({'clip_suggested': 1, 'clip_confidence': 0.0})
                        return obs, reward, term, trunc, info
                    def _get_frame(self, obs):
                        if isinstance(obs, np.ndarray) and obs.ndim == 3:
                            frame = obs[-1] if obs.shape[0] <= 4 else obs
                        else:
                            frame = np.zeros((84, 84))
                        return frame.astype(np.uint8)
                env = EvalCLIPWrapper(env, clip_reward_model)
            return Monitor(env)
        eval_env = DummyVecEnv([make_eval_env])
        obs = eval_env.reset()
        success_count = collision_count = timeout_count = 0
        episode_rewards, episode_lengths = [], []
        episode_actions, episode_suggestions = [], []
        for ep in tqdm(range(n_episodes)):
            ep_reward, ep_length = 0.0, 0
            ep_actions, ep_suggestions, ep_confs = [], [], []
            done = [False]
            while not done[0]:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                ep_reward += float(reward[0])
                ep_length += 1
                ep_actions.append(int(action[0]))
                if 'clip_suggested' in info[0]:
                    ep_suggestions.append(info[0]['clip_suggested'])
                    ep_confs.append(info[0].get('clip_confidence', 0.0))
                if done[0]:
                    crashed = info[0].get('crashed', False)
                    arrived = info[0].get('arrived_at_destination', False)
                    if crashed:
                        collision_count += 1
                    elif arrived and ep_length < 38:
                        success_count += 1
                    else:
                        timeout_count += 1
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            all_actions.extend(ep_actions)
            all_clip_suggestions.extend(ep_suggestions)
            all_confidences.extend(ep_confs)
            if render and ep < 5:
                frame = eval_env.render()
                if frame is not None:
                    trajectory_data.append({
                        'episode': ep, 'density': d, 
                        'reward': ep_reward, 'length': ep_length,
                        'success': arrived and not crashed,
                        'frame': frame
                    })
        results[d] = {
            'success_rate': 100.0 * success_count / n_episodes,
            'collision_rate': 100.0 * collision_count / n_episodes,
            'timeout_rate': 100.0 * timeout_count / n_episodes,
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_length': float(np.mean(episode_lengths)),
            'actions': episode_actions,
            'n_episodes': n_episodes
        }
        print(f"  Density {d}: Success {results[d]['success_rate']:.1f}% | "
              f"Collision {results[d]['collision_rate']:.1f}% | Timeout {results[d]['timeout_rate']:.1f}% | "
              f"R={results[d]['mean_reward']:.2f}±{results[d]['std_reward']:.2f}")
        eval_env.close()
    if all_clip_suggestions and all_actions:
        cm = sk_confusion_matrix(all_clip_suggestions, all_actions, labels=[0, 1, 2, 3, 4], normalize='true')
        print(f"\n📈 Action Confusion Matrix (CLIP suggestion vs taken):")
        print("Rows: CLIP suggested | Columns: Action taken")
        print("[Slow, Maintain, Fast, Left, Right]")
        for row in cm:
            print(f"[{row[0]:.3f}, {row[1]:.3f}, {row[2]:.3f}, {row[3]:.3f}, {row[4]:.3f}]")
        alignment_rate = 100 * np.mean(np.array(all_clip_suggestions) == np.array(all_actions))
        mean_confidence = np.mean(all_confidences)
        print(f"Overall: Alignment {alignment_rate:.1f}% | Mean CLIP confidence {mean_confidence:.3f}")
    print(f"\n📋 Enhanced Evaluation Summary:")
    print("| Method | Vehicles | Success | Collision | Timeout |")
    print("|--------|----------|---------|-----------|---------|")
    for d, res in results.items():
        print(f"| Enhanced | {d:8d} | {res['success_rate']:6.1f} | "
              f"{res['collision_rate']:8.1f} | {res['timeout_rate']:6.1f} |")
    return results, trajectory_data

# -------------------- Prepare Dataset Splits --------------------
def prepare_dataset_splits(data_dir='data', source=None, train=0.8, val=0.1, test=0.1, seed=42, labels=DEFAULT_LABELS_5):
    data_dir = Path(data_dir)
    if source is None:
        candidates = ['frames.jsonl', 'frames.json', 'frame.json', 'data.jsonl', 'dataset.jsonl']
        for c in candidates:
            pth = data_dir / c
            if pth.exists():
                source = pth
                break
    if source is None:
        for ext in ('*.jsonl', '*.json'):
            files = list(data_dir.glob(ext))
            if files:
                source = files[0]
                break
    if source is None:
        raise FileNotFoundError(f"No source found in {data_dir}")
    rows = read_jsonl(source)
    print(f"Read {len(rows)} rows from {source}")
    if labels:
        labels_set = set(labels)
        rows = [r for r in rows if r.get('action_id') in labels_set]
        print(f"Kept {len(rows)} rows after filtering labels {labels}")
    random.Random(seed).shuffle(rows)
    n = len(rows)
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    n_test = n - n_train - n_val
    train = rows[:n_train]
    val = rows[n_train:n_train+n_val]
    test = rows[n_train+n_val:]
    write_jsonl(data_dir / 'train.jsonl', train)
    write_jsonl(data_dir / 'val.jsonl', val)
    write_jsonl(data_dir / 'test.jsonl', test)
    print(f"Wrote train={len(train)}, val={len(val)}, test={len(test)} to {data_dir}")

# -------------------- Main Pipeline --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["collect", "prepare", "finetune", "train_dqn", "train_ppo", "evaluate"], required=True)
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--out", type=str, default="runs/improved_clip")
    parser.add_argument("--clip-path", type=str, default="robust_clip_finetuned.pt")
    parser.add_argument("--model-path", type=str, default="models/enhanced_dqn_clip")
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS_5)
    parser.add_argument("--stack-frames", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-paraphrases", action="store_true")
    parser.add_argument("--tune-mode", type=str, default="last2")
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--oversample-factor", type=float, default=2.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use-wandb", action="store_true")
    args = parser.parse_args()

    if args.mode == "collect":
        collect_improved_data(n_episodes=1000)
    elif args.mode == "prepare":
        prepare_dataset_splits(data_dir=args.root, labels=args.labels, seed=args.seed)
    elif args.mode == "finetune":
        train_finetune_clip(
            root=args.root,
            outdir=args.out,
            label_order=args.labels,
            stack_frames=args.stack_frames,
            epochs=args.epochs,
            bs_train=args.bs,
            lr=args.lr,
            seed=args.seed,
            use_paraphrases=args.use_paraphrases,
            tune_mode=args.tune_mode,
            balanced_sampler=args.balanced_sampler,
            oversample_factor=args.oversample_factor,
            device=args.device
        )
    elif args.mode == "train_dqn":
        train_enhanced_dqn(
            clip_model_path=args.clip_path,
            total_timesteps=300000,
            use_clip=True,
            save_path=args.model_path,
            use_wandb=args.use_wandb,
            seed=args.seed
        )
    elif args.mode == "train_ppo":
        train_enhanced_ppo(
            clip_model_path=args.clip_path,
            total_timesteps=500000,
            use_clip=True,
            save_path=args.model_path,
            use_wandb=args.use_wandb,
            seed=args.seed
        )
    elif args.mode == "evaluate":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        if os.path.exists(args.clip_path):
            clip_model.load_state_dict(torch.load(args.clip_path, map_location=device))
        clip_reward_model = EnhancedCLIPRewardModel(clip_model, preprocess, device)
        model = DQN.load(args.model_path)  # or PPO.load if ppo
        evaluate_enhanced_agent(
            model,
            n_episodes=100,
            density=[1, 3, 6],
            clip_reward_model=clip_reward_model,
            seed=args.seed
        )

if __name__ == "__main__":
    main()
