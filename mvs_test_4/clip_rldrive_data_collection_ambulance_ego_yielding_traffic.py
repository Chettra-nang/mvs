#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP-RLDrive — CLIP fine-tuning for 3 actions {SLOWER, IDLE, FASTER}

Input folder layout:
  <root>/
    images/*.png
    train.jsonl  val.jsonl  test.jsonl
Each JSONL row (example):
  {"image":"images/000123_IDLE.png","action_id":"IDLE","instruction":"Maintain your current speed.", ...}

Install:
  pip install torch torchvision open_clip_torch scikit-learn matplotlib tqdm pillow

Typical run (paper-like: 4-frame stack, canonical prompts, no paraphrases):
  python train_clip_finetune_v2.py \
    --root /path/to/ambulance_intersection \
    --out  runs/clip_v1 \
    --labels SLOWER IDLE FASTER \
    --stack-frames 4 \
    --text-bank-mode canonical --no-paraphrases \
    --tune-mode last2 \
    --epochs 50 --bs 64 --lr 1e-5

Notes
- tune-mode:
    logit_only  : freeze encoders, tune only logit_scale  (very stable, low cap)
    last2       : tune last 2 vision blocks + full text encoder + logit_scale (good balance)
    all         : tune everything (may overfit on small data)
- The script selects τ by sweeping on validation; saved in the checkpoint and used for test/inference.
"""

import os, re, json, random, glob, contextlib
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import open_clip

# -------------------- labels & prompts --------------------
DEFAULT_LABELS_3 = ["SLOWER", "IDLE", "FASTER"]

PROMPTS = {
    "SLOWER": "Slow down because there might be a collision.",
    "IDLE":   "Maintain your current speed.",
    "FASTER": "Speed up since the chance of collision is low.",
}

PARAPHRASES = {
    "SLOWER": [
        "Slow down; collision risk.", "Reduce speed—caution ahead.",
        "Decelerate; gap is tight.", "Ease off; possible hazard."
    ],
    "IDLE": [
        "Maintain current speed.", "Hold speed and lane.",
        "Stay steady in this lane.", "Continue unchanged."
    ],
    "FASTER": [
        "Accelerate; safe gap ahead.", "Speed up; path looks clear.",
        "Pick up pace—no blockers.", "Increase speed toward target."
    ],
}

# -------------------- tiny utils --------------------
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def read_jsonl(path: Path) -> List[dict]:
    rows=[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t=line.strip()
            if t: rows.append(json.loads(t))
    return rows

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

# -------------------- dataset --------------------
class FrameTextDataset(Dataset):
    """
    Yields:
      x  : Tensor (3*S, H, W)  with CLIP mean/std
      txt: instruction string
      y  : class id (Long)
      rel: relative image path (str)
    """
    def __init__(self, root: Path, jsonl_name: str, label_order: List[str],
                 stack_frames: int = 4, image_size: int = 224, augment: bool = False):
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

        # CLIP mean/std tensors
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:,None,None]
        self.std  = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:,None,None]

    def __len__(self): return len(self.rows)

    def _open_by_index(self, idx: int) -> Image.Image:
        # Try exact indexed filename pattern first
        p = find_image_by_index(self.images_dir, idx)
        if p is None:
            # try with lower bound
            p = find_image_by_index(self.images_dir, max(0, idx))
        if p is None:
            # try globbing for filenames that contain the zero-padded index
            try:
                hits = list(self.images_dir.glob(f"*{idx:06d}*.png"))
                if hits:
                    p = hits[0]
            except Exception:
                p = None
        if p is None:
            # fallback: find the nearest indexed image by parsing numeric tokens in filenames
            candidates = []
            for f in self.images_dir.glob("*.png"):
                fid = parse_index_from_filename(str(f))
                if fid is not None:
                    candidates.append((abs(fid - idx), f))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                p = candidates[0][1]
        if p is None:
            # No image found; raise a helpful error instead of passing None to PIL
            raise FileNotFoundError(f"No image found for index {idx} under {self.images_dir}")
        return Image.open(p).convert("RGB")

    def _maybe_jitter(self, im: Image.Image) -> Image.Image:
        if not self.augment:
            return im
        # mild brightness/contrast jitter
        if random.random() < 0.7:
            b = 1.0 + np.random.uniform(-0.08, 0.08)
            c = 1.0 + np.random.uniform(-0.08, 0.08)
            im = ImageOps.autocontrast(im)
            im = Image.eval(im, lambda v: int(np.clip(v * c, 0, 255)))
            im = Image.eval(im, lambda v: int(np.clip(v * b, 0, 255)))
        return im

    def _build_stack(self, idx: int) -> torch.Tensor:
        frames = []
        for k in range(self.S-1, -1, -1):  # t-(S-1) ... t
            im = self._open_by_index(max(0, idx - k))
            im = letterbox_square(im, target=self.HW, fill=0)
            im = self._maybe_jitter(im)
            t = torch.from_numpy(np.array(im)).permute(2,0,1).float()/255.0  # (3,H,W)
            t = (t - self.mean) / self.std
            frames.append(t)
        return torch.cat(frames, dim=0)  # (3S,H,W)

    def __getitem__(self, i: int):
        r = self.rows[i]
        rel = r["image"]
        idx = parse_index_from_filename(rel) or r.get("index", i)
        x = self._build_stack(idx)
        text = r.get("instruction", "") or self.labels[self.lab2id[r["action_id"]]]
        y = self.lab2id[r["action_id"]]
        return x, text, torch.tensor(y, dtype=torch.long), rel

# -------------------- CLIP adaptor --------------------
def adapt_clip_first_layer_for_stack(model, stack_frames: int, device: str):
    """
    Inflate visual.conv1 to accept 3*stack_frames input channels by
    tiling pretrained RGB kernels and scaling by 1/stack_frames.
    """
    assert hasattr(model.visual, "conv1"), "This CLIP variant lacks .visual.conv1"
    conv1: nn.Conv2d = model.visual.conv1
    out_ch, in_ch, kh, kw = conv1.weight.shape
    if in_ch == 3 * stack_frames:
        return model
    assert in_ch == 3, f"Expected 3 input channels, got {in_ch}"
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

# -------------------- text bank --------------------
def build_prompts_per_class(rows: List[dict], label_order: List[str],
                            mode: str = "dataset", use_paraphrases: bool = False) -> Dict[str, List[str]]:
    """
    mode: 'dataset' uses JSONL instructions,
          'canonical' uses PROMPTS only,
          'both' combines them.
    """
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
                    label_order: List[str], model, device, max_per_class=64) -> torch.Tensor:
    bank = []
    for lab in label_order:
        texts = prompts_per_class.get(lab, [lab.lower()])[:max_per_class]
        tok = open_clip.tokenize(texts).to(device)
        T = model.encode_text(tok)
        T = T / T.norm(dim=-1, keepdim=True)
        bank.append(T.mean(dim=0, keepdim=True))
    return torch.cat(bank, dim=0)  # (C,D)

# -------------------- losses --------------------
def clip_contrastive_loss(img_feats, txt_feats, logit_scale):
    # symmetric InfoNCE used by CLIP
    logits_per_image = logit_scale * img_feats @ txt_feats.t()  # (B,B)
    logits_per_text  = logits_per_image.t()
    labels = torch.arange(img_feats.size(0), device=img_feats.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text,  labels)
    return 0.5 * (loss_i + loss_t)

def ce_classification_from_bank(img_feats, class_bank, y, tau=0.35, weight=None):
    logits = (img_feats @ class_bank.t()) / float(tau)
    return F.cross_entropy(logits, y, weight=weight), logits

# -------------------- eval helpers --------------------
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

def sweep_tau(model, loader, device, class_bank, grid=(0.25,0.30,0.35,0.40,0.45)):
    best_tau, best_f1, best_metrics = None, -1.0, None
    for t in grid:
        m = evaluate(model, loader, device, class_bank, tau_cls=t)
        if m["macro_f1"] > best_f1:
            best_tau, best_f1, best_metrics = t, m["macro_f1"], m
    return best_tau, best_f1, best_metrics

def save_confusion(cm: np.ndarray, labels: List[str], out_png: str, title: Optional[str]=None):
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i,j])), ha="center", va="center",
                    color=("white" if cm[i,j] > cm.max()*0.5 else "black"))
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    if title: ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

# -------------------- trainer --------------------
def train_finetune_clip(
    root: str, outdir: str,
    train_file="train.jsonl", val_file="val.jsonl", test_file="test.jsonl",
    label_order: List[str] = None,
    stack_frames: int = 4, image_size=224,
    epochs=50, bs_train=64, bs_eval=128,
    lr=1e-5, lr_text=None, lr_head=None, weight_decay=0.05,
    seed=42, tau_grid=(0.25,0.30,0.35,0.40,0.45),
    lambda_contrast=1.0, lambda_cls=1.0,
    text_bank_mode="canonical", use_paraphrases=False,
    tune_mode="last2",        # 'logit_only' | 'last2' | 'all'
    class_weight=True,        # apply class-weighted CE
    patience=10
):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # labels
    if label_order is None:
        labs = sorted(set([r.get("action_id") for r in read_jsonl(Path(root)/train_file)]))
        label_order = labs
    print("Labels:", label_order)

    # datasets / loaders
    ds_tr = FrameTextDataset(Path(root), train_file, label_order, stack_frames, image_size, augment=True)
    ds_va = FrameTextDataset(Path(root), val_file,   label_order, stack_frames, image_size, augment=False)
    ds_te = FrameTextDataset(Path(root), test_file,  label_order, stack_frames, image_size, augment=False)
    dl_tr = DataLoader(ds_tr, batch_size=bs_train, shuffle=True,  num_workers=0, pin_memory=False)
    dl_va = DataLoader(ds_va, batch_size=bs_eval,  shuffle=False, num_workers=0, pin_memory=False)
    dl_te = DataLoader(ds_te, batch_size=bs_eval,  shuffle=False, num_workers=0, pin_memory=False)

    # class weights (optional)
    weight_vec = None
    if class_weight:
        cnt = Counter([r["action_id"] for r in ds_tr.rows])
        w = torch.tensor([1.0/max(1, cnt[l]) for l in label_order], dtype=torch.float32)
        w = (w / w.sum()) * len(label_order)
        weight_vec = w.to(device)
        print("Class weights:", {l: round(float(wi),3) for l,wi in zip(label_order, weight_vec)})

    # model
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
    model.to(device)
    model = adapt_clip_first_layer_for_stack(model, stack_frames, device)

    # ----- choose what to tune -----
    for p in model.parameters(): p.requires_grad = False
    param_groups = []

    if tune_mode == "logit_only":
        model.logit_scale.requires_grad_(True)
        param_groups.append({"params":[model.logit_scale], "lr": lr_head or lr})

    elif tune_mode == "last2":
        # unfreeze last 2 vision blocks + text encoder + logit_scale
        for n,p in model.named_parameters():
            if n.startswith(("visual.transformer.resblocks.10",
                             "visual.transformer.resblocks.11")):
                p.requires_grad = True
            if n.startswith(("transformer.", "token_embedding", "positional_embedding")):
                p.requires_grad = True
        model.logit_scale.requires_grad_(True)
        vg = [p for n,p in model.named_parameters()
              if p.requires_grad and n.startswith("visual.")]
        # exclude visual params and logit_scale from the text/group selection
        tg = [p for n,p in model.named_parameters()
              if p.requires_grad and (not n.startswith("visual.")) and (n != "logit_scale")]
        param_groups += [
            {"params": vg, "lr": lr},
            {"params": tg, "lr": lr_text or lr},
            {"params": [model.logit_scale], "lr": lr_head or lr},
        ]

    elif tune_mode == "all":
        for p in model.parameters(): p.requires_grad = True
        param_groups += [
            {"params": [p for n,p in model.named_parameters() if n.startswith("visual.")], "lr": lr},
            {"params": [p for n,p in model.named_parameters()
                        if n.startswith(("transformer.","token_embedding","positional_embedding"))], "lr": lr_text or lr},
            {"params": [model.logit_scale], "lr": lr_head or lr},
        ]
    else:
        raise ValueError("tune_mode must be one of {'logit_only','last2','all'}")

    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1,epochs), eta_min=1e-6)

    # AMP context (fixes deprecation)
    if device == "cuda":
        def _autocast():
            return torch.amp.autocast('cuda')
    else:
        def _autocast():
            return contextlib.nullcontext()

    # text prompts → class bank (recomputed each epoch because text encoder may change)
    all_rows = ds_tr.rows + ds_va.rows + ds_te.rows
    prompts_per_class = build_prompts_per_class(
        all_rows, label_order, mode=text_bank_mode, use_paraphrases=use_paraphrases
    )

    best = {"score": -1.0, "path": None, "cm": None, "tau": None}
    since = 0

    for ep in range(epochs):
        with torch.no_grad():
            class_bank = build_text_bank(prompts_per_class, label_order, model, device)

        # ---- train ----
        model.train()
        loss_meter = acc_meter = 0.0; nb = 0
        for x, txt, y, rel in tqdm(dl_tr, desc=f"Epoch {ep:02d} [train]"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            tok = open_clip.tokenize(list(txt)).to(device)

            optimizer.zero_grad(set_to_none=True)
            with _autocast():
                img = model.encode_image(x); img = img / img.norm(dim=-1, keepdim=True)
                txtf = model.encode_text(tok); txtf = txtf / txtf.norm(dim=-1, keepdim=True)

                loss_con = clip_contrastive_loss(img, txtf, model.logit_scale.exp())
                loss_ce, logits_cls = ce_classification_from_bank(
                    img, class_bank, y, tau=0.35, weight=weight_vec
                )
                loss = lambda_contrast * loss_con + lambda_cls * loss_ce

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                acc = (logits_cls.argmax(dim=1) == y).float().mean().item()
            loss_meter += float(loss.item()); acc_meter += acc; nb += 1

        sched.step()
        tr_loss = loss_meter/max(1,nb); tr_acc = acc_meter/max(1,nb)

        # ---- val: sweep τ and select best by macro-F1 ----
        with torch.no_grad():
            class_bank = build_text_bank(prompts_per_class, label_order, model, device)
        tau_best, f1_best, va = sweep_tau(model, dl_va, device, class_bank, grid=tau_grid)

        print(f"Epoch {ep:02d} | train loss={tr_loss:.4f} acc={tr_acc:.3f} | "
              f"val acc={va['acc']:.3f} f1={va['macro_f1']:.3f} (τ*={tau_best:.2f}, f1*={f1_best:.3f})")

        if f1_best > best["score"]:
            since = 0
            best.update(score=f1_best, tau=float(tau_best),
                        path=str(Path(outdir)/"clip_finetune_best.pt"),
                        cm=va["cm"])
            torch.save({
                "state_dict": model.state_dict(),
                "labels": label_order,
                "stack_frames": int(stack_frames),
                "tau_cls": float(tau_best),
                # convert prompts_per_class (possibly defaultdict) to plain dict of lists
                "prompts_per_class": {k: list(v) for k,v in prompts_per_class.items()},
                "tune_mode": tune_mode,
            }, best["path"])
            save_confusion(va["cm"], label_order, str(Path(outdir)/"val_confusion.png"), "Validation")
            print(f"💾 saved best → {best['path']}")
        else:
            since += 1
            if since >= patience:
                print(f"[early-stop] no improvement for {patience} epochs.")
                break

    # ---- test ----
    if best["path"] is None:
        raise RuntimeError("No best checkpoint saved.")
    try:
        ck = torch.load(best["path"], map_location=device)
    except Exception:
        # PyTorch 2.6+ may use weights_only=True by default which disallows some globals.
        # Retry loading the full checkpoint (only if file is trusted).
        ck = torch.load(best["path"], map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    tau_selected = ck.get("tau_cls", best["tau"])
    with torch.no_grad():
        class_bank = build_text_bank(prompts_per_class, label_order, model, device)
    te = evaluate(model, dl_te, device, class_bank, tau_cls=tau_selected)
    print(f"[TEST] acc={te['acc']:.3f} macroF1={te['macro_f1']:.3f} (τ={tau_selected:.2f})")
    save_confusion(te["cm"], label_order, str(Path(outdir)/"test_confusion.png"), "Test")

    rep = classification_report(te["y_true"], te["y_pred"],
                                target_names=label_order, output_dict=True)
    with open(Path(outdir)/"metrics.json", "w") as f:
        json.dump({
            "labels": label_order,
            "tau_selected": float(tau_selected),
            "val_best_macro_f1": float(best["score"]),
            "test_acc": float(te["acc"]),
            "test_macro_f1": float(te["macro_f1"]),
            "test_cm": te["cm"].tolist(),
            "classification_report": rep,
            "tune_mode": tune_mode,
            "stack_frames": int(stack_frames),
        }, f, indent=2)

    print("Done. Best checkpoint:", best["path"])
    return best["path"]

# -------------------- CLI --------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True, help="Folder with train/val/test.jsonl and images/")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--labels", nargs="*", default=DEFAULT_LABELS_3)
    p.add_argument("--stack-frames", type=int, default=4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--bs-eval", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr-text", type=float, default=None)
    p.add_argument("--lr-head", type=float, default=None)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tau-grid", nargs="+", type=float, default=[0.25,0.30,0.35,0.40,0.45])
    p.add_argument("--lambda-contrast", type=float, default=1.0)
    p.add_argument("--lambda-cls", type=float, default=1.0)
    p.add_argument("--text-bank-mode", type=str, default="canonical",
                   choices=["dataset","canonical","both"])
    p.add_argument("--use-paraphrases", action="store_true")
    p.add_argument("--tune-mode", type=str, default="last2",
                   choices=["logit_only","last2","all"])
    p.add_argument("--no-class-weight", action="store_true")
    p.add_argument("--patience", type=int, default=10)
    args = p.parse_args()

    train_finetune_clip(
        root=args.root,
        outdir=args.out,
        train_file="train.jsonl",
        val_file="val.jsonl",
        test_file="test.jsonl",
        label_order=args.labels,
        stack_frames=args.stack_frames,
        image_size=args.image_size,
        epochs=args.epochs,
        bs_train=args.bs,
        bs_eval=args.bs_eval,
        lr=args.lr, lr_text=args.lr_text, lr_head=args.lr_head,
        weight_decay=args.wd,
        seed=args.seed,
        tau_grid=tuple(args.tau_grid),
        lambda_contrast=args.lambda_contrast,
        lambda_cls=args.lambda_cls,
        text_bank_mode=args.text_bank_mode,
        use_paraphrases=args.use_paraphrases,
        tune_mode=args.tune_mode,
        class_weight=(not args.no_class_weight),
        patience=args.patience,
    )
