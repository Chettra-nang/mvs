
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
import highway_env  # noqa: F401 - needed to register envs
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import clip
from PIL import Image, ImageEnhance
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False
from tqdm import tqdm
try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False
    LoraConfig = None
    def get_peft_model(*args, **kwargs):
        raise RuntimeError("PEFT not available")
try:
    import ollama
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False
import types
import json
import argparse
from pathlib import Path
import warnings
from collections import deque
import random
from typing import Optional, Dict, Any

warnings.filterwarnings("ignore", category=UserWarning)

# Added from collector.py for prompts and paraphrases
PROMPTS = {
    "SLOWER":     "Reduce speed—collision risk ahead.",
    "IDLE":       "Keep the current lane and speed.",
    "FASTER":     "Accelerate; a safe gap is ahead.",
    "LANE_LEFT":  "Change to the left lane.",
    "LANE_RIGHT": "Change to the right lane."
}
PARAPHRASES = {
    "SLOWER": [
        "Reduce speed—caution ahead.", "Back off the throttle; traffic up ahead.",
        "Slow down to stay safe.", "Drop speed; the gap is tight.", "Decelerate—possible hazard ahead."
    ],
    "IDLE": [
        "Hold speed and lane.", "Stay steady in this lane.", "Maintain pace; no lane change.",
        "Remain in lane with current speed.", "Continue unchanged."
    ],
    "FASTER": [
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

# ============================================================================
# PART 0: Enhanced Data Language Encoder (Extended to 5 actions)
# ============================================================================

class EnhancedDataLanguageEncoder:
    """
    Advanced encoder with directional awareness and richer contextual descriptions.
    Includes relative positions, headings, and conflict detection for better CLIP alignment.
    Extended to recommend among 5 actions including lane changes.
    """
    
    def __init__(self):
        # Tunable thresholds (meters, m/s, seconds)
        self.speed_lo = 3.0
        self.speed_hi = 7.0
        self.near_gap = 15.0
        self.medium_gap = 30.0
        self.ttc_danger = 2.5
        self.ttc_caution = 4.0
        self.density_near_radius = 25.0
        self.conflict_angle_threshold = 45.0  # degrees for crossing paths
        
        # Action phrases (extended to 5, matching collector.py)
        self.ACTIONS = [
            "SLOWER",
            "IDLE",
            "FASTER",
            "LANE_LEFT",
            "LANE_RIGHT"
        ]
    
    def _find_conflicting_vehicles(self, env, ego):
        conflicts = []
        try:
            if hasattr(ego.heading, '__len__') and len(ego.heading) >= 2:
                ego_heading = np.degrees(np.arctan2(ego.heading[1], ego.heading[0])) % 360
            else:
                ego_heading = float(np.degrees(ego.heading)) % 360
        except Exception:
            ego_heading = 0.0
        
        for v in env.unwrapped.road.vehicles:
            if v is ego:
                continue
            try:
                if hasattr(v.heading, '__len__') and len(v.heading) >= 2:
                    v_heading = np.degrees(np.arctan2(v.heading[1], v.heading[0])) % 360
                else:
                    v_heading = float(np.degrees(v.heading)) % 360
            except Exception:
                v_heading = ego_heading
            
            rel_pos = v.position - ego.position
            angle_to_v = np.degrees(np.arctan2(rel_pos[1], rel_pos[0])) % 360
            
            heading_diff = min(abs(ego_heading - v_heading), 360 - abs(ego_heading - v_heading))
            if (heading_diff > 90 and heading_diff < 270) or (30 < angle_to_v < 150):
                gap = np.linalg.norm(rel_pos)
                rel_speed = ego.speed - v.speed
                closing_speed = max(1e-3, rel_speed * np.cos(np.radians(v_heading - ego_heading)))
                ttc = gap / closing_speed if closing_speed > 1e-3 else float("inf")
                
                conflicts.append({
                    'vehicle': v, 'gap': gap, 'ttc': ttc, 'direction': angle_to_v,
                    'rel_speed': rel_speed, 'heading_diff': heading_diff
                })
        
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
        """
        Generate action recommendation and corresponding instruction.
        Returns: (action_id, instruction)
        """
        ego = env.unwrapped.vehicle
        if ego is None:
            return "IDLE", PROMPTS["IDLE"]
        
        speed = float(ego.speed)
        conflicts = self._find_conflicting_vehicles(env, ego)
        total_density, sectors = self._density_and_directions(env, ego)
        
        # Enhanced action recommendation
        needs_slow = (len(conflicts) > 0 and conflicts[0]['ttc'] < self.ttc_caution) or \
                     (sectors['front'] > 0 and speed > self.speed_hi) or total_density >= 4
        needs_speed = (speed < self.speed_lo and len(conflicts) == 0 and 
                       sectors['front'] == 0 and total_density < 2)
        needs_left = (sectors['front'] > 1 and sectors['left'] == 0 and 
                      sectors['right'] > sectors['left']) or \
                     (conflicts and 225 < conflicts[0]['direction'] < 315)
        needs_right = (sectors['front'] > 1 and sectors['right'] == 0 and 
                       sectors['left'] > sectors['right']) or \
                      (conflicts and 45 < conflicts[0]['direction'] < 135)
        
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
        
        # Select random paraphrase for variety
        action_key = action_id
        phrases = [PROMPTS[action_key]] + PARAPHRASES.get(action_key, [])
        instruction = random.choice(phrases)
        
        return action_id, instruction

# ============================================================================
# PART 1: Enhanced CLIP Fine-tuning with Augmentation
# ============================================================================

class AugmentedIntersectionDataset(Dataset):
    """Dataset with on-the-fly augmentation for better generalization."""
    
    def __init__(self, data_path, transform=None, augment_prob=0.5):
        self.data_path = data_path
        self.transform = transform
        self.augment_prob = augment_prob
        # Load dataset file. Support either a single JSON array or JSONL (one JSON object per line).
        p = Path(data_path)
        if not p.exists():
            # try looking in the local data folder next to this script
            alt = Path(__file__).parent / 'data' / p.name
            if alt.exists():
                p = alt
        self.data = []
        try:
            with open(p, 'r', encoding='utf-8') as f:
                try:
                    self.data = json.load(f)
                except json.JSONDecodeError:
                    # Fallback: JSONL (one JSON object per line)
                    f.seek(0)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            self.data.append(json.loads(line))
                        except Exception:
                            # ignore malformed lines
                            continue
        except FileNotFoundError:
            raise
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Resolve image path: try the path as-is, then fall back to data/ directory next to this script
        img_p = Path(item['image'])
        if not img_p.exists():
            alt = Path(__file__).parent / 'data' / item['image']
            if alt.exists():
                img_p = alt
            else:
                # also try joining data/images if dataset used 'images/...' prefix
                alt2 = Path(__file__).parent / 'data' / Path(item['image']).name
                if alt2.exists():
                    img_p = alt2
        image = Image.open(img_p).convert('RGB')
        
        # Augmentation
        if random.random() < self.augment_prob:
            # Random rotation (±15°)
            angle = random.uniform(-15, 15)
            image = image.rotate(angle, resample=Image.BICUBIC)
            
            # Brightness jitter (±0.15)
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(0.85, 1.15)
            image = enhancer.enhance(factor)
        
        if self.transform:
            image = self.transform(image)
        text = item['instruction']  # Use 'instruction' field for CLIP
        return image, text

class RobustCLIPFineTuner:
    """Ultra-Stable: LoRA on visuals, gradual 1-block unfreeze, activation clamping."""

    def __init__(self, model_name="ViT-B/32", device=None, initial_freeze_epochs=15):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.initial_freeze_epochs = initial_freeze_epochs
        self.model_name = model_name
        
        # Freeze all; low scale init
        for p in self.model.parameters():
            p.requires_grad = False
        with torch.no_grad():
            self.model.logit_scale.data = torch.tensor(-4.60517, device=self.device)
            print(f"🔧 Ultra-stable init: scale=-4.605, all frozen")
        
        self.num_trainable = 0
        self.global_step = 0
        self.warmup_steps = 0
        self.base_lr = 5e-4
        self.warmup_lr = 0.0
        self.unfrozen_blocks = []
        self.lora_applied = False
        self.action_descriptions = list(PROMPTS.values())

    def _apply_lora_to_blocks(self, block_indices=[10, 11]):
        if self.lora_applied:
            return
        lora_config = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        visual_model = self.model.visual.transformer
        for i in block_indices:
            if i < len(visual_model.resblocks):
                try:
                    get_peft_model(visual_model.resblocks[i], lora_config)
                except Exception:
                    pass
                for name, module in visual_model.resblocks[i].named_modules():
                    if any(k in name for k in ['q_proj', 'k_proj', 'v_proj', 'out_proj']):
                        try:
                            module.requires_grad_(True)
                        except Exception:
                            pass
                print(f"✓ Applied LoRA to visual block {i}")
        self.lora_applied = True
        self._update_trainable()

    def _unfreeze_stage(self, stage):
        if stage == 0:
            try:
                if hasattr(self.model, 'logit_scale'):
                    self.model.logit_scale.requires_grad_(True)
                if hasattr(self.model, 'text_projection'):
                    self.model.text_projection.requires_grad_(True)
            except Exception:
                pass
            print("✓ Unfroze scale + text_projection (~262k params)")
        elif stage == 1:
            self._apply_lora_to_blocks([11])
            self.unfrozen_blocks = [11]
        elif stage == 2:
            self._apply_lora_to_blocks([10, 11])
            self.unfrozen_blocks = [10, 11]
        self._update_trainable()

    def _update_trainable(self):
        self.num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable params: {self.num_trainable}")

    def _patch_visual_forwards(self):
        if hasattr(self, '_patched'):
            return
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
                try:
                    x = self_block.__class__.forward(self_block, x)
                except Exception:
                    pass
            return x
        for resblock in self.model.visual.transformer.resblocks:
            resblock.forward = types.MethodType(clamped_forward, resblock)
        self._patched = True
        print("✓ Patched visual forwards with activation clamping")

    def get_optimizer(self):
        named = dict(self.model.named_parameters())
        def pick_params_by_keys(keys):
            out = []
            for k in keys:
                for name, p in named.items():
                    if k in name and p is not None and getattr(p, 'requires_grad', False):
                        out.append(p)
            return out
        scale_params = [p for name, p in named.items() if ('logit_scale' in name or name == 'logit_scale') and getattr(p, 'requires_grad', False)]
        proj_params = [p for name, p in named.items() if ('text_projection' in name or 'text_proj' in name) and getattr(p, 'requires_grad', False)]
        lora_params = [p for name, p in named.items() if 'lora' in name.lower() and getattr(p, 'requires_grad', False)] if self.lora_applied else []
        seen = set()
        other_params = []
        for p in proj_params + lora_params:
            if id(p) not in seen:
                seen.add(id(p))
                other_params.append(p)
        other_optim = None
        scale_optim = None
        try:
            if other_params:
                other_optim = torch.optim.AdamW(other_params, lr=self.base_lr, weight_decay=0.01)
            if scale_params:
                scale_optim = torch.optim.SGD(scale_params, lr=max(1e-6, self.base_lr * 0.001))
        except TypeError as e:
            print(f"Optimizer construction error: {e}")
            other_optim, scale_optim = None, None
        return other_optim, scale_optim

    @staticmethod
    def contrastive_loss(logits_per_image, logits_per_text, device, temperature=0.07):
        bsz = logits_per_image.shape[0]
        labels = torch.arange(bsz, device=device)
        logits_i = torch.clamp(logits_per_image / temperature, -15, 15)
        logits_t = torch.clamp(logits_per_text / temperature, -15, 15)
        loss_i = F.cross_entropy(logits_i, labels)
        loss_t = F.cross_entropy(logits_t, labels)
        loss = 0.5 * (loss_i + loss_t)
        if not torch.isfinite(loss):
            return torch.tensor(5.0, device=device, requires_grad=True)
        return loss

    def _safe_encode_image(self, images):
        if not hasattr(self, '_patched'):
            self._patch_visual_forwards()
        features = self.model.encode_image(images)
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        features = torch.clamp(features, -2.0, 2.0)
        features += 1e-6 * torch.randn_like(features)
        if not torch.isfinite(features).all():
            features.zero_()
        return F.normalize(features, dim=-1)

    def _safe_encode_text(self, tokens):
        features = self.model.encode_text(tokens)
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        features = torch.clamp(features, -2.0, 2.0)
        features += 1e-6 * torch.randn_like(features)
        if not torch.isfinite(features).all():
            features.zero_()
        return F.normalize(features, dim=-1)

    def _handle_gradients(self):
        total_nonfinite = 0
        total_elements = 0
        details = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None and param.requires_grad:
                mask = ~torch.isfinite(param.grad)
                nonfinite = mask.sum().item()
                total_nonfinite += nonfinite
                total_elements += param.numel()
                if nonfinite > 0:
                    param.grad[mask] = 0.0
                    if nonfinite / param.numel() > 0.5:
                        details[name] = f"{nonfinite}/{param.numel()} ({nonfinite/param.numel()*100:.1f}%)"
        fraction = total_nonfinite / max(1, total_elements)
        if details:
            print(f"⚠️ Handled NaN grads (fraction={fraction:.1%}): {details}")
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.01)
        return fraction < 0.1

    def train(self, dataset_path, epochs=15, batch_size=2, save_path="stable_clip_lora.pt", 
              weight_decay=0.1, max_skip_rate=0.2):
        dataset = AugmentedIntersectionDataset(dataset_path, transform=self.preprocess, augment_prob=0.2)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        
        other_optim, scale_optim = self.get_optimizer()
        scheduler_other = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(other_optim, T_0=5, eta_min=self.base_lr/20) if other_optim is not None else None
        scheduler_scale = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(scale_optim, T_0=5, eta_min=self.base_lr/50) if scale_optim is not None else None
        
        self.model.train()
        best_loss = float('inf')
        stage = 0
        self._unfreeze_stage(0)
        
        for epoch in range(epochs):
            if epoch % 2 == 0 and epoch > 0:
                if hasattr(self, 'last_avg_loss') and self.last_avg_loss < 1.5:
                    stage += 1
                    if stage <= 2:
                        self._unfreeze_stage(stage)
                        other_optim, scale_optim = self.get_optimizer()
                        print(f"🔓 Advanced to stage {stage}: {self.num_trainable} params")
                    else:
                        print("Max stage reached; full LoRA on blocks 10-11")
                else:
                    print(f"Loss {getattr(self, 'last_avg_loss', 'inf'):.2f} >=1.5; staying at stage {stage}")

            loss_accum, batches, skipped = 0.0, 0, 0
            pbar = tqdm(loader, desc=f"Stable CLIP FT Epoch {epoch+1}/{epochs} (stage {stage})")
            
            for batch_idx, (images, texts) in enumerate(pbar):
                if self.global_step < self.warmup_steps:
                    progress = self.global_step / self.warmup_steps
                    lr_other = self.warmup_lr + progress * (self.base_lr - self.warmup_lr)
                    if other_optim is not None:
                        other_optim.param_groups[0]['lr'] = lr_other
                    if scale_optim is not None:
                        scale_optim.param_groups[0]['lr'] = lr_other * 0.005
                    self.global_step += 1
                
                images = images.to(self.device)
                tokens = clip.tokenize(texts, truncate=True).to(self.device)
                target_scale = 2.8
                current_scale = self.model.logit_scale
                if self.global_step > max(50, self.warmup_steps):
                    scale_reg = 0.01 * (current_scale - target_scale) ** 2
                else:
                    scale_reg = torch.tensor(0.0, device=self.device)
                with torch.no_grad():
                    self.model.logit_scale.data = torch.clamp(self.model.logit_scale.data, -6.0, 6.0)
                scale = self.model.logit_scale
                logit_scale = scale.exp()
                if batch_idx % 10 == 0:
                    try:
                        print(f"DEBUG SCALE: batch={batch_idx} global_step={self.global_step} current_param={float(current_scale):.3f} logit_scale={float(logit_scale):.3f} reg={float(scale_reg):.3f}")
                    except Exception:
                        pass

                if other_optim is not None:
                    other_optim.zero_grad()
                if scale_optim is not None:
                    scale_optim.zero_grad()
                
                try:
                    image_features = self._safe_encode_image(images)
                    text_features = self._safe_encode_text(tokens)
                    if not (torch.isfinite(image_features).all() and torch.isfinite(text_features).all()):
                        skipped += 1
                        continue
                    sim_img = torch.clamp(image_features @ text_features.t(), -3.0, 3.0)
                    sim_text = torch.clamp(text_features @ image_features.t(), -3.0, 3.0)
                    logits_per_image = logit_scale * sim_img
                    logits_per_text = logit_scale * sim_text
                    logits_per_image.clamp_(-20, 20)
                    logits_per_text.clamp_(-20, 20)
                    loss = self.contrastive_loss(logits_per_image, logits_per_text, self.device)
                    loss_for_backward = loss + scale_reg
                    if not torch.isfinite(loss_for_backward) or loss_for_backward.item() > 15:
                        skipped += 1
                        continue
                    loss_for_backward.backward()
                    proceed = self._handle_gradients()
                    if not proceed:
                        skipped += 1
                        continue
                    if other_optim is not None:
                        other_optim.step()
                    if scale_optim is not None:
                        scale_optim.step()
                    with torch.no_grad():
                        self.model.logit_scale.clamp_(-6.0, 6.0)
                        for name, p in self.model.named_parameters():
                            if p.requires_grad and ('lora' in name or 'text_projection' in name):
                                p.clamp_(-1.0, 1.0)
                    if scheduler_other is not None:
                        scheduler_other.step()
                    if scheduler_scale is not None:
                        scheduler_scale.step()
                    loss_accum += loss.item()
                    batches += 1
                    self.last_avg_loss = loss_accum / max(1, batches)
                    if batch_idx % 20 == 0:
                        avg_sim = sim_img.diag().mean().item()
                        print(f"Batch {batch_idx}: stage={stage}, scale={scale.item():.3f}, sim={avg_sim:.3f}, loss={loss.item():.3f}")
                    pbar.set_postfix({
                        "loss": f"{loss.item():.3f}",
                        "scale": f"{scale.item():.2f}",
                        "sim": f"{avg_sim:.3f}",
                        "nan_frac": f"{0:.1%}",
                        "skipped": skipped
                    })
                except Exception as e:
                    print(f"Batch {batch_idx} error: {e}")
                    skipped += 1
                    continue
            avg_loss = loss_accum / max(1, batches)
            skip_rate = skipped / max(1, len(loader))
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, skip={skip_rate:.1%}, trainable={self.num_trainable}")
            if skip_rate > max_skip_rate:
                print("High skips: Re-freeze visuals, scale-only")
                for name, p in self.model.named_parameters():
                    if 'visual' in name and p.requires_grad:
                        p.requires_grad = False
                other_optim, scale_optim = self.get_optimizer()
            if avg_loss < best_loss and batches > len(loader)*0.8:
                torch.save(self.model.state_dict(), save_path)
                best_loss = avg_loss
                print(f"✓ Saved stable LoRA model: {avg_loss:.4f}")
        return self.model

    def load_finetuned(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        return self.model

# ============================================================================
# PART 2: Enhanced CLIP Reward Model with Better Calibration
# ============================================================================

class EnhancedCLIPRewardModel:
    """Enhanced CLIP reward with temperature calibration and action probability scaling."""
    
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

# ============================================================================
# PART 3: Enhanced Environment Wrapper with Normalized Rewards
# ============================================================================

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

# ============================================================================
# PART 4: Enhanced CNN Feature Extractor (Deeper Architecture)
# ============================================================================

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

# ============================================================================
# PART 5: Enhanced Training Callback with Curriculum and Early Stopping
# ============================================================================

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
        if self.n_calls % 25000 < self.eval_freq:
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

# ============================================================================
# PART 6: Enhanced Environment Creation with Curriculum
# ============================================================================

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

# ============================================================================
# PART 7: Enhanced DQN Training (Paper-Aligned + Improvements)
# ============================================================================

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

# ============================================================================
# PART 8: Enhanced PPO Training with Target KL and Variable Clip
# ============================================================================

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

# ============================================================================
# PART 9: Enhanced Evaluation with Density Variants and Confusion Matrix
# ============================================================================

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
    for density in density:
        print(f"\n📊 Evaluating at density {density} vehicles...")
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
                "initial_vehicle_count": density,
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
        for ep in tqdm(range(n_episodes), desc=f"Density {density}"):
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
                        'episode': ep, 'density': density, 
                        'reward': ep_reward, 'length': ep_length,
                        'success': arrived and not crashed,
                        'frame': frame
                    })
        results[density] = {
            'success_rate': 100.0 * success_count / n_episodes,
            'collision_rate': 100.0 * collision_count / n_episodes,
            'timeout_rate': 100.0 * timeout_count / n_episodes,
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_length': float(np.mean(episode_lengths)),
            'actions': episode_actions,
            'n_episodes': n_episodes
        }
        print(f"  Density {density}: Success {results[density]['success_rate']:.1f}% | "
              f"Collision {results[density]['collision_rate']:.1f}% | Timeout {results[density]['timeout_rate']:.1f}% | "
              f"R={results[density]['mean_reward']:.2f}±{results[density]['std_reward']:.2f}")
        eval_env.close()
    if all_clip_suggestions and all_actions:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(all_clip_suggestions, all_actions, 
                              labels=[0, 1, 2, 3, 4], normalize='true')
        print(f"\n📈 Action Confusion Matrix (CLIP suggestion vs taken):")
        print("Rows: CLIP suggested | Columns: Action taken")
        print("[Slow, Maintain, Fast, Left, Right]")
        for row in cm:
            print(f"[{row[0]:.3f}, {row[1]:.3f}, {row[2]:.3f}, {row[3]:.3f}, {row[4]:.3f}]")
        alignment_rate = 100 * np.mean(np.array(all_clip_suggestions) == np.array(all_actions))
        mean_confidence = np.mean(all_confidences)
        print(f"Overall: Alignment {alignment_rate:.1f}% | Mean CLIP confidence {mean_confidence:.3f}")
    print(f"\n📋 Enhanced Evaluation Summary (matching paper Table 2):")
    print("| Method | Vehicles | Success | Collision | Timeout |")
    print("|--------|----------|---------|-----------|---------|")
    for density, res in results.items():
        print(f"| Enhanced | {density:8d} | {res['success_rate']:6.1f} | "
              f"{res['collision_rate']:8.1f} | {res['timeout_rate']:6.1f} |")
    return results, trajectory_data

# ============================================================================
# PART 10: Enhanced Data Collection with Curriculum and Richer Descriptions
# ============================================================================

def collect_enhanced_intersection_data(n_episodes=500, save_path=None, 
                                       use_dle=True, curriculum_densities=True, use_ollama=False,
                                       ollama_model: str = 'llama3.1:8b'):
    """
    Enhanced data collection: Episodes with curriculum density, matching specified label format.
    Labels include index, image path (images/f_XXXXXX.png), action_id, instruction, episode, t, ego_speed, n_controlled, courtesy.
    """
    print("\n" + "="*80)
    print("📸 Enhanced Data Collection for CLIP Fine-tuning")
    print(f"Episodes: {n_episodes} | Curriculum: {curriculum_densities} | DLE: {use_dle} | Ollama: {use_ollama}")
    print("Output: 600x600 RGB + specified label format")
    print("="*80)
    
    # Setup environment for data collection (RGB rendering)
    env = gym.make("intersection-v1", render_mode="rgb_array")
    
    # Base config with curriculum support
    base_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": False,
        },
        "action": {
            "type": "DiscreteMetaAction",
            "lateral": True,
            "longitudinal": True,
            "target_speeds": [0, 4.5, 9],
        },
        "duration": 40,
        "simulation_frequency": 15,
        "policy_frequency": 1,
        "reward_speed_range": [7.0, 9.0],
        "normalize_reward": False,
        "offroad_terminal": False,
        "screen_width": 600,
        "screen_height": 600,
        "centering_position": [0.3, 0.5],
        "scaling": 5.5,
        "show_trajectories": False,
    }
    
    # Density curriculum for data collection
    density_schedule = [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7, 8] * 30
    if not curriculum_densities:
        density_schedule = [5] * n_episodes
    
    dataset = []
    # Use repository-local data folder next to this script: <script_folder>/data
    base_data_dir = Path(__file__).parent / "data"
    base_data_dir.mkdir(parents=True, exist_ok=True)

    img_dir = base_data_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize enhanced encoder
    dle = EnhancedDataLanguageEncoder() if use_dle else None
    
    # Global index for dataset entries
    global_index = 0
    
    print(f"Collecting {n_episodes} episodes across varying density...")
    
    for episode in tqdm(range(n_episodes), desc="Collecting enhanced data"):
        # Set density for this episode
        density_idx = min(episode // 30, len(density_schedule) - 1)
        current_density = density_schedule[density_idx]
        
        # Update config
        config = base_config.copy()
        config.update({
            "initial_vehicle_count": current_density,
            "spawn_probability": 0.1 + 0.025 * current_density,
        })
        env.unwrapped.config.update(config)
        
        obs, info = env.reset(seed=episode)
        done = False
        step = 0
        max_steps = 40 * 15
        
        while not done and step < max_steps:
            # Get high-quality RGB frame
            rgb_frame = env.render()
            
            if rgb_frame is not None and len(rgb_frame.shape) == 3:
                # Generate action_id and instruction
                if use_ollama and OLLAMA_AVAILABLE:
                    try:
                        action_id, instruction = generate_description_with_ollama(env, model=ollama_model)
                    except Exception as e:
                        print(f"Ollama generation failed: {e}; falling back to DLE/heuristic")
                        if use_dle and dle is not None:
                            action_id, instruction = dle.describe(env)
                        else:
                            action_id, instruction = _heuristic_description(env, current_density)
                elif use_dle and dle is not None:
                    action_id, instruction = dle.describe(env)
                else:
                    action_id, instruction = _heuristic_description(env, current_density)
                
                # Save image with specified naming (f_XXXXXX.png)
                img_filename = f"f_{global_index:06d}.png"
                img_path = os.path.join(img_dir, img_filename)
                img = Image.fromarray(rgb_frame)
                img.save(img_path, quality=95, optimize=True)
                
                # Count controlled vehicles (vehicles within 25m)
                ego = env.unwrapped.vehicle
                n_controlled = 0
                if ego is not None:
                    try:
                        n_controlled = len([v for v in env.unwrapped.road.vehicles 
                                            if v is not ego and np.linalg.norm(v.position - ego.position) < 25])
                    except Exception:
                        n_controlled = 3  # Fallback to 3 as per example
                
                # Ego speed (default to example value if unavailable)
                ego_speed = float(ego.speed) if ego is not None else 11.11111111111111
                
                # Create dataset entry matching specified format
                dataset.append({
                    "index": global_index,
                    "image": f"images/{img_filename}",
                    "action_id": action_id,
                    "instruction": instruction,
                    "episode": episode,
                    "t": step,
                    "ego_speed": ego_speed,
                    "n_controlled": n_controlled,
                    "courtesy": True  # Fixed to true as per example
                })
                
                global_index += 1
            
            # Random action for exploration
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
        
        if episode % 100 == 0:
            print(f"Episode {episode}: Collected {len(dataset)} samples | "
                  f"Current density: {current_density}")
    
    # Save dataset into the repo-local data folder next to this script.
    # Default: a single JSON array named `frame.json` and a JSONL compatibility copy `frames.jsonl`.
    if save_path is None:
        save_path = base_data_dir / "frame.json"
    else:
        save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON array (frame.json)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    # Also write a JSONL copy for tools that expect line-delimited JSON
    frames_jsonl = base_data_dir / "frames.jsonl"
    with open(frames_jsonl, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n✅ Enhanced dataset saved to {save_path}")
    print(f"✅ JSONL compatibility copy written to {frames_jsonl}")
    print(f"📈 Total samples: {len(dataset)} across {n_episodes} episodes")
    print(f"🎯 Density distribution: {min(density_schedule):d}-{max(density_schedule):d} vehicles")
    print(f"🖼️ Images saved to: {img_dir}/ (600x600 RGB, high quality)")
    
    if use_dle:
        print("📝 Descriptions generated with Enhanced Data Language Encoder")
    if use_ollama and OLLAMA_AVAILABLE:
        print(f"📝 Some descriptions generated with Ollama ({ollama_model})")
    
    env.close()
    return dataset

def _heuristic_description(env, current_density):
    """Simple fallback heuristic for action_id and instruction."""
    ego = env.unwrapped.vehicle
    if ego is not None:
        try:
            speed = ego.speed
        except Exception:
            speed = 0.0
        try:
            nearby_vehicles = len([v for v in env.unwrapped.road.vehicles 
                                   if v is not ego and np.linalg.norm(v.position - ego.position) < 25])
        except Exception:
            nearby_vehicles = 0
        if nearby_vehicles >= current_density * 0.5 and speed > 6:
            action_id = "SLOWER"
            phrases = [PROMPTS["SLOWER"]] + PARAPHRASES.get("SLOWER", [])
            instruction = random.choice(phrases)
        elif nearby_vehicles < 2 and speed < 4:
            action_id = "FASTER"
            phrases = [PROMPTS["FASTER"]] + PARAPHRASES.get("FASTER", [])
            instruction = random.choice(phrases)
        else:
            action_id = "IDLE"
            phrases = [PROMPTS["IDLE"]] + PARAPHRASES.get("IDLE", [])
            instruction = random.choice(phrases)
    else:
        action_id = "IDLE"
        instruction = PROMPTS["IDLE"]
    return action_id, instruction

def generate_description_with_ollama(env, model='llama3.1:8b', timeout=60):
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
    if isinstance(resp, dict):
        text = resp.get('message', {}).get('content', '')
    else:
        text = str(resp)
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

# ============================================================================
# PART 11: Complete Enhanced Pipeline
# ============================================================================

def main_enhanced_pipeline():
    USE_WANDB = False
    N_SEEDS = 3
    COLLECT_DATA = True
    FINETUNE_CLIP = False
    TRAIN_DQN = False
    TRAIN_PPO = False
    EVALUATE_VANILLA = False
    DETAILED_EVAL = False
    data_path = "enhanced_intersection_dataset.json"
    clip_path = "robust_clip_finetuned.pt"
    models_dir = "enhanced_models"
    os.makedirs(models_dir, exist_ok=True)
    if USE_WANDB and WANDB_AVAILABLE:
        try:
            wandb.init(
                project="clip-rldrive-enhanced", 
                name="enhanced_pipeline_v1",
                config={
                    "n_episodes_data": 500,
                    "clip_epochs": 25,
                    "dqn_timesteps": 30000,
                    "ppo_timesteps": 50000,
                    "curriculum": True,
                    "n_seeds": N_SEEDS
                }
            )
            print("✓ Enhanced Weights & Biases initialized")
        except Exception as e:
            print(f"Warning: wandb init failed: {e}")
            USE_WANDB = False
    else:
        print("ℹ️ Running without wandb logging")
    if COLLECT_DATA:
        print("\n" + "="*80)
        print("STEP 1: Enhanced Data Collection (500 episodes, curriculum density)")
        print("="*80)
        dataset = collect_enhanced_intersection_data(
            n_episodes=500,
            save_path=data_path,
            use_dle=True,
            curriculum_densities=True
        )
    else:
        print(f"ℹ️ Using existing dataset: {data_path}")
        p = Path(data_path)
        if not p.exists():
            alt = Path(__file__).parent / 'data' / p.name
            if alt.exists():
                p = alt
        dataset = []
        try:
            with open(p, 'r', encoding='utf-8') as f:
                try:
                    dataset = json.load(f)
                except json.JSONDecodeError:
                    f.seek(0)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            dataset.append(json.loads(line))
                        except Exception:
                            continue
        except FileNotFoundError:
            raise
    if FINETUNE_CLIP:
        print("\n" + "="*80)
        print("STEP 2: Enhanced CLIP Fine-tuning (25 epochs, deeper adaptation)")
        print("="*80)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        ft = RobustCLIPFineTuner(model_name="ViT-B/32", device=device)
        ft.train(
            dataset_path=data_path,
            epochs=25,
            batch_size=4,
            save_path=clip_path,
            weight_decay=1e-5
        )
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        clip_model.load_state_dict(torch.load(clip_path, map_location=device))
        test_rm = EnhancedCLIPRewardModel(clip_model, preprocess, device)
        print("\n🔍 Quick CLIP validation:")
        for i in range(3):
            sample = dataset[i]
            img = Image.open(sample['image'])
            frame = np.array(img.resize((224, 224)))
            r_clip, best_act, conf, probs = test_rm.score(frame)
            action_names = ["SLOW", "IDLE", "FAST", "LEFT", "RIGHT"]
            print(f"Sample {i}: '{sample['instruction'][:50]}...' -> "
                  f"Action: {action_names[best_act]} ({conf:.3f}, r={r_clip:.3f})")
    else:
        print(f"ℹ️ Using existing CLIP model: {clip_path}")
    dqn_results = {}
    if TRAIN_DQN:
        print("\n" + "="*80)
        print("STEP 3: Training Enhanced DQN (Multiple Seeds)")
        print("="*80)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        if os.path.exists(clip_path):
            clip_model.load_state_dict(torch.load(clip_path, map_location=device))
        clip_reward_model = EnhancedCLIPRewardModel(clip_model, preprocess, device)
        for seed in range(N_SEEDS):
            seed_path = f"{models_dir}/enhanced_dqn_clip_seed{seed}"
            print(f"\n🔄 Training DQN Seed {seed}/{N_SEEDS-1}")
            dqn_model, env_fn = train_enhanced_dqn(
                clip_model_path=clip_path if os.path.exists(clip_path) else None,
                total_timesteps=300000,
                use_clip=True,
                save_path=seed_path,
                use_wandb=USE_WANDB,
                seed=42 + seed * 10
            )
            eval_results, _ = evaluate_enhanced_agent(
                dqn_model, n_episodes=30, density=[3],
                clip_reward_model=clip_reward_model,
                seed=42 + seed * 10
            )
            dqn_results[f'dqn_clip_seed{seed}'] = eval_results
            if hasattr(env_fn, 'close'):
                env_fn().close()
    if TRAIN_PPO:
        print("\n" + "="*80)
        print("STEP 4: Training Enhanced PPO (Single Seed - Experimental)")
        print("="*80)
        ppo_model, ppo_env_fn = train_enhanced_ppo(
            clip_model_path=clip_path if os.path.exists(clip_path) else None,
            total_timesteps=500000,
            use_clip=True,
            save_path=f"{models_dir}/enhanced_ppo_clip",
            use_wandb=USE_WANDB,
            seed=42
        )
    if DETAILED_EVAL and TRAIN_DQN:
        print("\n" + "="*80)
        print("STEP 5: Detailed Enhanced Evaluation")
        print("="*80)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        if os.path.exists(clip_path):
            clip_model.load_state_dict(torch.load(clip_path, map_location=device))
        clip_reward_model = EnhancedCLIPRewardModel


# Run pipeline when executed as script
if __name__ == "__main__":
    main_enhanced_pipeline()