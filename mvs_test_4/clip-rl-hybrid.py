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
    "LANE_RIGHT": "Change to the right lane.",
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
            "slow down and be cautious",
            "maintain current speed", 
            "speed up and go faster",
            "change to the left lane.",
            "change to the right lane."
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
        ego = env.unwrapped.vehicle
        if ego is None:
            return "intersection scene with no ego vehicle available; maintain safe speed.", self.ACTIONS[1]
        
        speed = float(ego.speed)
        conflicts = self._find_conflicting_vehicles(env, ego)
        total_density, sectors = self._density_and_directions(env, ego)
        
        # Ego state
        speed_str = ("very low speed" if speed < self.speed_lo else
                     "moderate speed" if speed < self.speed_hi else "high speed")
        
        # Traffic pattern
        if total_density == 0:
            traffic_str = "clear intersection with no nearby vehicles"
        elif total_density <= 2:
            traffic_str = f"light traffic: {sectors['front']} ahead, {sectors['left']} crossing from left"
        else:
            traffic_str = f"heavy traffic: {sectors['front']} ahead, {sectors['left']} from left, {sectors['right']} from right"
        
        # Risk assessment
        if conflicts:
            nearest = conflicts[0]
            if nearest['ttc'] < self.ttc_danger:
                risk_str = f"imminent collision risk from {nearest['direction']:.0f}° vehicle {nearest['gap']:.0f}m away"
            elif nearest['ttc'] < self.ttc_caution:
                risk_str = f"potential crossing risk from {nearest['direction']:.0f}° direction"
            else:
                risk_str = f"vehicle {nearest['gap']:.0f}m away at {nearest['direction']:.0f}° moving at {nearest['rel_speed']:.1f}m/s relative"
        else:
            risk_str = "no immediate crossing vehicles detected"
        
        # Enhanced action recommendation with lane changes
        needs_slow = (len(conflicts) > 0 and conflicts[0]['ttc'] < self.ttc_caution) or \
                     (sectors['front'] > 0 and speed > self.speed_hi) or total_density >= 4
        needs_speed = (speed < self.speed_lo and len(conflicts) == 0 and 
                       sectors['front'] == 0 and total_density < 2)
        needs_left = (sectors['front'] > 1 and sectors['left'] < 1 and sectors['right'] > sectors['left']) or \
                     (conflicts and 45 < conflicts[0]['direction'] < 135)  # Conflict from right, go left to pass
        needs_right = (sectors['front'] > 1 and sectors['right'] < 1 and sectors['left'] > sectors['right']) or \
                      (conflicts and 225 < conflicts[0]['direction'] < 315)  # Conflict from left, go right to yield
        
        if needs_left:
            action_index = 3
        elif needs_right:
            action_index = 4
        elif needs_slow:
            action_index = 0
        elif needs_speed:
            action_index = 2
        else:
            action_index = 1
        
        action_key = ACTION_MAP[action_index]
        phrases = [PROMPTS[action_key]] + PARAPHRASES.get(action_key, [])
        action_phrase = random.choice(phrases)
        
        # Rich description
        desc = (f"unsignalized intersection preparing left turn; ego at {speed_str} ({speed:.1f}m/s); "
                f"{traffic_str}; {risk_str}; recommended action: {action_phrase}")
        
        return desc, action_phrase

# ============================================================================
# The rest of the code remains the same, except for changes in env config for lateral=True
# ============================================================================

class AugmentedIntersectionDataset(Dataset):
    def __init__(self, data_path, transform=None, augment_prob=0.5):
        self.data_path = data_path
        self.transform = transform
        self.augment_prob = augment_prob
        with open(data_path, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image']).convert('RGB')
        
        if random.random() < self.augment_prob:
            angle = random.uniform(-15, 15)
            image = image.rotate(angle, resample=Image.BICUBIC)
            
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(0.85, 1.15)
            image = enhancer.enhance(factor)
        
        if self.transform:
            image = self.transform(image)
        text = item['description']
        return image, text

class RobustCLIPFineTuner:
    def __init__(self, model_name="ViT-B/32", device=None, initial_freeze_epochs=15):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.initial_freeze_epochs = initial_freeze_epochs
        self.model_name = model_name
        
        for p in self.model.parameters():
            p.requires_grad = False
        with torch.no_grad():
            self.model.logit_scale.data = torch.tensor(-4.60517, device=self.device)
        
        self.num_trainable = 0
        self.global_step = 0
        self.warmup_steps = 0
        self.base_lr = 5e-4
        self.warmup_lr = 0.0
        self.unfrozen_blocks = []
        self.lora_applied = False
        self.action_descriptions = list(PROMPTS.values())  # Updated to 5

    def _apply_lora_to_blocks(self, block_indices=[10, 11]):
        if self.lora_applied or not PEFT_AVAILABLE:
            return
        lora_config = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        visual_model = self.model.visual.transformer
        for i in block_indices:
            if i < len(visual_model.resblocks):
                try:
                    get_peft_model(visual_model.resblocks[i], lora_config)
                except:
                    pass
                for name, module in visual_model.resblocks[i].named_modules():
                    if any(k in name for k in ['q_proj', 'k_proj', 'v_proj', 'out_proj']):
                        try:
                            module.requires_grad_(True)
                        except:
                            pass
        self.lora_applied = True
        self._update_trainable()

    def _unfreeze_stage(self, stage):
        if stage == 0:
            if hasattr(self.model, 'logit_scale'):
                self.model.logit_scale.requires_grad_(True)
            if hasattr(self.model, 'text_projection'):
                self.model.text_projection.requires_grad_(True)
        elif stage == 1:
            self._apply_lora_to_blocks([11])
            self.unfrozen_blocks = [11]
        elif stage == 2:
            self._apply_lora_to_blocks([10, 11])
            self.unfrozen_blocks = [10, 11]
        self._update_trainable()

    def _update_trainable(self):
        self.num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _patch_visual_forwards(self):
        if hasattr(self, '_patched'):
            return
        def clamped_forward(self_block, x):
            try:
                qkv = self_block.attn.in_proj(x)
                qkv = torch.clamp(qkv, -5.0, 5.0)
                x = self_block.attn(qkv)
                x_mlp = self_block.mlp(x)
                x_mlp = torch.clamp(x_mlp, -10.0, 10.0)
                x = x + x_mlp
                x = torch.clamp(x, -10.0, 10.0)
            except:
                try:
                    x = self_block.__class__.forward(self_block, x)
                except:
                    pass
            return x
        for resblock in self.model.visual.transformer.resblocks:
            resblock.forward = types.MethodType(clamped_forward, resblock)
        self._patched = True

    def get_optimizer(self):
        named = dict(self.model.named_parameters())

        def pick_params_by_keys(keys):
            out = []
            for k in keys:
                for name, p in named.items():
                    if k in name and p.requires_grad:
                        out.append(p)
            return out

        scale_params = [p for name, p in named.items() if 'logit_scale' in name and p.requires_grad]
        proj_params = [p for name, p in named.items() if 'text_projection' in name and p.requires_grad]
        lora_params = [p for name, p in named.items() if 'lora' in name and p.requires_grad] if self.lora_applied else []

        seen = set()
        other_params = []
        for p in proj_params + lora_params:
            if id(p) not in seen:
                seen.add(id(p))
                other_params.append(p)

        other_optim = torch.optim.AdamW(other_params, lr=self.base_lr, weight_decay=0.01) if other_params else None
        scale_optim = torch.optim.SGD(scale_params, lr=max(1e-6, self.base_lr * 0.001)) if scale_params else None

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
        scheduler_other = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(other_optim, T_0=5, eta_min=self.base_lr/20) if other_optim else None
        scheduler_scale = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(scale_optim, T_0=5, eta_min=self.base_lr/50) if scale_optim else None
        
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
                    else:
                        print("Max stage reached")
                else:
                    print(f"Loss {getattr(self, 'last_avg_loss', 'inf'):.2f} >=1.5; staying at stage {stage}")

            loss_accum, batches, skipped = 0.0, 0, 0
            pbar = tqdm(loader, desc=f"Stable CLIP FT Epoch {epoch+1}/{epochs} (stage {stage})")
            
            for batch_idx, (images, texts) in enumerate(pbar):
                if self.global_step < self.warmup_steps:
                    progress = self.global_step / self.warmup_steps
                    lr_other = self.warmup_lr + progress * (self.base_lr - self.warmup_lr)
                    if other_optim:
                        other_optim.param_groups[0]['lr'] = lr_other
                    if scale_optim:
                        scale_optim.param_groups[0]['lr'] = lr_other * 0.005
                    self.global_step += 1
                
                images = images.to(self.device)
                tokens = clip.tokenize(texts, truncate=True).to(self.device)

                target_scale = 2.8
                current_scale = self.model.logit_scale
                scale_reg = 0.01 * (current_scale - target_scale) ** 2 if self.global_step > max(50, self.warmup_steps) else torch.tensor(0.0, device=self.device)

                self.model.logit_scale.data = torch.clamp(self.model.logit_scale.data, -6.0, 6.0)
                scale = self.model.logit_scale
                logit_scale = scale.exp()

                if other_optim:
                    other_optim.zero_grad()
                if scale_optim:
                    scale_optim.zero_grad()
                
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
                
                if other_optim:
                    other_optim.step()
                if scale_optim:
                    scale_optim.step()
                
                self.model.logit_scale.clamp_(-6.0, 6.0)
                for name, p in self.model.named_parameters():
                    if p.requires_grad and ('lora' in name or 'text_projection' in name):
                        p.clamp_(-1.0, 1.0)

                if scheduler_other:
                    scheduler_other.step()
                if scheduler_scale:
                    scheduler_scale.step()
                
                loss_accum += loss.item()
                batches += 1
                self.last_avg_loss = loss_accum / max(1, batches)
                
                if batch_idx % 20 == 0:
                    avg_sim = sim_img.diag().mean().item()
                
                pbar.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "scale": f"{scale.item():.2f}",
                    "sim": f"{avg_sim:.3f}",
                    "nan_frac": f"{0:.1%}",
                    "skipped": skipped
                })
            
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
        
        return self.model

    def load_finetuned(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        return self.model

# ============================================================================
# PART 2: Enhanced CLIP Reward Model (Extended to 5 actions)
# ============================================================================

class EnhancedCLIPRewardModel:
    def __init__(self, clip_model, preprocess, device=None, temperature=0.07):
        self.model = clip_model.eval()
        self.preprocess = preprocess
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        
        self.action_texts = list(PROMPTS.values())  # 5 actions
        
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
        
        img = Image.fromarray(frame_np)
        processed = self.preprocess(img).unsqueeze(0).to(self.device)
        features = self.model.encode_image(processed)
        return F.normalize(features, dim=-1)
    
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
# PART 3: Enhanced Environment Wrapper
# ============================================================================

class EnhancedCLIPRewardWrapper(gym.Wrapper):
    def __init__(self, env, clip_reward_model, weight_clip=1.2, max_episode_steps=40, 
                 use_curriculum=False, current_density_level=0):
        super().__init__(env)
        self.rm = clip_reward_model
        self.weight_clip = weight_clip
        self.max_episode_steps = max_episode_steps * 15  # Hz
        self.use_curriculum = use_curriculum
        self.current_density_level = current_density_level
        
        self.density_levels = [
            (1, 0.1),
            (3, 0.15),
            (5, 0.2),
            (8, 0.3)
        ]
        
        if self.use_curriculum:
            self._update_density()
        
        self.clip_reward_count = 0
        self.total_steps = 0
        self.episode_step = 0
        self.episode_rewards = deque(maxlen=100)
    
    def _update_density(self):
        if self.current_density_level < len(self.density_levels):
            initial_count, spawn_prob = self.density_levels[self.current_density_level]
            self.env.unwrapped.config.update({
                "initial_vehicle_count": initial_count,
                "spawn_probability": spawn_prob
            })
    
    def advance_curriculum(self):
        if not self.use_curriculum or self.current_density_level >= len(self.density_levels) - 1:
            return
        
        recent_success_rate = np.mean(self.episode_rewards) / self.max_episode_steps * 100
        if recent_success_rate > 70:
            self.current_density_level += 1
            self._update_density()
    
    def reset(self, **kwargs):
        self.episode_step = 0
        obs, _ = self.env.reset(**kwargs)
        return obs
    
    def step(self, action):
        obs, r_env, terminated, truncated, info = self.env.step(action)
        frame = self._get_frame_from_obs(obs)
        
        r_clip, suggested_action, confidence, action_probs = self.rm.score(frame, action)
        
        self.total_steps += 1
        self.episode_step += 1
        
        r_env_norm = r_env / self.max_episode_steps if self.max_episode_steps > 0 else r_env
        
        r_clip_scaled = self.weight_clip * r_clip
        
        follow_bonus = 0.1 * confidence if action == suggested_action else 0.0
        
        living_penalty = -0.01 / self.max_episode_steps
        
        r_total = r_env_norm + r_clip_scaled + follow_bonus + living_penalty
        
        self.clip_reward_count += 1
        info.update({
            'clip_reward': r_clip,
            'clip_confidence': confidence,
            'clip_suggested': suggested_action,
            'base_reward': r_env,
            'base_reward_norm': r_env_norm,
            'follow_bonus': follow_bonus,
            'episode_step': self.episode_step,
            'density_level': self.current_density_level
        })
        
        if terminated or truncated:
            self.episode_rewards.append(self.episode_step)
            if 'episode' in info:
                info['episode']['r'] = r_total
                
        return obs, r_total, terminated, truncated, info
    
    def _get_frame_from_obs(self, obs):
        arr = np.array(obs)
        
        if arr.ndim == 3:
            frame = arr[-1] if arr.shape[0] <= 4 else arr
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
# PART 4: Enhanced CNN Feature Extractor
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
        return self.linear(self.cnn(observations))

# ============================================================================
# PART 5: Enhanced Training Callback
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
                    self.curriculum_wrapper.advance_curriculum()
            
            mean_reward = np.mean(list(self.episode_rewards)[-50:]) if len(self.episode_rewards) > 0 else -float('inf')
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.no_improvement_steps = 0
            else:
                self.no_improvement_steps += self.eval_freq
                
            if self.no_improvement_steps >= self.patience * self.eval_freq:
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
        
        if self.use_wandb:
            wandb.log(metrics)
        
        if self.n_calls % 25000 < self.eval_freq:
            self._detailed_evaluation()
    
    def _detailed_evaluation(self):
        mean_reward, std_reward = evaluate_policy(
            self.model, self.eval_env, n_eval_episodes=20, deterministic=True, 
            return_episode_rewards=True
        )

# ============================================================================
# PART 6: Enhanced Environment Creation (lateral=True)
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
                "lateral": True,  # Changed to True for 5 actions
                "longitudinal": True,
                "target_speeds": [0, 4.5, 9],
            },
            "duration": 40,
            "simulation_frequency": 15,
            "policy_frequency": 15,
            "vehicles_count": 5,
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
                max_episode_steps=40
            )
        
        return env
    
    return make_env

# ============================================================================
# PART 7: Enhanced DQN Training
# ============================================================================

def train_enhanced_dqn(clip_model_path, total_timesteps=300000, use_clip=True, 
                       save_path="models/enhanced_dqn_clip", use_wandb=False, seed=42):
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
            return True
    
    lr_callback = LRSchedulerCallback()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[callback, lr_callback],
        log_interval=10,
        progress_bar=True,
        tb_log_name=f"dqn_{'clip' if use_clip else 'vanilla'}_{seed}"
    )
    
    model.save(save_path)
    
    eval_env.close()
    env.close()
    
    return model, env_fn

# ============================================================================
# PART 8: Enhanced PPO Training
# ============================================================================

def train_enhanced_ppo(clip_model_path, total_timesteps=500000, use_clip=True,
                       save_path="models/enhanced_ppo_clip", use_wandb=False, seed=42):
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
            current_kl = self.model.logger.name_to_value.get('train/approx_kl', 0)
            self.last_kl = current_kl
                
            if current_kl > self.kl_threshold * 2:
                self.model.clip_range = min(0.3, self.model.clip_range * 0.9)
            elif current_kl < self.kl_threshold / 2:
                self.model.clip_range = max(0.1, self.model.clip_range * 1.1)
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
                self.model.ent_coef = ent_coef
            return True
    
    ent_callback = EntropySchedulingCallback(total_timesteps)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[callback, KLMonitoringCallback(), ent_callback],
        log_interval=10,
        progress_bar=True,
        tb_log_name=f"ppo_{'clip' if use_clip else 'vanilla'}_{seed}"
    )
    
    model.save(save_path)
    
    eval_env.close()
    env.close()
    
    return model, env_fn

# ============================================================================
# PART 9: Enhanced Evaluation
# ============================================================================

def evaluate_enhanced_agent(model, n_episodes=100, densities=[1, 3, 6], render=False, 
                            clip_reward_model=None, seed=42):
    results = {}
    all_actions = []
    all_clip_suggestions = []
    all_confidences = []
    trajectory_data = []
    
    for density in densities:
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
                        r_clip, sugg, conf, probs = self.rm.score(frame, action)
                        info.update({
                            'clip_suggested': sugg,
                            'clip_confidence': conf,
                            'clip_probs': probs.tolist()
                        })
                        return obs, reward, term, trunc, info
                    
                    def _get_frame(self, obs):
                        arr = np.array(obs)
                        frame = arr[-1] if arr.ndim == 3 and arr.shape[0] <= 4 else arr
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
            
        success_rate = 100.0 * success_count / n_episodes
        collision_rate = 100.0 * collision_count / n_episodes
        timeout_rate = 100.0 * timeout_count / n_episodes
        
        results[density] = {
            'success_rate': success_rate,
            'collision_rate': collision_rate, 
            'timeout_rate': timeout_rate,
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_length': float(np.mean(episode_lengths)),
            'n_episodes': n_episodes
        }
        
        eval_env.close()
    
    if all_clip_suggestions and all_actions:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(all_clip_suggestions, all_actions, labels=[0,1,2,3,4], normalize='true')
        
        alignment_rate = 100 * np.mean(np.array(all_clip_suggestions) == np.array(all_actions))
        mean_confidence = np.mean(all_confidences)
    
    return results, trajectory_data

# ============================================================================
# PART 10: Enhanced Data Collection (Combined logic)
# ============================================================================

def collect_enhanced_intersection_data(n_episodes=500, save_path="enhanced_intersection_dataset.json", 
                                       use_dle=True, curriculum_densities=True, use_ollama=False,
                                       ollama_model: str = 'llama3.1:8b'):
    env = gym.make("intersection-v1", render_mode="rgb_array")
    
    base_config = {
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
    
    density_schedule = [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7, 8] * (n_episodes // 17 + 1)
    if not curriculum_densities:
        density_schedule = [5] * n_episodes
    
    dataset = []
    img_dir = "enhanced_intersection_images"
    os.makedirs(img_dir, exist_ok=True)
    
    dle = EnhancedDataLanguageEncoder() if use_dle else None
    
    for episode in tqdm(range(n_episodes), desc="Collecting enhanced data"):
        current_density = density_schedule[episode % len(density_schedule)]
        
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
            rgb_frame = env.render()
            
            if rgb_frame is not None and len(rgb_frame.shape) == 3:
                desc, action_phrase = dle.describe(env) if dle else _heuristic_description(env, current_density)
                
                img = Image.fromarray(rgb_frame)
                img_filename = f"ep{episode:03d}_t{step:03d}_d{current_density}.png"
                img_path = os.path.join(img_dir, img_filename)
                img.save(img_path, quality=95, optimize=True)
                
                dataset.append({
                    "image": img_path,
                    "description": desc,
                    "action": action_phrase,
                    "episode": episode,
                    "step": step,
                    "density": current_density,
                    "ego_speed": float(env.unwrapped.vehicle.speed) if env.unwrapped.vehicle else 0.0
                })
            
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
    
    with open(save_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    env.close()
    return dataset

# ============================================================================
# PART 11: Complete Enhanced Pipeline
# ============================================================================

def main_enhanced_pipeline():
    USE_WANDB = False
    N_SEEDS = 3
    
    COLLECT_DATA = True
    FINETUNE_CLIP = True
    TRAIN_DQN = True
    TRAIN_PPO = True
    EVALUATE_VANILLA = True
    DETAILED_EVAL = True
    
    data_path = "enhanced_intersection_dataset.json"
    clip_path = "robust_clip_finetuned.pt"
    models_dir = "enhanced_models"
    os.makedirs(models_dir, exist_ok=True)
    
    if USE_WANDB and WANDB_AVAILABLE:
        wandb.init(project="clip-rldrive-enhanced", name="enhanced_pipeline_v1")
    
    if COLLECT_DATA:
        dataset = collect_enhanced_intersection_data(
            n_episodes=500,
            save_path=data_path,
            use_dle=True,
            curriculum_densities=True
        )
    else:
        with open(data_path, 'r') as f:
            dataset = json.load(f)
    
    if FINETUNE_CLIP:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ft = RobustCLIPFineTuner(device=device)
        ft.train(
            dataset_path=data_path,
            epochs=25,
            batch_size=4,
            save_path=clip_path
        )
    
    dqn_results = {}
    if TRAIN_DQN:
        for seed in range(N_SEEDS):
            seed_path = f"{models_dir}/enhanced_dqn_clip_seed{seed}"
            dqn_model, env_fn = train_enhanced_dqn(
                clip_model_path=clip_path,
                total_timesteps=300000,
                use_clip=True,
                save_path=seed_path,
                use_wandb=USE_WANDB,
                seed=42 + seed * 10
            )
            eval_results, _ = evaluate_enhanced_agent(
                dqn_model, n_episodes=30, densities=[3],
                clip_reward_model=EnhancedCLIPRewardModel(*clip.load("ViT-B/32", device="cpu"), "cpu"),
                seed=42 + seed * 10
            )
            dqn_results[f'dqn_clip_seed{seed}'] = eval_results
    
    if TRAIN_PPO:
        ppo_model, ppo_env_fn = train_enhanced_ppo(
            clip_model_path=clip_path,
            total_timesteps=500000,
            use_clip=True,
            save_path=f"{models_dir}/enhanced_ppo_clip",
            use_wandb=USE_WANDB,
            seed=42
        )
    
    if DETAILED_EVAL and TRAIN_DQN:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        if os.path.exists(clip_path):
            clip_model.load_state_dict(torch.load(clip_path, map_location=device))
        clip_reward_model = EnhancedCLIPRewardModel(clip_model, preprocess, device)
        
        best_dqn_path = f"{models_dir}/enhanced_dqn_clip_seed0"
        enhanced_dqn = DQN.load(best_dqn_path, device=device)
        enhanced_results, traj_data = evaluate_enhanced_agent(
            enhanced_dqn,
            n_episodes=100,
            densities=[1, 3, 6],
            clip_reward_model=clip_reward_model,
            seed=42
        )
        
        if EVALUATE_VANILLA:
            vanilla_dqn_path = f"{models_dir}/dqn_vanilla.zip"
            if os.path.exists(vanilla_dqn_path):
                vanilla_dqn = DQN.load(vanilla_dqn_path, device=device)
                vanilla_results, _ = evaluate_enhanced_agent(
                    vanilla_dqn,
                    n_episodes=100,
                    densities=[1, 3, 6],
                    seed=42
                )
    
    if USE_WANDB and WANDB_AVAILABLE:
        wandb.finish()
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced CLIP-RLDrive pipeline and utilities")
    sub = parser.add_subparsers(dest='command')

    p_pipeline = sub.add_parser('pipeline', help='Run the full enhanced pipeline')

    p_over = sub.add_parser('oversample', help='Oversample minority action entries in dataset JSON')
    p_over.add_argument('--input', default='enhanced_intersection_dataset.json')
    p_over.add_argument('--output', default='enhanced_intersection_dataset_oversampled.json')
    p_over.add_argument('--action', required=True)
    p_over.add_argument('--multiplier', type=int, default=10)
    p_over.add_argument('--no-shuffle', dest='shuffle', action='store_false')
    p_over.set_defaults(shuffle=True)
    p_over.add_argument('--seed', type=int, default=42)

    p_collect = sub.add_parser('collect', help='Collect enhanced intersection dataset')
    p_collect.add_argument('--n-episodes', type=int, default=500)
    p_collect.add_argument('--save-path', type=str, default='enhanced_intersection_dataset.json')
    p_collect.add_argument('--use-dle', action='store_true')
    p_collect.add_argument('--no-dle', action='store_false', dest='use_dle')
    p_collect.set_defaults(use_dle=True)
    p_collect.add_argument('--use-ollama', action='store_true')
    p_collect.add_argument('--no-ollama', action='store_false', dest='use_ollama')
    p_collect.set_defaults(use_ollama=False)
    p_collect.add_argument('--ollama-model', type=str, default='llama3.1:8b')

    args = parser.parse_args()

    def oversample_dataset(input_path, output_path, action_value, multiplier=10, shuffle=True, seed=42):
        with open(input_path, 'r') as f:
            data = json.load(f)
        matches = [d for d in data if d.get('action', '').strip().lower() == action_value.strip().lower()]
        duplicates = []
        for _ in range(multiplier - 1):
            duplicates.extend([dict(d) for d in matches])
        new_data = data + duplicates
        if shuffle:
            random.seed(seed)
            random.shuffle(new_data)
        with open(output_path, 'w') as f:
            json.dump(new_data, f, indent=2)

    if args.command == 'oversample':
        oversample_dataset(args.input, args.output, args.action, args.multiplier, args.shuffle, args.seed)
    elif args.command == 'collect':
        collect_enhanced_intersection_data(n_episodes=args.n_episodes, save_path=args.save_path,
                                           use_dle=args.use_dle, curriculum_densities=True,
                                           use_ollama=args.use_ollama, ollama_model=args.ollama_model)
    else:
        main_enhanced_pipeline()