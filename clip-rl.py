"""
CLIP-RLDrive: Full Training Pipeline (Fixed/Aligned)
Based on: https://arxiv.org/html/2412.16201v1
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
import highway_env  # noqa: F401 - needed to register envs
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
import clip
from PIL import Image
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False
from tqdm import tqdm
import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# PART 1: CLIP Fine-tuning Components
# ============================================================================

class IntersectionDataset(Dataset):
    """Dataset for (image, text description) pairs from intersection scenarios"""
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        with open(data_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        text = item['description']
        return image, text


class CLIPFineTuner:
    """Fine-tune CLIP model on intersection driving data (last layers only)"""
    def __init__(self, model_name="ViT-B/32", device=None, freeze_bottom_layers=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)

        if freeze_bottom_layers:
            for name, p in self.model.named_parameters():
                p.requires_grad = False
            # Unfreeze last visual block + ln_post + text_proj (lightweight)
            for key, p in self.model.named_parameters():
                if ("visual.transformer.resblocks.11" in key or
                    "visual.ln_post" in key or
                    "visual.proj" in key or
                    "logit_scale" in key):
                    p.requires_grad = True

        self.action_descriptions = [
            "slow down and be cautious",
            "maintain current speed",
            "speed up and go faster"
        ]

    @staticmethod
    def contrastive_loss(logits_per_image, logits_per_text, device):
        bsz = logits_per_image.shape[0]
        labels = torch.arange(bsz, device=device)
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        return 0.5 * (loss_i + loss_t)

    def train(self, dataset_path, epochs=15, batch_size=32, lr=5e-6, save_path="clip_finetuned.pt"):
        dataset = IntersectionDataset(dataset_path, transform=self.preprocess)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

        params = [p for p in self.model.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

        self.model.train()
        # Disable mixed precision training to avoid FP16 gradient scaling issues
        
        for epoch in range(epochs):
            total, batches = 0.0, 0
            pbar = tqdm(loader, desc=f"CLIP FT Epoch {epoch+1}/{epochs}")
            for images, texts in pbar:
                images = images.to(self.device)
                tokens = clip.tokenize(texts, truncate=True).to(self.device)
                optim.zero_grad(set_to_none=True)
                
                # Regular FP32 training to avoid gradient scaling issues
                logits_per_image, logits_per_text = self.model(images, tokens)
                loss = self.contrastive_loss(logits_per_image, logits_per_text, self.device)
                if torch.isnan(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optim.step()
                
                total += loss.item(); batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            print(f"Epoch {epoch+1}: avg loss = {total/max(1,batches):.4f}")

        torch.save(self.model.state_dict(), save_path)
        print(f"Fine-tuned CLIP model saved to {save_path}")
        return self.model

    def load_finetuned(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        return self.model


# ============================================================================
# PART 2: CLIP Reward Model (Calibrated + Gated)
# ============================================================================

class CLIPRewardModel:
    """CLIP reward with calibrated temperature and gating support."""
    def __init__(self, clip_model, preprocess, device=None, action_texts=None):
        self.model = clip_model.eval()
        self.preprocess = preprocess
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.action_texts = action_texts or [
            "slow down and be cautious",
            "maintain current speed",
            "speed up and go faster"
        ]
        with torch.no_grad():
            ttok = clip.tokenize(self.action_texts).to(self.device)
            tfeat = self.model.encode_text(ttok)
            self.text_features = F.normalize(tfeat, dim=-1)
            self.scale = self.model.logit_scale.exp().detach()

    @torch.no_grad()
    def _encode_image(self, frame_np):
        if frame_np.ndim == 2:  # grayscale -> 3ch
            frame_np = np.stack([frame_np]*3, axis=-1)
        img = Image.fromarray(frame_np.astype('uint8'))
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        x = self.model.encode_image(x)
        return F.normalize(x, dim=-1)

    @torch.no_grad()
    def score(self, frame_np):
        """
        Returns:
          r_clip: cosine similarity for best action
          best_action: int
          prob: softmax prob for best action
        """
        img_f = self._encode_image(frame_np)
        logits = (self.scale * (img_f @ self.text_features.T)).softmax(dim=-1)
        best_action = int(logits.argmax(dim=-1).item())
        r_clip = torch.sum(img_f * self.text_features[best_action], dim=-1).item()
        prob = logits[0, best_action].item()
        return float(r_clip), best_action, float(prob)

    def get_action_probabilities(self, frame_np):
        img_f = self._encode_image(frame_np)
        logits = (self.scale * (img_f @ self.text_features.T)).softmax(dim=-1)
        return logits[0].detach().cpu().numpy()


# ============================================================================
# PART 3: Custom Environment Wrapper with CLIP Reward (Gated)
# ============================================================================

class CLIPRewardWrapper(gym.Wrapper):
    """Add CLIP reward only when action == CLIP suggestion and prob ≥ threshold."""
    def __init__(self, env, clip_reward_model, weight_clip=1.2, prob_threshold=0.6):
        super().__init__(env)
        self.rm = clip_reward_model
        self.weight_clip = weight_clip
        self.prob_threshold = prob_threshold

    def step(self, action):
        obs, r_env, terminated, truncated, info = self.env.step(action)
        frame = self._get_frame_from_obs(obs)

        try:
            r_clip, suggested, prob = self.rm.score(frame)
        except Exception as e:
            r_clip, suggested, prob = 0.0, action, 0.0
            print(f"CLIP score error: {e}")

        # gate shaping
        if (int(action) == int(suggested)) and (prob >= self.prob_threshold) and np.isfinite(r_clip):
            r_total = float(r_env) + self.weight_clip * float(r_clip)
            info['clip_applied'] = True
        else:
            r_total = float(r_env)
            info['clip_applied'] = False

        info.update({
            'clip_reward': float(r_clip),
            'clip_prob': float(prob),
            'suggested_action': int(suggested),
            'base_reward': float(r_env)
        })
        return obs, r_total, terminated, truncated, info

    def _get_frame_from_obs(self, obs):
        # obs expected shape (C,H,W) stack; take last frame
        if isinstance(obs, np.ndarray):
            arr = obs
        else:
            arr = np.array(obs)
        if arr.ndim == 3:
            frame = arr[-1]
        elif arr.ndim == 4:
            frame = arr[0, -1]
        else:
            frame = arr
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
        return frame


# ============================================================================
# PART 4: Custom CNN Feature Extractor for DQN/PPO
# ============================================================================

class CustomCNN(BaseFeaturesExtractor):
    """Custom CNN for grayscale stacks"""
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))


# ============================================================================
# PART 5: Training Callback for Logging
# ============================================================================

class TrainingCallback(BaseCallback):
    """Custom callback for logging training metrics"""
    def __init__(self, eval_env, eval_freq=1000, use_wandb=False, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.episode_rewards, self.episode_lengths, self.clip_rewards = [], [], []

    def _on_step(self):
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
            # Collect CLIP rewards from every step, not just episode end
            if 'clip_reward' in info and np.isfinite(info['clip_reward']):
                self.clip_rewards.append(info['clip_reward'])
        if self.n_calls % self.eval_freq == 0:
            mean_reward = float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0.0
            mean_length = float(np.mean(self.episode_lengths[-100:])) if self.episode_lengths else 0.0
            mean_clip = float(np.mean(self.clip_rewards[-100:])) if self.clip_rewards else 0.0
            print(f"\nStep {self.n_calls}: meanR={mean_reward:.2f} len={mean_length:.1f} clip={mean_clip:.4f}")
            if self.use_wandb:
                wandb.log({
                    "train/mean_reward": mean_reward,
                    "train/mean_length": mean_length,
                    "train/mean_clip_reward": mean_clip,
                    "train/step": self.n_calls
                })
        return True


# ============================================================================
# PART 6: Environment Setup (Paper-Aligned)
# ============================================================================

def create_intersection_env(use_clip_reward=True, clip_reward_model=None, seed=42):
    """Create Highway-env intersection environment, aligned with the paper."""
    def make_env():
        env = gym.make("intersection-v1", render_mode=None)
        env.unwrapped.config.update({
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (128, 64),      # paper: (H,W)
                "stack_size": 4,
                "weights": [0.2989, 0.5870, 0.1140],
            },
            "action": {
                "type": "DiscreteMetaAction",
                "lateral": False,
                "longitudinal": True,
                "target_speeds": [0, 4.5, 9],        # slower / idle / faster
            },
            "duration": 30,                          # seconds
            "simulation_frequency": 15,              # Hz
            "initial_vehicle_count": 5,
            "spawn_probability": 0.2,
            "collision_reward": -5,
            "arrived_reward": 2,
            "high_speed_reward": 1,
            "reward_speed_range": [7.0, 9.0],
            "normalize_reward": False,
            "offroad_terminal": False,
        })
        env.reset(seed=seed)
        env = Monitor(env)  # expose info["episode"]

        if use_clip_reward and (clip_reward_model is not None):
            env = CLIPRewardWrapper(env, clip_reward_model, weight_clip=1.2, prob_threshold=0.6)
        return env
    return make_env


# ============================================================================
# PART 7: DQN Training (Paper Hyperparams)
# ============================================================================

def train_dqn_with_clip(
    clip_model_path,
    total_timesteps=8000,
    use_clip=True,
    save_path="models/dqn_clip",
    use_wandb=False
):
    print("="*80)
    print("Training DQN with CLIP Reward Shaping")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    if use_clip:
        if clip_model_path and os.path.exists(clip_model_path):
            clip_model.load_state_dict(torch.load(clip_model_path, map_location=device))
        clip_reward_model = CLIPRewardModel(clip_model, preprocess, device)
    else:
        clip_reward_model = None

    env = DummyVecEnv([create_intersection_env(use_clip, clip_reward_model)])

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )

    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=5e-4,      # paper
        buffer_size=15000,       # paper
        learning_starts=1000,
        batch_size=32,           # paper
        tau=1.0,
        gamma=0.95,              # paper
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=1.0,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log=f"./logs/dqn_{'clip' if use_clip else 'vanilla'}/"
    )

    callback = TrainingCallback(env, eval_freq=1000, use_wandb=use_wandb and WANDB_AVAILABLE)

    print(f"\nStarting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=10,
        progress_bar=True
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\nModel saved to {save_path}")
    return model, env


# ============================================================================
# PART 8: PPO Training (Optional)
# ============================================================================

def train_ppo_with_clip(
    clip_model_path,
    total_timesteps=8000,
    use_clip=True,
    save_path="models/ppo_clip",
    use_wandb=False
):
    print("="*80)
    print("Training PPO with CLIP Reward Shaping")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    if use_clip:
        if clip_model_path and os.path.exists(clip_model_path):
            clip_model.load_state_dict(torch.load(clip_model_path, map_location=device))
        clip_reward_model = CLIPRewardModel(clip_model, preprocess, device)
    else:
        clip_reward_model = None

    env = DummyVecEnv([create_intersection_env(use_clip, clip_reward_model)])

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )

    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=5e-4,
        n_steps=128,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log=f"./logs/ppo_{'clip' if use_clip else 'vanilla'}/"
    )

    callback = TrainingCallback(env, eval_freq=1000, use_wandb=use_wandb and WANDB_AVAILABLE)

    print(f"\nStarting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=10,
        progress_bar=True
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\nModel saved to {save_path}")
    return model, env


# ============================================================================
# PART 9: Evaluation
# ============================================================================

def evaluate_agent(model, n_episodes=50, render=False):
    print("\n" + "="*80)
    print("Evaluating Agent")
    print("="*80)

    env = model.get_env()
    success_count = collision_count = timeout_count = 0
    episode_rewards, episode_lengths = [], []
    clip_rewards_history, actions_history = [], []

    for _ in tqdm(range(n_episodes), desc="Evaluation"):
        obs = env.reset()
        done = [False]
        ep_reward, ep_length = 0.0, 0
        ep_clip_rewards, ep_actions = [], []

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += float(reward[0])
            ep_length += 1
            ep_actions.append(int(action[0]))
            if 'clip_reward' in info[0]:
                ep_clip_rewards.append(float(info[0]['clip_reward']))

            if done[0]:
                if info[0].get('crashed', False):
                    collision_count += 1
                elif info[0].get('arrived_at_destination', False):
                    success_count += 1
                else:
                    timeout_count += 1

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        clip_rewards_history.extend(ep_clip_rewards)
        actions_history.extend(ep_actions)

    success_rate = 100.0 * success_count / n_episodes
    collision_rate = 100.0 * collision_count / n_episodes
    timeout_rate = 100.0 * timeout_count / n_episodes

    print(f"\nEvaluation Results ({n_episodes} episodes):")
    print(f"  Success Rate: {success_rate:.2f}%")
    print(f"  Collision Rate: {collision_rate:.2f}%")
    print(f"  Timeout Rate: {timeout_rate:.2f}%")
    print(f"  Mean Episode Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean Episode Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    if clip_rewards_history:
        print(f"  Mean CLIP Reward: {np.mean(clip_rewards_history):.4f}")

    results = {
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'timeout_rate': timeout_rate,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'actions': actions_history,
        'clip_rewards': clip_rewards_history
    }
    return results


# ============================================================================
# PART 10: Data Collection for CLIP Fine-tuning (from OBS, not render)
# ============================================================================

def _obs_frame_to_rgb_pil(obs_last_uint8):
    # obs_last_uint8 is (H,W) uint8; replicate to 3ch for CLIP preprocess
    img = Image.fromarray(obs_last_uint8)
    return Image.merge("RGB", (img, img, img))

def collect_intersection_data(n_episodes=100, save_path="intersection_dataset.json"):
    """
    Collect (image, description) pairs using stable, high-precision heuristics.
    Saves the LAST grayscale observation frame (not env.render()) as 3ch PNG.
    """
    print("\n" + "="*80)
    print("Collecting Intersection Data for CLIP Fine-tuning")
    print("="*80)

    env = gym.make("intersection-v1", render_mode=None)
    env.unwrapped.config.update({
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],
        },
        "action": {"type": "DiscreteMetaAction", "lateral": False, "longitudinal": True},
        "duration": 30,
        "simulation_frequency": 15,
        "initial_vehicle_count": 5,
        "spawn_probability": 0.2,
    })
    env.reset()
    dataset = []
    img_dir = "intersection_images"
    os.makedirs(img_dir, exist_ok=True)

    for episode in tqdm(range(n_episodes), desc="Collecting data"):
        obs, info = env.reset()
        done, step = False, 0
        while not done:
            # derive label (conservative heuristic)
            ego = env.unwrapped.vehicle
            label = None
            if ego is not None:
                speed = ego.speed
                near = [v for v in env.unwrapped.road.vehicles if v is not ego
                        and np.linalg.norm(v.position - ego.position) < 20]
                if len(near) >= 2 and speed > 5:
                    label = "slow down and be cautious"
                elif speed < 3 and len(near) == 0:
                    label = "speed up and go faster"
                else:
                    label = "maintain current speed"

            last = obs[-1] if obs.ndim == 3 else obs
            if last.max() <= 1.0:
                last_u8 = (last * 255).astype(np.uint8)
            else:
                last_u8 = last.astype(np.uint8)

            if label is not None:
                img = _obs_frame_to_rgb_pil(last_u8)
                img_path = f"{img_dir}/ep{episode}_t{step}.png"
                img.save(img_path)
                dataset.append({"image": img_path, "description": label})

            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

    with open(save_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\nDataset saved to {save_path} | total samples: {len(dataset)}")
    env.close()
    return dataset


# ============================================================================
# PART 11: Main Pipeline
# ============================================================================

def main():
    """Full training pipeline"""
    USE_WANDB = True
    COLLECT_DATA = True    # set True to (re)build dataset
    FINETUNE_CLIP = True   # set True to fine-tune CLIP
    TRAIN_DQN = True
    TRAIN_PPO = True
    TRAIN_VANILLA = True

    if USE_WANDB and not WANDB_AVAILABLE:
        print("wandb not available; continuing without logging.")
        USE_WANDB = False

    if USE_WANDB:
        try:
            wandb.init(project="clip-rldrive", name="full_pipeline")
            print("✓ Weights & Biases initialized")
        except Exception as e:
            print(f"Warning: wandb init failed: {e}\nContinuing without wandb.")
            USE_WANDB = False

    if COLLECT_DATA:
        print("\n" + "="*80)
        print("STEP 1: Data Collection")
        print("="*80)
        collect_intersection_data(n_episodes=100, save_path="intersection_dataset.json")

    clip_model_path = "clip_finetuned.pt"
    if FINETUNE_CLIP:
        print("\n" + "="*80)
        print("STEP 2: Fine-tuning CLIP")
        print("="*80)
        ft = CLIPFineTuner(model_name="ViT-B/32")
        ft.train(
            dataset_path="intersection_dataset.json",
            epochs=15,
            batch_size=32,
            lr=1e-5,
            save_path=clip_model_path
        )

    if TRAIN_DQN:
        print("\n" + "="*80)
        print("STEP 3: Training DQN with CLIP")
        print("="*80)
        dqn_clip_model, _ = train_dqn_with_clip(
            clip_model_path=clip_model_path if os.path.exists(clip_model_path) else None,
            total_timesteps=8000,
            use_clip=True,
            save_path="models/dqn_clip",
            use_wandb=USE_WANDB
        )
        results_dqn_clip = evaluate_agent(dqn_clip_model, n_episodes=50)

        if TRAIN_VANILLA:
            print("\n" + "="*80)
            print("Training Vanilla DQN (no CLIP)")
            print("="*80)
            dqn_vanilla_model, _ = train_dqn_with_clip(
                clip_model_path=None,
                total_timesteps=8000,
                use_clip=False,
                save_path="models/dqn_vanilla",
                use_wandb=USE_WANDB
            )
            results_dqn_vanilla = evaluate_agent(dqn_vanilla_model, n_episodes=50)

    if TRAIN_PPO:
        print("\n" + "="*80)
        print("STEP 4: Training PPO with CLIP")
        print("="*80)
        ppo_clip_model, _ = train_ppo_with_clip(
            clip_model_path=clip_model_path if os.path.exists(clip_model_path) else None,
            total_timesteps=8000,
            use_clip=True,
            save_path="models/ppo_clip",
            use_wandb=USE_WANDB
        )
        results_ppo_clip = evaluate_agent(ppo_clip_model, n_episodes=50)

        if TRAIN_VANILLA:
            print("\n" + "="*80)
            print("Training Vanilla PPO (no CLIP)")
            print("="*80)
            ppo_vanilla_model, _ = train_ppo_with_clip(
                clip_model_path=None,
                total_timesteps=8000,
                use_clip=False,
                save_path="models/ppo_vanilla",
                use_wandb=USE_WANDB
            )
            results_ppo_vanilla = evaluate_agent(ppo_vanilla_model, n_episodes=50)

    print("\n" + "="*80)
    print("FINAL RESULTS (quick view)")
    print("="*80)
    # (Prints happened inside evaluate; add more here if you run multiple variants)

    if USE_WANDB and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
