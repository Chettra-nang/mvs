"""
RTX 5090 High-Performance Configuration
Optimized for 32GB VRAM + 64GB RAM
"""

# Data Collection (scaled up)
COLLECT_DATA_EPISODES = 5000  # 10x more data
DATASET_PATH = "large_intersection_dataset.json"

# CLIP Fine-tuning (high-performance)
CLIP_BATCH_SIZE = 256         # 8x larger batches
CLIP_EPOCHS = 50              # More thorough training
CLIP_LR = 1e-5               # Slightly higher for larger batches
CLIP_WORKERS = 8             # More data loading workers

# RL Training (production scale)
RL_TIMESTEPS = 1_000_000     # 125x more training
PARALLEL_ENVS = 16           # Multi-environment training
DQN_BUFFER_SIZE = 100_000    # Larger replay buffer
PPO_N_STEPS = 512            # Larger rollout buffer

# Model scaling
FEATURES_DIM = 1024          # Larger feature extractor
CNN_CHANNELS = [64, 128, 256] # Deeper CNN

# Evaluation
EVAL_EPISODES = 200          # More thorough evaluation

def get_rtx5090_config():
    """Returns optimized config for RTX 5090"""
    return {
        'data_collection': {
            'n_episodes': COLLECT_DATA_EPISODES,
            'save_path': DATASET_PATH
        },
        'clip_training': {
            'batch_size': CLIP_BATCH_SIZE,
            'epochs': CLIP_EPOCHS,
            'lr': CLIP_LR,
            'num_workers': CLIP_WORKERS
        },
        'rl_training': {
            'total_timesteps': RL_TIMESTEPS,
            'n_envs': PARALLEL_ENVS,
            'buffer_size': DQN_BUFFER_SIZE,
            'n_steps': PPO_N_STEPS
        },
        'model': {
            'features_dim': FEATURES_DIM,
            'cnn_channels': CNN_CHANNELS
        },
        'evaluation': {
            'n_episodes': EVAL_EPISODES
        }
    }

# Memory usage estimates
print("RTX 5090 Memory Usage Estimates:")
print(f"CLIP Training: ~16GB VRAM (batch_size={CLIP_BATCH_SIZE})")
print(f"RL Training: ~8GB VRAM ({PARALLEL_ENVS} parallel envs)")
print(f"Dataset Storage: ~{COLLECT_DATA_EPISODES * 0.1:.1f}GB")
print(f"Total Training Time: ~4-6 hours (vs 30min current)")