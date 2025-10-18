# Quick Start Guide for RTX 5090

## 🚀 Repository Successfully Pushed!

Your code is now at: **https://github.com/Chettra-nang/mvs**

## 📥 On Your RTX 5090 Machine

### 1. Clone the Repository
```bash
git clone git@github.com:Chettra-nang/mvs.git
cd mvs
```

### 2. Setup Environment
```bash
# Create virtual environment
python -m venv clip_rl_env
source clip_rl_env/bin/activate  # Linux/Mac
# or
clip_rl_env\Scripts\activate     # Windows

# Install PyTorch with CUDA 12.1 (for RTX 5090)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install stable-baselines3[extra]
pip install highway-env
pip install git+https://github.com/openai/CLIP.git
pip install gymnasium
pip install tqdm
pip install pillow
pip install wandb  # optional
```

### 3. Run RTX 5090 Optimized Training
```bash
# Full pipeline with RTX 5090 optimization
python clip_rl_rtx5090.py
```

## 📊 What to Expect

### Training Time (RTX 5090)
- **Data Collection**: ~2-3 hours (5000 episodes)
- **CLIP Fine-tuning**: ~1-2 hours (50 epochs, batch_size=256)
- **DQN Training**: ~2-3 hours (1M timesteps, 16 envs)
- **PPO Training**: ~2-3 hours (1M timesteps, 16 envs)
- **Total**: ~8-12 hours for complete pipeline

### Memory Usage
- **VRAM**: ~16-20GB (out of 32GB available)
- **RAM**: ~32-40GB (out of 64GB available)
- **Storage**: ~60-80GB for dataset

### Expected Performance
- **Dataset**: ~50,000 image-text pairs
- **Success Rate**: 10-30% (intersection is challenging)
- **Collision Rate**: 20-40%
- **Training Speed**: ~100-150 FPS with 16 parallel envs

## 🎛️ Configuration Options

Edit `clip_rl_rtx5090.py` to adjust:

```python
RTX5090_CONFIG = {
    'data_collection_episodes': 5000,    # Reduce if storage limited
    'clip_batch_size': 256,              # Reduce if OOM
    'clip_epochs': 50,                   # Increase for better CLIP
    'rl_timesteps': 1_000_000,          # Increase for better RL
    'parallel_envs': 16,                 # Adjust based on CPU cores
    'features_dim': 1024,                # Network capacity
}
```

## 🐛 Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
'clip_batch_size': 128,  # instead of 256
'parallel_envs': 8,      # instead of 16
```

### Slow Data Collection
```python
# Reduce episodes
'data_collection_episodes': 2000,  # instead of 5000
```

### Want Faster Testing
```bash
# Use the basic version first
python clip-rl.py  # 30min quick test
```

## 📈 Monitoring

### TensorBoard
```bash
tensorboard --logdir=./logs
# Open: http://localhost:6006
```

### Weights & Biases
```bash
wandb login
# Training will auto-log to wandb
```

## 🎯 Files Overview

- `clip_rl_rtx5090.py` - RTX 5090 optimized (1M timesteps, 16 envs)
- `clip-rl.py` - Basic version (8K timesteps, 1 env)
- `rtx5090_config.py` - Configuration reference
- `README.md` - Full documentation

## 💡 Pro Tips

1. **Start with basic version** to verify setup works
2. **Monitor GPU usage** with `nvidia-smi -l 1`
3. **Use tmux/screen** for long training sessions
4. **Enable wandb** for experiment tracking
5. **Save checkpoints** regularly during training

## 🚀 Ready to Train!

Your RTX 5090 will handle this workload easily. The pipeline is optimized to use ~60% of your GPU capacity, leaving headroom for stability.

Good luck with your training! 🎉
