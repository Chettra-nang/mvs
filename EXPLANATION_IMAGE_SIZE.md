# Why Are the Images So Small?

## Current Issue

The collected images are **128x64 pixels grayscale** because:

1. **RL Optimization**: Small images = faster training for RL agents
2. **Observation Type**: Using `GrayscaleObservation` for the RL agent
3. **Top-down view**: Bird's-eye view, not realistic camera

## Two Solutions

### Solution 1: Use High-Quality RGB for CLIP Training (Recommended)

**Pros:**
- Better image quality (600x600 RGB)
- More visual details for CLIP
- Realistic rendering

**Cons:**
- Larger storage (~50MB vs 2MB for 100 episodes)
- Slower data collection

**Usage:**
```bash
python collect_better_data.py
```

Then modify training to use `intersection_dataset_hq.json`

### Solution 2: Keep Small Images (Current Approach)

**Pros:**
- Fast data collection
- Small storage
- Matches RL agent's view

**Cons:**
- Low resolution
- Less visual information
- May limit CLIP performance

## Recommendation for RTX 5090

For production training on RTX 5090, use **Solution 1**:

```python
# In clip_rl_rtx5090.py, modify data collection:

def collect_intersection_data(n_episodes=5000, save_path="large_intersection_dataset.json"):
    """Use RGB rendering for better CLIP training"""
    env = gym.make("intersection-v1", render_mode="rgb_array")
    env.unwrapped.config.update({
        "screen_width": 600,
        "screen_height": 600,
        # ... other configs
    })
    
    # Use env.render() instead of observation frames
    rgb_frame = env.render()
    img = Image.fromarray(rgb_frame)
    # Save as high-quality PNG
```

## Hybrid Approach (Best of Both Worlds)

For the RL agent:
- Use 128x64 grayscale observations (fast)

For CLIP training:
- Collect 600x600 RGB images separately (high quality)

This way:
- RL training stays fast
- CLIP gets good visual data
- Best performance overall

## Storage Comparison

| Approach | Image Size | Storage (100 eps) | Storage (5000 eps) |
|----------|-----------|-------------------|-------------------|
| Current (128x64 gray) | ~500 bytes | ~2 MB | ~100 MB |
| High-quality (600x600 RGB) | ~50 KB | ~50 MB | ~2.5 GB |

For RTX 5090 with 5000 episodes, expect ~2.5GB storage for high-quality dataset.

## Quick Fix

Run the improved data collection:

```bash
# Collect high-quality data
python collect_better_data.py

# Then use it for CLIP training
# Edit clip_rl_rtx5090.py line with dataset path:
# dataset_path="intersection_dataset_hq.json"
```

This will give you much better images for CLIP training!
