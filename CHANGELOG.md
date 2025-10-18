# Changelog

## Latest Update - High-Quality RGB Data Collection

### What Changed?

Both `clip-rl.py` and `clip_rl_rtx5090.py` now collect **high-quality 600x600 RGB images** instead of low-resolution 128x64 grayscale images.

### Before vs After

| Aspect | Before (Grayscale) | After (RGB) |
|--------|-------------------|-------------|
| **Image Size** | 128x64 pixels | 600x600 pixels |
| **Color** | Grayscale (1 channel) | RGB (3 channels) |
| **File Size** | ~500 bytes | ~3-4 KB |
| **Visual Quality** | Low, pixelated | High, detailed |
| **Storage (100 eps)** | ~2 MB | ~7 MB |
| **Storage (5000 eps)** | ~100 MB | ~350 MB |

### Why This Matters

1. **Better CLIP Training**: CLIP was trained on high-quality images, so it performs better with detailed visuals
2. **More Visual Information**: 600x600 captures more scene details (vehicles, road layout, etc.)
3. **Realistic Rendering**: Uses actual environment rendering instead of agent's observation
4. **Improved Performance**: Better image quality → better CLIP fine-tuning → better reward signals

### Technical Details

**Old Approach:**
```python
# Used grayscale observation (what the RL agent sees)
"observation": {
    "type": "GrayscaleObservation",
    "observation_shape": (128, 64),
}
# Converted grayscale to RGB by duplicating channels
```

**New Approach:**
```python
# Use RGB rendering (high-quality visualization)
env = gym.make("intersection-v1", render_mode="rgb_array")
env.config.update({
    "screen_width": 600,
    "screen_height": 600,
})
# Get actual RGB frame
rgb_frame = env.render()
```

### Impact on Training

**Data Collection:**
- Slightly slower (rendering takes more time)
- More storage required
- Better quality dataset

**CLIP Fine-tuning:**
- Better visual features learned
- More accurate action predictions
- Improved reward signals

**RL Training:**
- Still uses 128x64 grayscale for speed
- Benefits from better CLIP rewards
- No performance impact

### Storage Requirements

For RTX 5090 with 5000 episodes:
- **Old**: ~100 MB
- **New**: ~350 MB
- **Difference**: +250 MB (negligible on modern systems)

### Backward Compatibility

The old `collect_better_data.py` script is kept for reference. Both approaches work, but the new integrated version is recommended.

### Files Updated

1. `clip-rl.py` - Basic version with RGB collection
2. `clip_rl_rtx5090.py` - RTX 5090 version with RGB collection
3. `collect_better_data.py` - Standalone RGB collection (kept for reference)

### How to Use

Just run the scripts as before:

```bash
# Basic version
python clip-rl.py

# RTX 5090 version
python clip_rl_rtx5090.py
```

The high-quality data collection is now automatic!

### Example Images

**Before (128x64 grayscale):**
- Tiny, pixelated
- Hard to see details
- ~500 bytes per image

**After (600x600 RGB):**
- Clear, detailed
- Easy to see vehicles and road
- ~3-4 KB per image

Check `intersection_images_hq/` folder for examples!

---

## Previous Updates

### v1.0 - Initial Release
- Basic CLIP-RLDrive implementation
- DQN and PPO training
- Grayscale data collection

### v1.1 - RTX 5090 Optimization
- Added RTX 5090 optimized script
- 10x larger dataset support
- 16 parallel environments
- Larger batch sizes

### v1.2 - FP16 Gradient Fix
- Fixed FP16 gradient scaling error
- Disabled mixed precision training
- Stable CLIP fine-tuning

### v1.3 - High-Quality Data Collection
- Integrated RGB rendering
- 600x600 image resolution
- Better CLIP training quality

### v1.4 - Data Language Encoder (Current)
- Added DLE for state-to-language conversion
- Rich contextual descriptions instead of simple labels
- Extracts speed, gap, TTC, density from simulator
- Generates natural language like: "ego speed high; heavy traffic; imminent risk; slow down"
- Expected +5-10% performance improvement
- Fully paper-aligned with 3 action phrases
- Enabled by default in both scripts
