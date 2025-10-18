"""
Improved Data Collection with Higher Quality RGB Images
Uses env.render() for better visual quality instead of grayscale observations
"""

import os
import numpy as np
import gymnasium as gym
import highway_env
from PIL import Image
from tqdm import tqdm
import json


def collect_high_quality_data(n_episodes=100, save_path="intersection_dataset_hq.json"):
    """
    Collect high-quality RGB images using env.render()
    Better for CLIP training than low-res grayscale observations
    """
    print("\n" + "="*80)
    print(f"Collecting High-Quality Intersection Data ({n_episodes} episodes)")
    print("Using RGB rendering for better visual quality")
    print("="*80)

    # Use RGB rendering mode for better image quality
    env = gym.make("intersection-v1", render_mode="rgb_array")
    env.unwrapped.config.update({
        "observation": {
            "type": "Kinematics",  # Use kinematics for logic, render for images
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": False,
        },
        "action": {
            "type": "DiscreteMetaAction",
            "lateral": False,
            "longitudinal": True,
        },
        "duration": 30,
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "initial_vehicle_count": 5,
        "spawn_probability": 0.3,  # More vehicles for diverse scenes
        "collision_reward": -5,
        "arrived_reward": 2,
        "high_speed_reward": 1,
        "reward_speed_range": [7.0, 9.0],
        "normalize_reward": False,
        "offroad_terminal": False,
        # Rendering config for better visuals
        "screen_width": 600,
        "screen_height": 600,
        "centering_position": [0.3, 0.5],
        "scaling": 5.5,
        "show_trajectories": False,
    })
    
    dataset = []
    img_dir = "intersection_images_hq"
    os.makedirs(img_dir, exist_ok=True)

    for episode in tqdm(range(n_episodes), desc="Collecting data"):
        obs, info = env.reset()
        done = False
        step = 0
        
        while not done and step < 100:  # Limit steps per episode
            # Get high-quality RGB frame
            rgb_frame = env.render()
            
            # Get ego vehicle info for labeling
            ego = env.unwrapped.vehicle
            label = None
            
            if ego is not None:
                speed = ego.speed
                # Get nearby vehicles
                nearby_vehicles = [
                    v for v in env.unwrapped.road.vehicles 
                    if v is not ego and np.linalg.norm(v.position - ego.position) < 25
                ]
                
                # More sophisticated labeling based on scene
                num_nearby = len(nearby_vehicles)
                
                if num_nearby >= 3 and speed > 6:
                    label = "slow down and be cautious"
                elif num_nearby >= 2 and speed > 7:
                    label = "slow down and be cautious"
                elif num_nearby == 0 and speed < 4:
                    label = "speed up and go faster"
                elif num_nearby <= 1 and speed < 5:
                    label = "speed up and go faster"
                else:
                    label = "maintain current speed"
                
                # Add more descriptive labels based on context
                if num_nearby >= 4:
                    label = "heavy traffic, slow down and be cautious"
                elif num_nearby == 0:
                    label = "clear road, speed up and go faster"

            # Save high-quality image
            if label is not None and rgb_frame is not None:
                img = Image.fromarray(rgb_frame)
                img_path = f"{img_dir}/ep{episode}_t{step}.png"
                img.save(img_path, quality=95)
                
                dataset.append({
                    "image": img_path,
                    "description": label,
                    "metadata": {
                        "episode": episode,
                        "step": step,
                        "speed": float(ego.speed) if ego else 0.0,
                        "nearby_vehicles": num_nearby if ego else 0
                    }
                })

            # Take random action for diverse scenarios
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

    # Save dataset
    with open(save_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\nDataset saved to {save_path}")
    print(f"Total samples: {len(dataset)}")
    print(f"Image directory: {img_dir}")
    print(f"Image size: 600x600 RGB")
    
    env.close()
    return dataset


def visualize_samples(dataset_path="intersection_dataset_hq.json", n_samples=5):
    """Show some sample images from the dataset"""
    import matplotlib.pyplot as plt
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Sample random images
    samples = np.random.choice(dataset, min(n_samples, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(1, n_samples, figsize=(20, 4))
    for idx, sample in enumerate(samples):
        img = Image.open(sample['image'])
        axes[idx].imshow(img)
        axes[idx].set_title(sample['description'], fontsize=8)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
    print(f"\nSample visualization saved to dataset_samples.png")


if __name__ == "__main__":
    # Collect high-quality dataset
    dataset = collect_high_quality_data(n_episodes=100, save_path="intersection_dataset_hq.json")
    
    # Visualize some samples
    try:
        visualize_samples("intersection_dataset_hq.json", n_samples=5)
    except Exception as e:
        print(f"Visualization skipped: {e}")
