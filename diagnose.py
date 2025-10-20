
import json
with open("enhanced_intersection_dataset.json", 'r') as f:
    data = json.load(f)
import random
for item in random.sample(data, 10):
    print(f"Image: {item['image_gray']} | Desc: {item['description']} | Action: {item['action']}")