#!/usr/bin/env python3
import json
from pathlib import Path

json_path = "/home/hice1/mchidambaram7/scratch/datasets/unified_dataset/flame_vqa_test.json"

with open(json_path, 'r') as f:
    data = json.load(f)

print(f"Total samples: {len(data)}\n")

# Check first 10 samples
for i, item in enumerate(data[:10]):
    metadata = item.get("metadata", {})
    image_path = metadata.get("original_path", "")
    has_fire = metadata.get("has_fire", False)
    
    conversations = item.get("conversations", [])
    answer = conversations[0].get("answer", "") if conversations else ""
    question = conversations[0].get("question", "") if conversations else ""
    
    print(f"Sample {i+1}:")
    print(f"  Path: {image_path[:80]}...")
    print(f"  has_fire (from metadata): {has_fire}")
    print(f"  Question: {question[:60]}...")
    print(f"  Answer: {answer[:100]}...")
    print()

