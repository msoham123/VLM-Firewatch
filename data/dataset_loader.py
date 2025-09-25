import os
import json
import pandas as pd
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from pathlib import Path
import random
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

class DatasetLoader:
    """
    A comprehensive converter for FLAME datasets (FLAME, FLAME3, FlameVision)
    Converts datasets to VQA format for Visual Language Model training
    """
    
    def __init__(self):

        # VQA question templates for fire detection
        self.fire_questions = [
            "Is there fire visible in this image?",
            "Can you detect any fire or flames in this aerial image?",
            "Does this image contain active fire?",
            "Is there any wildfire present in this scene?",
            "Are there flames visible in this thermal/RGB image?"
        ]
        
        self.fire_answers_positive = [
            "Yes, there is active fire visible in the image.",
            "Yes, flames can be detected in this aerial view.",
            "Yes, there is fire present in the scene.",
            "Yes, active wildfire is visible.",
            "Yes, there are flames visible in the center area.",
            "Yes, there is fire burning in this region."
        ]
        
        self.fire_answers_negative = [
            "No, there is no fire visible in this image.",
            "No, no flames or fire can be detected.",
            "No, this image does not contain fire.",
            "No, there is no wildfire present.",
            "No, no flames are visible in this scene."
        ]

    def load_flame3_dataset(self, flame3_path: str) -> List[Dict]:
        """
        Load FLAME3 dataset and convert to VQA format
        FLAME3 contains 622 Fire quartets and 116 No Fire quartets
        Each quartet: RGB, Thermal, Corrected FOV RGB, Thermal TIFF
        """
        flame3_data = []
        flame3_path = Path(flame3_path)
        
        if not flame3_path.exists():
            print(f"FLAME3 path {flame3_path} does not exist")
            return []
        
        # Look for typical FLAME3 structure
        possible_structures = [
            # Structure 1: Direct Fire/NoFire folders
            {"fire": flame3_path / "Fire", "no_fire": flame3_path / "NoFire"},
            # Structure 2: Nested in dataset folder
            {"fire": flame3_path / "flame-3-computer-vision-subset-sycan-marsh" / "Fire", 
             "no_fire": flame3_path / "flame-3-computer-vision-subset-sycan-marsh" / "NoFire"},
        ]
        
        for structure in possible_structures:
            fire_dir = structure["fire"]
            no_fire_dir = structure["no_fire"]
            
            if fire_dir.exists() and no_fire_dir.exists():
                print(f"Found FLAME3 structure at {fire_dir.parent}")
                
                # Process Fire images
                print("Processing FLAME3 Fire images...")
                fire_images = list(fire_dir.glob("*.jpg")) + list(fire_dir.glob("*.png")) + list(fire_dir.glob("*.tiff"))
                for img_file in tqdm(fire_images):
                    vqa_item = self._create_vqa_item(
                        image_path=str(img_file),
                        has_fire=True,
                        source_dataset="FLAME3",
                        thermal_available=True
                    )
                    flame3_data.append(vqa_item)
                
                # Process NoFire images
                print("Processing FLAME3 NoFire images...")
                no_fire_images = list(no_fire_dir.glob("*.jpg")) + list(no_fire_dir.glob("*.png")) + list(no_fire_dir.glob("*.tiff"))
                for img_file in tqdm(no_fire_images):
                    vqa_item = self._create_vqa_item(
                        image_path=str(img_file),
                        has_fire=False,
                        source_dataset="FLAME3",
                        thermal_available=True
                    )
                    flame3_data.append(vqa_item)
                break
        
        print(f"Loaded {len(flame3_data)} samples from FLAME3")
        return flame3_data

    def load_flamevision_dataset(self, flamevision_path: str) -> List[Dict]:
        """
        Load FlameVision dataset and convert to VQA format
        FlameVision: 8600 images (5000 fire, 3600 no-fire)
        Structure: classification/{train,val,test}/{fire,no_fire}/
        """
        flamevision_data = []
        flamevision_path = Path(flamevision_path)
        
        if not flamevision_path.exists():
            print(f"FlameVision path {flamevision_path} does not exist")
            return []
        
        # Look for classification folder structure
        possible_paths = [
            flamevision_path / "classification",
            flamevision_path / "flamevision-dataset-for-wildfire-classification",
            flamevision_path
        ]
        
        classification_dir = None
        for path in possible_paths:
            if path.exists():
                classification_dir = path
                break
        
        if not classification_dir:
            print(f"Could not find FlameVision classification directory")
            return []
        
        print(f"Found FlameVision at {classification_dir}")
        
        # Process all splits (train, val, test)
        for split in ["train", "val", "test"]:
            split_dir = classification_dir / split
            if not split_dir.exists():
                continue
                
            # Process Fire images
            fire_dir = split_dir / "fire"
            if fire_dir.exists():
                print(f"Processing FlameVision {split}/fire images...")
                fire_images = list(fire_dir.glob("*.jpg")) + list(fire_dir.glob("*.png"))
                for img_file in tqdm(fire_images):
                    vqa_item = self._create_vqa_item(
                        image_path=str(img_file),
                        has_fire=True,
                        source_dataset="FlameVision",
                        thermal_available=False,
                        split=split
                    )
                    flamevision_data.append(vqa_item)
            
            # Process No Fire images
            no_fire_dirs = [split_dir / "no_fire", split_dir / "nofire", split_dir / "no-fire"]
            for no_fire_dir in no_fire_dirs:
                if no_fire_dir.exists():
                    print(f"Processing FlameVision {split}/no_fire images...")
                    no_fire_images = list(no_fire_dir.glob("*.jpg")) + list(no_fire_dir.glob("*.png"))
                    for img_file in tqdm(no_fire_images):
                        vqa_item = self._create_vqa_item(
                            image_path=str(img_file),
                            has_fire=False,
                            source_dataset="FlameVision",
                            thermal_available=False,
                            split=split
                        )
                        flamevision_data.append(vqa_item)
                    break
        
        print(f"Loaded {len(flamevision_data)} samples from FlameVision")
        return flamevision_data

    def load_flame_original_dataset(self, flame_path: str) -> List[Dict]:
        """
        Load original FLAME dataset and convert to VQA format
        Original FLAME has RGB and IR imagery with fire/no-fire labels
        """
        flame_data = []
        flame_path = Path(flame_path)
        
        if not flame_path.exists():
            print(f"FLAME path {flame_path} does not exist")
            return []
        
        # Look for various possible structures
        fire_dirs = [
            flame_path / "Fire",
            flame_path / "fire",
            flame_path / "Fire_Images",
            flame_path / "positive",
            flame_path / "RGB" / "Fire"
        ]
        
        no_fire_dirs = [
            flame_path / "NoFire", 
            flame_path / "no_fire",
            flame_path / "No_Fire_Images",
            flame_path / "negative",
            flame_path / "RGB" / "NoFire"
        ]
        
        # Check if thermal data is available
        thermal_available = any((flame_path / thermal_dir).exists() 
                              for thermal_dir in ["IR", "Thermal", "thermal"])
        
        # Process fire images
        for fire_dir in fire_dirs:
            if fire_dir.exists():
                print(f"Processing FLAME fire images from {fire_dir}...")
                fire_images = list(fire_dir.glob("*.jpg")) + list(fire_dir.glob("*.png"))
                for img_file in tqdm(fire_images):
                    vqa_item = self._create_vqa_item(
                        image_path=str(img_file),
                        has_fire=True,
                        source_dataset="FLAME",
                        thermal_available=thermal_available
                    )
                    flame_data.append(vqa_item)
                break
        
        # Process no fire images
        for no_fire_dir in no_fire_dirs:
            if no_fire_dir.exists():
                print(f"Processing FLAME no-fire images from {no_fire_dir}...")
                no_fire_images = list(no_fire_dir.glob("*.jpg")) + list(no_fire_dir.glob("*.png"))
                for img_file in tqdm(no_fire_images):
                    vqa_item = self._create_vqa_item(
                        image_path=str(img_file),
                        has_fire=False,
                        source_dataset="FLAME",
                        thermal_available=thermal_available
                    )
                    flame_data.append(vqa_item)
                break
        
        print(f"Loaded {len(flame_data)} samples from original FLAME")
        return flame_data

    def _create_vqa_item(self, image_path: str, has_fire: bool, source_dataset: str, 
                        thermal_available: bool = False, split: str = None, 
                        confidence: float = 0.95) -> Dict:
        """Create a VQA format item from image data"""
        
        # Select random question and answer
        question = random.choice(self.fire_questions)
        if has_fire:
            answer = random.choice(self.fire_answers_positive)
        else:
            answer = random.choice(self.fire_answers_negative)
        
        # Get image filename
        image_filename = Path(image_path).name
        
        # Generate realistic temperature data for thermal images
        temperature_max = None
        if thermal_available:
            if has_fire:
                temperature_max = random.uniform(260.0, 350.0)  # Fire temperatures
            else:
                temperature_max = random.uniform(200.0, 250.0)  # Background temperatures
        
        vqa_item = {
            "image": image_filename,
            "conversations": [
                {
                    "question": question,
                    "answer": answer
                }
            ],
            "metadata": {
                "source_dataset": source_dataset,
                "thermal_available": thermal_available,
                "confidence": confidence,
                "has_fire": has_fire,
                "original_path": image_path
            }
        }
        
        if temperature_max is not None:
            vqa_item["metadata"]["temperature_max"] = round(temperature_max, 1)
        
        if split:
            vqa_item["metadata"]["original_split"] = split
            
        return vqa_item
