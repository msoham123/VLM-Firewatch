import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataset_loader import DatasetLoader

class DatasetAggregator:
    """
    A specialized class for aggregating FLAME datasets and managing VQA format conversion
    Handles dataset aggregation, statistics, splitting, and saving functionality
    """

    def __init__(self, loader: DatasetLoader):
        self.loader = loader

    def aggregate_datasets(self, flame_path: str = None, flame3_path: str = None, 
                          flamevision_path: str = None) -> List[Dict]:
        """
        Aggregate all FLAME datasets into unified VQA format
        
        Args:
            flame_path: Path to original FLAME dataset
            flame3_path: Path to FLAME3 dataset
            flamevision_path: Path to FlameVision dataset
            
        Returns:
            List of VQA formatted dataset items
        """
        all_data = []
        
        # Load each dataset if path exists
        if flame_path:
            print(f"Aggregrating modified FLAME dataset from {flame_path}")
            flame_data = self.loader.load_flame_original_dataset(flame_path)
            all_data.extend(flame_data)
        else:
            print(f"FLAME path {flame_path} not found, skipping...")
        

        if flame3_path:
            print(f"Aggregrating modified FLAME3 dataset from {flame3_path}")
            flame3_data = self.loader.load_flame3_dataset(flame3_path)
            all_data.extend(flame3_data)
        else:
            print(f"FLAME3 path {flame3_data} not found, skipping...")

        if flamevision_path:
            print(f"Aggregrating modified FlameVision dataset from {flamevision_path}")
            flamevision_data = self.loader.load_flamevision_dataset(flamevision_path)
            all_data.extend(flamevision_data)
        else:
            print(f"FlameVision path {flamevision_data} not found, skipping...")

        print(f"\nTotal aggregated samples: {len(all_data)}")
        self._print_dataset_stats(all_data)
        
        return all_data

    def _print_dataset_stats(self, data: List[Dict]):
        """
        Print statistics about the aggregated dataset
        
        Args:
            data: List of VQA dataset items
        """
        if not data:
            return
        
        total_samples = len(data)
        fire_samples = sum(1 for item in data if item["metadata"]["has_fire"])
        no_fire_samples = total_samples - fire_samples
        
        datasets = {}
        thermal_available = 0
        
        for item in data:
            source = item["metadata"]["source_dataset"]
            datasets[source] = datasets.get(source, 0) + 1
            if item["metadata"]["thermal_available"]:
                thermal_available += 1
        
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        print(f"Total samples: {total_samples}")
        print(f"Fire samples: {fire_samples} ({fire_samples/total_samples*100:.1f}%)")
        print(f"No-fire samples: {no_fire_samples} ({no_fire_samples/total_samples*100:.1f}%)")
        print(f"Thermal data available: {thermal_available} ({thermal_available/total_samples*100:.1f}%)")
        print("\nBy dataset:")
        for dataset, count in datasets.items():
            print(f"  {dataset}: {count} samples ({count/total_samples*100:.1f}%)")
        print("="*50)

    def create_train_val_test_split(self, data: List[Dict], 
                                  train_ratio: float = 0.7, 
                                  val_ratio: float = 0.15, 
                                  test_ratio: float = 0.15) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split the aggregated data into train, validation, and test sets
        Maintains class balance across splits as specified in your project plan
        
        Args:
            data: List of VQA dataset items
            train_ratio: Proportion of data for training (default: 0.7)
            val_ratio: Proportion of data for validation (default: 0.15)
            test_ratio: Proportion of data for testing (default: 0.15)
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Separate fire and no-fire samples
        fire_samples = [item for item in data if item["metadata"]["has_fire"]]
        no_fire_samples = [item for item in data if not item["metadata"]["has_fire"]]
        
        # Shuffle the samples
        random.shuffle(fire_samples)
        random.shuffle(no_fire_samples)
        
        def split_samples(samples, train_r, val_r, test_r):
            n = len(samples)
            train_end = int(n * train_r)
            val_end = train_end + int(n * val_r)
            
            return (samples[:train_end], 
                   samples[train_end:val_end], 
                   samples[val_end:])
        
        # Split fire samples
        fire_train, fire_val, fire_test = split_samples(fire_samples, train_ratio, val_ratio, test_ratio)
        
        # Split no-fire samples  
        no_fire_train, no_fire_val, no_fire_test = split_samples(no_fire_samples, train_ratio, val_ratio, test_ratio)
        
        # Combine and shuffle
        train_data = fire_train + no_fire_train
        val_data = fire_val + no_fire_val
        test_data = fire_test + no_fire_test
        
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
        
        print(f"\nDataset splits created ({train_ratio*100:.0f}/{val_ratio*100:.0f}/{test_ratio*100:.0f} as per project plan):")
        print(f"Train: {len(train_data)} samples")
        print(f"Validation: {len(val_data)} samples")
        print(f"Test: {len(test_data)} samples")
        
        return train_data, val_data, test_data
    
    def get_dataset_summary(self, data: List[Dict]) -> Dict:
        """
        Get a summary dictionary of dataset statistics
        
        Args:
            data: List of VQA dataset items
            
        Returns:
            Dictionary containing dataset statistics
        """
        if not data:
            return {}
        
        total_samples = len(data)
        fire_samples = sum(1 for item in data if item["metadata"]["has_fire"])
        no_fire_samples = total_samples - fire_samples
        
        datasets = {}
        thermal_available = 0
        
        for item in data:
            source = item["metadata"]["source_dataset"]
            datasets[source] = datasets.get(source, 0) + 1
            if item["metadata"]["thermal_available"]:
                thermal_available += 1
        
        return {
            "total_samples": total_samples,
            "fire_samples": fire_samples,
            "no_fire_samples": no_fire_samples,
            "fire_percentage": fire_samples/total_samples*100 if total_samples > 0 else 0,
            "thermal_available": thermal_available,
            "thermal_percentage": thermal_available/total_samples*100 if total_samples > 0 else 0,
            "datasets": datasets
        }

    def save_vqa_dataset(self, data: List[Dict], output_path: str, split_name: str = ""):
        """Save VQA dataset to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if split_name:
            filename = f"flame_vqa_{split_name}.json"
        else:
            filename = "flame_vqa_dataset.json"
        
        output_file = output_path.parent / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(data)} samples to {output_file}")
        return str(output_file)