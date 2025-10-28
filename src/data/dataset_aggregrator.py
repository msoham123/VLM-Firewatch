import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataset_loader import DatasetLoader
from collections import defaultdict


class DatasetAggregator:
    """
    A specialized class for aggregating FLAME datasets and managing VQA format conversion
    Handles dataset aggregation, statistics, splitting, and saving functionality
    """

    def __init__(self, loader: DatasetLoader):
        self.loader = loader

    def validate_dataset_paths(self, flame_path: str = None, flame3_path: str = None, 
                              flamevision_path: str = None) -> Dict[str, bool]:
        """
        Validate that dataset paths exist and are accessible
        
        Args:
            flame_path: Path to original FLAME dataset
            flame3_path: Path to FLAME3 dataset  
            flamevision_path: Path to FlameVision dataset
            
        Returns:
            Dictionary mapping dataset names to validation status
        """
        validation_results = {}
        
        if flame_path:
            validation_results["FLAME"] = Path(flame_path).exists()
        else:
            validation_results["FLAME"] = False
            
        if flame3_path:
            validation_results["FLAME3"] = Path(flame3_path).exists()
        else:
            validation_results["FLAME3"] = False
            
        if flamevision_path:
            validation_results["FlameVision"] = Path(flamevision_path).exists()
        else:
            validation_results["FlameVision"] = False
            
        return validation_results

    def get_aggregation_plan(self, flame_path: str = None, flame3_path: str = None, 
                           flamevision_path: str = None) -> Dict:
        """
        Get a plan of what datasets will be aggregated
        
        Args:
            flame_path: Path to original FLAME dataset
            flame3_path: Path to FLAME3 dataset  
            flamevision_path: Path to FlameVision dataset
            
        Returns:
            Dictionary with aggregation plan details
        """
        validation_results = self.validate_dataset_paths(flame_path, flame3_path, flamevision_path)
        
        plan = {
            "total_datasets": len([p for p in [flame_path, flame3_path, flamevision_path] if p]),
            "available_datasets": sum(validation_results.values()),
            "datasets": {
                "FLAME": {
                    "path": flame_path,
                    "available": validation_results["FLAME"],
                    "expected_samples": "~1000-2000 (varies by structure)"
                },
                "FLAME3": {
                    "path": flame3_path,
                    "available": validation_results["FLAME3"],
                    "expected_samples": "738 (622 Fire + 116 No Fire quartets)"
                },
                "FlameVision": {
                    "path": flamevision_path,
                    "available": validation_results["FlameVision"],
                    "expected_samples": "8600 (5000 fire + 3600 no-fire)"
                }
            }
        }
        
        return plan

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
        dataset_counts = {}
        
        print("="*60)
        print("STARTING DATASET AGGREGATION")
        print("="*60)
        
        # Load each dataset if path exists
        if flame_path:
            print(f"\n🔄 Loading FLAME dataset from: {flame_path}")
            try:
                flame_data = self.loader.load_flame_original_dataset(flame_path)
                all_data.extend(flame_data)
                dataset_counts["FLAME"] = len(flame_data)
                print(f"✅ Successfully loaded {len(flame_data)} samples from FLAME")
            except Exception as e:
                print(f"❌ Error loading FLAME dataset: {e}")
                dataset_counts["FLAME"] = 0
        else:
            print(f"⚠️  FLAME path not provided, skipping...")
            dataset_counts["FLAME"] = 0

        if flame3_path:
            print(f"\n🔄 Loading FLAME3 dataset from: {flame3_path}")
            try:
                flame3_data = self.loader.load_flame3_dataset(flame3_path)
                all_data.extend(flame3_data)
                dataset_counts["FLAME3"] = len(flame3_data)
                print(f"✅ Successfully loaded {len(flame3_data)} samples from FLAME3")
            except Exception as e:
                print(f"❌ Error loading FLAME3 dataset: {e}")
                dataset_counts["FLAME3"] = 0
        else:
            print(f"⚠️  FLAME3 path not provided, skipping...")
            dataset_counts["FLAME3"] = 0

        if flamevision_path:
            print(f"\n🔄 Loading FlameVision dataset from: {flamevision_path}")
            try:
                flamevision_data = self.loader.load_flamevision_dataset(flamevision_path)
                all_data.extend(flamevision_data)
                dataset_counts["FlameVision"] = len(flamevision_data)
                print(f"✅ Successfully loaded {len(flamevision_data)} samples from FlameVision")
            except Exception as e:
                print(f"❌ Error loading FlameVision dataset: {e}")
                dataset_counts["FlameVision"] = 0
        else:
            print(f"⚠️  FlameVision path not provided, skipping...")
            dataset_counts["FlameVision"] = 0

        print(f"\n" + "="*60)
        print(f"AGGREGATION COMPLETE - Total samples: {len(all_data)}")
        print("="*60)
        
        # Print individual dataset counts
        for dataset, count in dataset_counts.items():
            if count > 0:
                print(f"  {dataset}: {count} samples")
            else:
                print(f"  {dataset}: Not loaded")
        
        if all_data:
            self._print_dataset_stats(all_data)
        else:
            print("⚠️  No data was loaded from any dataset!")
        
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
        Split using First Fit Decreasing algorithm for better balance.
        """
        
        def extract_video_id(item):
            image_name = item['image']
            path = item['metadata']['original_path']
            
            if 'frame' in image_name.lower():
                import re
                match = re.search(r'frame(\d+)', image_name.lower())
                if match:
                    frame_num = int(match.group(1))
                    parent_dir = str(Path(path).parent)
                    # Group into sequences of 100 frames
                    sequence_id = frame_num // 100
                    video_id = f"{parent_dir}_seq{sequence_id}"
                else:
                    video_id = str(Path(path).parent)
            else:
                video_id = path
            
            return video_id
        
        # Group by video
        video_groups = defaultdict(list)
        for item in data:
            video_id = extract_video_id(item)
            video_groups[video_id].append(item)
        
        # Separate fire and no-fire videos
        fire_videos = []
        no_fire_videos = []
        
        for video_id, frames in video_groups.items():
            fire_count = sum(1 for frame in frames if frame['metadata']['has_fire'])
            video_info = {
                'video_id': video_id,
                'frames': frames,
                'frame_count': len(frames),
                'is_fire': fire_count > len(frames) / 2
            }
            
            if video_info['is_fire']:
                fire_videos.append(video_info)
            else:
                no_fire_videos.append(video_info)
        
        print(f"\nGrouping Analysis:")
        print(f"  Total samples: {len(data)}")
        print(f"  Unique videos: {len(video_groups)}")
        print(f"  Fire videos: {len(fire_videos)}, No-fire videos: {len(no_fire_videos)}")
        
        # FIRST FIT DECREASING: Sort by size (largest first)
        def split_by_first_fit_decreasing(videos, train_r, val_r, test_r):
            """
            Use First Fit Decreasing bin packing for balanced splits.
            """
            # Sort videos by frame count (descending)
            sorted_videos = sorted(videos, key=lambda v: v['frame_count'], reverse=True)
            
            # Initialize bins
            total_frames = sum(v['frame_count'] for v in videos)
            target_train = total_frames * train_r
            target_val = total_frames * val_r
            target_test = total_frames * test_r
            
            train_videos = []
            val_videos = []
            test_videos = []
            
            train_size = 0
            val_size = 0
            test_size = 0
            
            # Assign each video to the bin that needs it most
            for video in sorted_videos:
                # Calculate how far each bin is from its target
                train_deficit = target_train - train_size
                val_deficit = target_val - val_size
                test_deficit = target_test - test_size
                
                # Assign to bin with largest deficit
                if train_deficit >= val_deficit and train_deficit >= test_deficit:
                    train_videos.append(video)
                    train_size += video['frame_count']
                elif val_deficit >= test_deficit:
                    val_videos.append(video)
                    val_size += video['frame_count']
                else:
                    test_videos.append(video)
                    test_size += video['frame_count']
            
            return train_videos, val_videos, test_videos
        
        # Split fire and no-fire separately (stratification)
        fire_train, fire_val, fire_test = split_by_first_fit_decreasing(
            fire_videos, train_ratio, val_ratio, test_ratio
        )
        no_fire_train, no_fire_val, no_fire_test = split_by_first_fit_decreasing(
            no_fire_videos, train_ratio, val_ratio, test_ratio
        )
        
        # Flatten to frames
        def flatten_videos(video_list):
            frames = []
            for video in video_list:
                frames.extend(video['frames'])
            return frames
        
        train_data = flatten_videos(fire_train) + flatten_videos(no_fire_train)
        val_data = flatten_videos(fire_val) + flatten_videos(no_fire_val)
        test_data = flatten_videos(fire_test) + flatten_videos(no_fire_test)
        
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
        
        # Verification
        total = len(train_data) + len(val_data) + len(test_data)
        print(f"\n✅ Split Results:")
        print(f"  Train: {len(train_data)} ({len(train_data)/total*100:.1f}%) - Target: {train_ratio*100:.1f}%")
        print(f"  Val: {len(val_data)} ({len(val_data)/total*100:.1f}%) - Target: {val_ratio*100:.1f}%")
        print(f"  Test: {len(test_data)} ({len(test_data)/total*100:.1f}%) - Target: {test_ratio*100:.1f}%")
        
        # Class distribution
        for name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
            fire_count = sum(1 for item in split_data if item['metadata']['has_fire'])
            print(f"  {name} fire ratio: {fire_count/len(split_data)*100:.1f}%")
        
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