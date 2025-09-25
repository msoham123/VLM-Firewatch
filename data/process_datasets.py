from dataset_loader import DatasetLoader
from dataset_configs import flame_config, flame3_config, flamevision_config, unified_config
from dataset_aggregrator import DatasetAggregator
import json
from pathlib import Path

# Main execution function
def main():
    """Main function to load FLAME datasets and convert to VQA format"""
    
    # Initialize the loader
    loader = DatasetLoader()

    aggregrator = DatasetAggregator(loader)
    
    # Load and aggregate datasets
    print("Loading and aggregating FLAME datasets...")
    all_data = aggregrator.aggregate_datasets(
        flame_path=flame_config["src"],
        # flame3_path=flame3_config["src"],
        # flamevision_path=flamevision_config["src"]
    )
    
    if not all_data:
        print("No data loaded. Please download datasets first.")
        return
    
    # Create train/val/test splits (70/15/15 as specified in project plan)
    train_data, val_data, test_data = aggregrator.create_train_val_test_split(all_data)
    
    # Save the datasets
    unified_src_path = Path(unified_config["src"])
    train_file = aggregrator.save_vqa_dataset(train_data, unified_src_path / "train.json", "train")
    val_file = aggregrator.save_vqa_dataset(val_data, unified_src_path / "val.json", "val") 
    test_file = aggregrator.save_vqa_dataset(test_data, unified_src_path/ "test.json", "test")
    
    # Save complete dataset as well
    complete_file = aggregrator.save_vqa_dataset(all_data, unified_src_path / "complete.json", "complete")
    
    # Show a sample VQA item
    if train_data:
        print("\nSample VQA item:")
        sample = train_data[0]
        print(json.dumps(sample, indent=2))
    
    print(f"\nVQA conversion complete!")
    print(f"Files saved to: {unified_src_path.absolute()}")

if __name__ == "__main__":
    main()
