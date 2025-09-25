
# Main execution function
def main():
    """Main function to load FLAME datasets and convert to VQA format"""
    
    # Initialize the loader
    loader = FLAMEDatasetLoader(base_path="./flame_datasets")
    
    # Check if datasets exist, if not show download instructions
    flame3_path = "./flame_datasets/flame3"
    flamevision_path = "./flame_datasets/flamevision"
    flame_path = "./flame_datasets/flame"
    
    if not any(Path(p).exists() for p in [flame3_path, flamevision_path, flame_path]):
        loader.download_datasets_manually()
        print("\nAfter downloading, run this script again to process the datasets.")
        return
    
    # Load and aggregate datasets
    print("Loading and aggregating FLAME datasets...")
    all_data = loader.aggregate_datasets(
        flame_path=flame_path,
        flame3_path=flame3_path,
        flamevision_path=flamevision_path
    )
    
    if not all_data:
        print("No data loaded. Please download datasets first.")
        return
    
    # Create train/val/test splits (70/15/15 as specified in project plan)
    train_data, val_data, test_data = loader.create_train_val_test_split(all_data)
    
    # Save the datasets
    output_dir = Path("./vqa_datasets")
    train_file = loader.save_vqa_dataset(train_data, output_dir / "train.json", "train")
    val_file = loader.save_vqa_dataset(val_data, output_dir / "val.json", "val") 
    test_file = loader.save_vqa_dataset(test_data, output_dir / "test.json", "test")
    
    # Save complete dataset as well
    complete_file = loader.save_vqa_dataset(all_data, output_dir / "complete.json", "complete")
    
    # Show a sample VQA item
    if train_data:
        print("\nSample VQA item:")
        sample = train_data[0]
        print(json.dumps(sample, indent=2))
    
    print(f"\nVQA conversion complete!")
    print(f"Files saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()
