# Example 1: Classification mode
print("=" * 50)
print("Example 1: Classification Mode")
print("=" * 50)

train_dataset = FireDataset(
    json_path='path/to/train.json',
    image_dir='path/to/images',
    transform=get_data_transforms(augment=True),
    mode='classification',
    return_metadata=True
)

# Get a sample
sample = train_dataset[0]
print(f"Image shape: {sample['image'].shape}")
print(f"Label: {sample['label']}")
print(f"Metadata: {sample['metadata']}")

# Create dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2
)

# Iterate through one batch
for batch in train_loader:
    if isinstance(batch, dict):
        images = batch['image']
        labels = batch['label']
    else:
        images, labels = batch
    print(f"Batch - Images: {images.shape}, Labels: {labels.shape}")
    break

# Example 2: VQA mode
print("\n" + "=" * 50)
print("Example 2: VQA Mode")
print("=" * 50)

vqa_dataset = FireDataset(
    json_path='path/to/train.json',
    image_dir='path/to/images',
    transform=get_data_transforms(augment=False),
    mode='vqa',
    return_metadata=True
)

# Get a sample
sample = vqa_dataset[0]
print(f"Image shape: {sample['image'].shape}")
print(f"Question: {sample['question']}")
print(f"Answer: {sample['answer']}")
print(f"Image name: {sample['image_name']}")

# Create dataloader with custom collate
vqa_loader = DataLoader(
    vqa_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=vqa_collate_fn
)

# Iterate through one batch
for batch in vqa_loader:
    print(f"Batch - Images: {batch['images'].shape}")
    print(f"Questions: {batch['questions']}")
    print(f"Answers: {batch['answers']}")
    break

# Example 3: Multi-QA mode (flattened dataset)
print("\n" + "=" * 50)
print("Example 3: Multi-QA Mode (Flattened)")
print("=" * 50)

multi_qa_dataset = FireDatasetMultiQA(
    json_path='path/to/train.json',
    image_dir='path/to/images',
    transform=get_data_transforms(augment=False),
    return_metadata=True
)

print(f"Total Q&A pairs: {len(multi_qa_dataset)}")

# Example 4: Complete dataloader setup
print("\n" + "=" * 50)
print("Example 4: Complete Dataloader Setup")
print("=" * 50)

train_loader, val_loader, test_loader = create_dataloaders(
    train_json='path/to/train.json',
    val_json='path/to/val.json',
    test_json='path/to/test.json',
    image_dir='path/to/images',
    batch_size=32,
    num_workers=4,
    mode='classification'
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")
