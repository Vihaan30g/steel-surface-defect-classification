from torch.utils.data import DataLoader

from dataset import (
    NEUDataset,
    train_transforms
)


train_dataset = NEUDataset(
    root_dir="data/NEU-DET/train/images",
    transform=train_transforms
)

print("Number of samples:")
print(len(train_dataset))

print("\nClasses:")
print(train_dataset.classes)

print("\nClass to index mapping:")
print(train_dataset.class_to_idx)

image, label = train_dataset[0]

print("\nTensor shape:")
print(image.shape)

print("\nLabel:")
print(label)


train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

batch_images, batch_labels = next(iter(train_loader))

print("\nBatch image tensor shape:")
print(batch_images.shape)

print("\nBatch labels:")
print(batch_labels)