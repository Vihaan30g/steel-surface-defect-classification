import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from resnet_model import ResNet18Transfer


from dataset import (
    NEUDataset,
    train_transforms,
    val_transforms
)

from config import (
    LEARNING_RATE,
    EPOCHS,
    BATCH_SIZE
)


# DEVICE
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")


# TENSORBOARD WRITER
writer = SummaryWriter(
    "runs/defect_classifier"
)


# DATASETS
train_dataset = NEUDataset(
    root_dir="data/NEU-DET/train/images",
    transform=train_transforms
)
val_dataset = NEUDataset(
    root_dir="data/NEU-DET/validation/images",
    transform=val_transforms
)


# DATALOADERS
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# MODEL
model = ResNet18Transfer().to(DEVICE)


# LOSS FUNCTION
loss_fn = nn.CrossEntropyLoss()


# OPTIMIZER
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# Main Training loop

best_val_accuracy = 0.0

for epoch in range(EPOCHS):

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")



    # TRAINING PHASE

    model.train()

    train_loss = 0.0

    train_correct = 0

    train_total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = loss_fn(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        train_total += labels.size(0)

        train_correct += (
            predicted == labels
        ).sum().item()

    train_loss = train_loss / len(train_loader)

    train_accuracy = (
        100 * train_correct / train_total
    )



    # VALIDATION PHASE
    model.eval()

    val_loss = 0.0

    val_correct = 0

    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = loss_fn(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_loss = val_loss / len(val_loader)

    val_accuracy = (
        100 * val_correct / val_total
    )



    # PRINT METRICS
    print(f"Train Loss: {train_loss:.4f}")

    print(f"Train Accuracy: {train_accuracy:.2f}%")

    print(f"Validation Loss: {val_loss:.4f}")

    print(f"Validation Accuracy: {val_accuracy:.2f}%")



    # SAVE BEST MODEL
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(
            model.state_dict(),
            "best_model.pth"
        )
        print("Best model saved.")



    # TENSORBOARD LOGGING

    writer.add_scalar(
        "Loss/Train",
        train_loss,
        epoch
    )

    writer.add_scalar(
        "Loss/Validation",
        val_loss,
        epoch
    )

    writer.add_scalar(
        "Accuracy/Train",
        train_accuracy,
        epoch
    )

    writer.add_scalar(
        "Accuracy/Validation",
        val_accuracy,
        epoch
    )



writer.close()

print("\nTraining completed.")