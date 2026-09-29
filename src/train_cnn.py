import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import os

from cnn_model import FaceCNN


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# Image transformations
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((50, 37)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((50, 37)),
    transforms.ToTensor()
])

# Load dataset
train_dataset_full = datasets.ImageFolder(
    "dataset",
    transform=train_transform
)

test_dataset_full = datasets.ImageFolder(
    "dataset",
    transform=test_transform
)

dataset = train_dataset_full

print("Classes:", dataset.classes)
print("Total images:", len(dataset))


# Create train/test split
indices = list(range(len(dataset)))

train_indices, test_indices = train_test_split(
    indices,
    test_size=0.25,
    random_state=42,
    stratify=dataset.targets
)


train_dataset = torch.utils.data.Subset(
    train_dataset_full,
    train_indices
)

test_dataset = torch.utils.data.Subset(
    test_dataset_full,
    test_indices
)


# Data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)


# Create CNN
model = FaceCNN(
    num_classes=len(dataset.classes)
)

model = model.to(device)


# Loss function
criterion = nn.CrossEntropyLoss()


# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# Training
epochs = 50

print("\nStarting CNN training...\n")
best_accuracy = 0.0


for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {running_loss / len(train_loader):.4f} "
        f"Training Accuracy: {train_accuracy:.2f}%"
    )
    # Check validation accuracy
model.eval()

correct_test = 0
total_test = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()


test_accuracy = 100 * correct_test / total_test

print(f"Validation Accuracy: {test_accuracy:.2f}%")


# Save best model
if test_accuracy > best_accuracy:

    best_accuracy = test_accuracy

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": dataset.classes,
            "accuracy": best_accuracy
        },
        "models/best_cnn_attendance_model.pth"
    )

    print("Best CNN model saved!")


# Evaluation
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()


test_accuracy = 100 * correct / total

print("\nCNN Test Accuracy:", f"{test_accuracy:.2f}%")
print("CNN training completed.")


# Save model
os.makedirs("models", exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": dataset.classes
    },
    "models/cnn_attendance_model.pth"
)

print("CNN model saved successfully!")