import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from cnn_model import FaceCNN


transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((50, 37)),
    transforms.ToTensor()
])


dataset = datasets.ImageFolder(
    "dataset",
    transform=transform
)


_, test_indices = train_test_split(
    list(range(len(dataset))),
    test_size=0.25,
    random_state=42,
    stratify=dataset.targets
)


test_dataset = torch.utils.data.Subset(
    dataset,
    test_indices
)


test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)


# Load best CNN
model = FaceCNN(
    num_classes=len(dataset.classes)
)

checkpoint = torch.load(
    "models/best_cnn_attendance_model.pth",
    map_location="cpu"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# Predictions
actual = []
predicted = []


with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        actual.extend(labels.tolist())
        predicted.extend(predictions.tolist())


# Confusion matrix
cm = confusion_matrix(
    actual,
    predicted
)


print("\nConfusion Matrix:")
print(cm)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=dataset.classes
)

display.plot(
    xticks_rotation=45
)

plt.title("CNN Face Recognition Confusion Matrix")
plt.tight_layout()

plt.show()