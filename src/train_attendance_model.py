import cv2
import os
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

dataset_path = "dataset"

X = []
y = []

person_names = sorted(os.listdir(dataset_path))

for label, person_name in enumerate(person_names):

    person_path = os.path.join(dataset_path, person_name)

    for image_name in os.listdir(person_path):

        image_path = os.path.join(person_path, image_name)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        image = cv2.resize(image, (37, 50))

        X.append(image.flatten() / 255.0)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Total images:", len(X))
print("Image features:", X.shape)
print("Classes:", person_names)

model = MLPClassifier(
    hidden_layer_sizes=(128,),
    max_iter=500,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy:", accuracy)
print("Test Accuracy (%):", accuracy * 100)

from sklearn.metrics import classification_report

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=person_names
    )
)

print("Attendance model trained successfully!")

os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "person_names": person_names
    },
    "models/attendance_model.pkl"
)

print("Attendance model saved successfully!")