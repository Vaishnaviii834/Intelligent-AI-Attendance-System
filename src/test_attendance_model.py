import joblib
import cv2
import os
import numpy as np
from sklearn.metrics import accuracy_score

data = joblib.load("models/attendance_model.pkl")

model = data["model"]
person_names = data["person_names"]

X = []
y = []

for label, person_name in enumerate(person_names):

    person_path = os.path.join("dataset", person_name)

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

predictions = model.predict(X)

print("\nSample predictions:")

for i in range(10):
    actual = person_names[y[i]]
    predicted = person_names[predictions[i]]
    print(f"Actual: {actual} | Predicted: {predicted}")

accuracy = accuracy_score(y, predictions)

print("Attendance Model Accuracy:", accuracy)
print("Attendance Model Accuracy (%):", accuracy * 100)