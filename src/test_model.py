import joblib
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import accuracy_score

# Load trained model
model = joblib.load("models/face_recognition_model.pkl")

# Load dataset
lfw_people = fetch_lfw_people(
    min_faces_per_person=70,
    resize=0.4
)

# Use the same test split as training
from sklearn.model_selection import train_test_split

X = lfw_people.images
y = lfw_people.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Flatten test images
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Predict
y_pred = model.predict(X_test_flat)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model testing completed!")
print("Test images:", len(X_test))
print("Test Accuracy:", accuracy)
print("Test Accuracy (%):", accuracy * 100)

from sklearn.metrics import classification_report

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=lfw_people.target_names
    )
)