from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

# Load LFW dataset
lfw_people = fetch_lfw_people(
    min_faces_per_person=70,
    resize=0.4
)

# Input images
X = lfw_people.images

# Labels
y = lfw_people.target

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Total images:", len(X))
print("Training images:", len(X_train))
print("Testing images:", len(X_test))
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
# Flatten each 50 × 37 image into a single feature vector
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

print("Flattened training shape:", X_train_flat.shape)
print("Flattened testing shape:", X_test_flat.shape)

from sklearn.neighbors import KNeighborsClassifier

# Create KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train_flat, y_train)

print("KNN model trained successfully!")

# Predict the test images
y_pred = knn.predict(X_test_flat)

# Calculate accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("KNN Accuracy:", accuracy)
print("KNN Accuracy (%):", accuracy * 100)

from sklearn.neural_network import MLPClassifier

# Create Neural Network model
nn = MLPClassifier(
    hidden_layer_sizes=(128,),
    max_iter=500,
    random_state=42
)

# Train the Neural Network
nn.fit(X_train_flat, y_train)

print("Neural Network trained successfully!")

# Predict test images using Neural Network
nn_pred = nn.predict(X_test_flat)

# Calculate Neural Network accuracy
nn_accuracy = accuracy_score(y_test, nn_pred)

print("Neural Network Accuracy:", nn_accuracy)
print("Neural Network Accuracy (%):", nn_accuracy * 100)

from sklearn.metrics import classification_report

print("\nNeural Network Classification Report:")
print(
    classification_report(
        y_test,
        nn_pred,
        target_names=lfw_people.target_names
    )
)

import joblib

joblib.dump(nn, "models/face_recognition_model.pkl")

print("Model saved successfully!")