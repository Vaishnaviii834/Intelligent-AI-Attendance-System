import cv2
import torch
import numpy as np
from torchvision import transforms

from cnn_model import FaceCNN
from attendance import mark_attendance


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load trained CNN
checkpoint = torch.load(
    "models/best_cnn_attendance_model.pth",
    map_location=device
)

person_names = checkpoint["classes"]


model = FaceCNN(
    num_classes=len(person_names)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()


# Image transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((50, 37)),
    transforms.ToTensor()
])


# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# Start webcam
cap = cv2.VideoCapture(0)


last_prediction = None
stable_count = 0


while True:

    ret, frame = cap.read()

    if not ret:
        print("Unable to access camera")
        break


    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )


    for (x, y, w, h) in faces[:1]:

        # Crop face
        face = gray[y:y+h, x:x+w]


        # Convert image for CNN
        face_tensor = transform(
            face
        ).unsqueeze(0).to(device)


        # Prediction
        with torch.no_grad():

            outputs = model(
                face_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )


        prediction = prediction.item()
        confidence = confidence.item()


        name = person_names[prediction]


        # Stabilize prediction
        if prediction == last_prediction:

            stable_count += 1

        else:

            last_prediction = prediction
            stable_count = 1


        # Mark attendance
        if stable_count >= 5:

            mark_attendance(name)

            stable_count = 0


        confidence_text = (
            f"{name} "
            f"({confidence * 100:.1f}%)"
        )


        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )


        # Display prediction
        cv2.putText(
            frame,
            confidence_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )


    # Display webcam
    cv2.imshow(
        "AI Attendance - CNN Recognition",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()