import cv2
import torch
from torchvision import transforms

from cnn_model import FaceCNN


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Load model
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


# Image preprocessing
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


# Prototype threshold
UNKNOWN_THRESHOLD = 0.75


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

        face = gray[y:y+h, x:x+w]

        face_tensor = transform(
            face
        ).unsqueeze(0).to(device)


        with torch.no_grad():

            outputs = model(face_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )


        confidence = confidence.item()
        prediction = prediction.item()


        if confidence >= UNKNOWN_THRESHOLD:

            name = person_names[prediction]

        else:

            name = "Unknown"


        text = f"{name} ({confidence * 100:.1f}%)"


        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )


        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )


    cv2.imshow(
        "CNN Unknown Face Test",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()