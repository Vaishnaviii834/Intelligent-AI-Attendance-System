import cv2
import joblib
from attendance import mark_attendance

last_prediction = None
stable_count = 0 

# Load trained model
data = joblib.load("models/attendance_model.pkl")

model = data["model"]
person_names = data["person_names"]

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Unable to access camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces[:1]:

        # Crop detected face
        face = gray[y:y+h, x:x+w]

        # Resize to model's training image size
        face = cv2.resize(face, (37, 50))

        # Convert to the format expected by the model
        face_flat = face.reshape(1, -1)

        # Predict identity
        probabilities = model.predict_proba(face_flat)[0]
        confidence = probabilities.max()
        prediction = probabilities.argmax()

        UNKNOWN_THRESHOLD = 0.75

        if confidence >= UNKNOWN_THRESHOLD:
            name = person_names[prediction]
        else:
            name = "Unknown"

        if prediction == last_prediction:
            stable_count += 1
        else:
            last_prediction = prediction
            stable_count = 1
        

        if stable_count >= 5:

            if name != "Unknown":
                mark_attendance(name)

            stable_count = 0
        confidence_text = f"{name} ({confidence * 100:.1f}%)"

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            confidence_text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

    cv2.imshow("AI Attendance - Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()