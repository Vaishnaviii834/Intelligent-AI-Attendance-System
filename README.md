# Intelligent AI Attendance System

This is a Python-based attendance system that uses face recognition to automatically mark attendance.

The project uses a CNN model built with PyTorch for face classification and OpenCV for detecting faces through the webcam.

## Features

- Face detection using OpenCV
- Face recognition using CNN
- Real-time webcam recognition
- Unknown face detection using confidence threshold
- Automatic attendance marking
- Prevents duplicate attendance on the same day
- Attendance stored in CSV file
- Simple desktop dashboard using Tkinter
- Confusion matrix for model evaluation

## Technologies Used

- Python
- PyTorch
- Torchvision
- OpenCV
- Scikit-learn
- NumPy
- Matplotlib
- Tkinter

## How the Project Works

1. The webcam captures the video.
2. OpenCV detects the face from the video.
3. The detected face is resized and converted into a format suitable for the CNN.
4. The CNN predicts the person.
5. The prediction confidence is checked.
6. If the person is recognized, attendance is marked.
7. If the face is not recognized with enough confidence, it is shown as `Unknown`.
8. Attendance is saved in `attendance.csv`.
9. The attendance dashboard displays the recorded entries.

## CNN Model

The CNN contains:

- 3 convolutional layers
- ReLU activation
- Max pooling
- Fully connected layers
- Dropout

The trained model is saved in the `models` folder.

## Model Result

The current CNN model achieved **74% accuracy** on the test split used in this project.

A confusion matrix is also generated to see which classes are being confused by the model.

## Project Structure

```text
Intelligent-AI-Attendance-System/
│
├── dataset/
├── models/
├── src/
├── attendance.csv
├── requirements.txt
├── README.md
└── .gitignore