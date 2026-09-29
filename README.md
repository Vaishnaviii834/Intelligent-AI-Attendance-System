\# Intelligent AI Attendance System



An AI-powered attendance management system that uses \*\*Convolutional Neural Networks (CNN), PyTorch, and OpenCV\*\* to recognize faces through a webcam and automatically record attendance.



\## Features



\- Real-time face detection using OpenCV

\- CNN-based face recognition using PyTorch

\- Confidence-based unknown-person filtering

\- Automatic attendance recording

\- Prevents duplicate attendance for the same person on the same date

\- CSV-based attendance storage

\- Desktop attendance dashboard using Tkinter

\- CNN model evaluation with confusion matrix

\- Train and test scripts included



\## Technologies Used



\- Python

\- PyTorch

\- Torchvision

\- OpenCV

\- Scikit-learn

\- NumPy

\- Matplotlib

\- Tkinter

\- CSV



\## Project Structure



```text

Intelligent-AI-Attendance-System/

│

├── dataset/

│   ├── Student\_A/

│   ├── Student\_B/

│   ├── Student\_C/

│   └── Student\_D/

│

├── models/

│   ├── best\_cnn\_attendance\_model.pth

│   ├── cnn\_attendance\_model.pth

│   └── attendance\_model.pkl

│

├── src/

│   ├── cnn\_model.py

│   ├── train\_cnn.py

│   ├── evaluate\_cnn.py

│   ├── cnn\_webcam\_recognition.py

│   ├── cnn\_webcam\_unknown\_test.py

│   ├── attendance.py

│   ├── attendance\_dashboard.py

│   └── view\_attendance.py

│

├── attendance.csv

├── requirements.txt

├── .gitignore

└── app.py

