from sklearn.datasets import fetch_lfw_people
import cv2
import os

lfw_people = fetch_lfw_people(
    min_faces_per_person=70,
    resize=0.4
)

person_name = "Person_4"
person_folder = os.path.join("dataset", person_name)

os.makedirs(person_folder, exist_ok=True)

count = 0

for i, image in enumerate(lfw_people.images):
    if lfw_people.target[i] == 4:   # Gerhard Schroeder
        image_uint8 = (image * 255).astype("uint8")

        file_path = os.path.join(
            person_folder,
            f"image_{count}.jpg"
        )

        cv2.imwrite(file_path, image_uint8)

        count += 1

        if count == 50:
            break

print(f"{count} images saved for {person_name}")