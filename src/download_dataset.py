from sklearn.datasets import fetch_lfw_people

print("Downloading LFW dataset...")

lfw_people = fetch_lfw_people(
    min_faces_per_person=70,
    resize=0.4
)

print("Dataset downloaded successfully!")
print("Number of images:", lfw_people.images.shape[0])
print("Image size:", lfw_people.images.shape[1:])
print("Number of people:", len(lfw_people.target_names))