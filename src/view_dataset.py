from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt

# Load the dataset
lfw_people = fetch_lfw_people(
    min_faces_per_person=70,
    resize=0.4
)

# Display first 10 images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(lfw_people.images[i], cmap="gray")
    ax.set_title(lfw_people.target_names[lfw_people.target[i]])
    ax.axis("off")

plt.tight_layout()
plt.show()
print("Images shape:", lfw_people.images.shape)
print("Targets shape:", lfw_people.target.shape)
print("Target names:", lfw_people.target_names)