# to visulaize the training dataset initially

import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


DATASET_PATH = Path("data/NEU-DET/train/images")

classes = sorted([folder.name for folder in DATASET_PATH.iterdir()])

print("Classes:")
for idx, class_name in enumerate(classes):
    print(f"{idx}: {class_name}")


fig, axes = plt.subplots(len(classes), 3, figsize=(10, 15))

for row_idx, class_name in enumerate(classes):

    class_folder = DATASET_PATH / class_name

    image_paths = list(class_folder.glob("*.jpg"))

    sampled_images = random.sample(image_paths, 3)

    for col_idx, image_path in enumerate(sampled_images):

        img = Image.open(image_path)

        axes[row_idx, col_idx].imshow(img)
        axes[row_idx, col_idx].set_title(class_name)
        axes[row_idx, col_idx].axis("off")

plt.tight_layout()
plt.show()