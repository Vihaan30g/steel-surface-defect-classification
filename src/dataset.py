from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


from config import (
    IMAGE_SIZE,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    FLIP_PROBABILITY,
    ROTATION_RANGE,
    COLOR_JITTER_BRIGHTNESS,
    COLOR_JITTER_CONTRAST
)


class NEUDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = Path(root_dir)
        self.transform = transform

        self.classes = sorted(
            [folder.name for folder in self.root_dir.iterdir()]
        )

        self.class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(self.classes)
        }

        self.samples = []

        for class_name in self.classes:
            class_folder = self.root_dir / class_name
            image_paths = list(class_folder.glob("*.jpg"))
            for image_path in image_paths:
                self.samples.append(
                    (
                        image_path,
                        self.class_to_idx[class_name]
                    )
                )



    def __len__(self):
        return len(self.samples)
    


    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label



# TRAIN TRANSFORMS
train_transforms = transforms.Compose([

    transforms.Resize(IMAGE_SIZE),

    transforms.RandomHorizontalFlip(
        p=FLIP_PROBABILITY
    ),

    transforms.RandomRotation(
        ROTATION_RANGE
    ),

    transforms.ColorJitter(
        brightness=COLOR_JITTER_BRIGHTNESS,
        contrast=COLOR_JITTER_CONTRAST
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=NORMALIZATION_MEAN,
        std=NORMALIZATION_STD
    )
])


# VALIDATION TRANSFORMS
val_transforms = transforms.Compose([

    transforms.Resize(IMAGE_SIZE),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=NORMALIZATION_MEAN,
        std=NORMALIZATION_STD
    )
])