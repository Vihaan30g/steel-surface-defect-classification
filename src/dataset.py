from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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


train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])