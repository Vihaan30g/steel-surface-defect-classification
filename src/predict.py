import argparse
from PIL import Image
import torch

from torchvision import transforms

from resnet_model import ResNet18Transfer

from config import (
    IMAGE_SIZE,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD
)


# DEVICE
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# CLASS NAMES
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]


# IMAGE TRANSFORMS
transform = transforms.Compose([

    transforms.Resize(IMAGE_SIZE),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=NORMALIZATION_MEAN,
        std=NORMALIZATION_STD
    )
])


# ARGUMENT PARSER
parser = argparse.ArgumentParser()

parser.add_argument(
    "--image",
    type=str,
    required=True,
    help="Path to image"
)

args = parser.parse_args()


# LOAD MODEL
model = ResNet18Transfer()

model.load_state_dict(
    torch.load(
        "models/best_model.pth",
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()


# LOAD IMAGE
image = Image.open(args.image).convert("RGB")
image = transform(image)
image = image.unsqueeze(0)
image = image.to(DEVICE)


# INFERENCE
with torch.no_grad():

    outputs = model(image)

    _, predicted = torch.max(outputs, 1)

    predicted_class = CLASS_NAMES[
        predicted.item()
    ]


# OUTPUT
print("\nPredicted Defect Class:")
print(predicted_class)