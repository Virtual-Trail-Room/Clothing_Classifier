# python3 predict.py /path/to/your/image.jpg

#!/usr/bin/env python3
import argparse
import io
import zipfile
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os

# -------------------------------
# 1. Define the ClothingClassifier
# -------------------------------
class ClothingClassifier(nn.Module):
    def __init__(self, num_classes, model_type='resnet', num_frozen_resnet_layers=5):
        super(ClothingClassifier, self).__init__()
        self.model_type = model_type

        # Load a pretrained ResNet50
        resnet = models.resnet50(pretrained=True)

        # Keep all layers except the final FC
        self.resnet_backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Add a bottleneck FC
        self.fc = nn.Linear(resnet.fc.in_features, 1024)

        # Add the final classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        # Freeze some portion of the backbone
        for param in list(self.resnet_backbone.parameters())[:-num_frozen_resnet_layers]:
            param.requires_grad = False

    def forward(self, x):
        x = self.resnet_backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.classifier(x)
        return x

    def predict(self, image, transform, device='cpu'):
        """
        Runs inference on a single PIL Image and returns:
          - The predicted label (e.g., "Blouse")
          - The garment class ("A", "B", or "C")
        """
        # The order of labels in your final layer
        selected_labels = [
            "Blouse", "Cardigan", "Jacket", "Sweater", "Tank", 
            "Tee", "Top", "Jeans", "Shorts", "Skirts", "Dress"
        ]
        label_mapping = {new: name for new, name in enumerate(selected_labels)}

        self.eval()
        image_transformed = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = self(image_transformed)
            predicted_idx = torch.argmax(preds, dim=1).item()

        # Class A if index < 7, B if < 10, else C
        garment_class = "A" if predicted_idx < 7 else ("B" if predicted_idx < 10 else "C")

        return label_mapping[predicted_idx], garment_class

# -------------------------------
# 2. Define the CustomResNetTransform
# -------------------------------
class CustomResNetTransform:
    def __init__(self, strong_aug=False):
        if strong_aug:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __call__(self, image):
        return self.transform(image)

# -------------------------------
# 3. Parse Arguments
# -------------------------------
parser = argparse.ArgumentParser(description="Run clothing classification on an image in-memory.")
parser.add_argument("image_path", help="Path to the input image file.")
parser.add_argument("--zip_file", default="clothing_classifier2-0-1.pth.zip",
                    help="Zip file containing the .pth model (default: clothing_classifier2-0-1.pth.zip)")
parser.add_argument("--model_in_zip", default="clothing_classifier2-0-1.pth",
                    help="Name of the .pth file inside the zip (default: clothing_classifier2-0-1.pth)")
parser.add_argument("--num_classes", type=int, default=11, help="Number of classes in the model (default=11)")
parser.add_argument("--frozen_layers", type=int, default=60,
                    help="Number of backbone layers to freeze (default=60)")
args = parser.parse_args()

# -------------------------------
# 4. Initialize Model + Load from ZIP in Memory
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ClothingClassifier(
    num_classes=args.num_classes,
    num_frozen_resnet_layers=args.frozen_layers,
    model_type='resnet'
).to(device)

if not os.path.exists(args.zip_file):
    raise FileNotFoundError(
        f"ZIP file not found: {args.zip_file} (needed to load the model)."
    )

with zipfile.ZipFile(args.zip_file, 'r') as zip_ref:
    # Check if the .pth file is actually inside
    if args.model_in_zip not in zip_ref.namelist():
        raise FileNotFoundError(
            f"{args.model_in_zip} not found inside {args.zip_file}."
        )
    with zip_ref.open(args.model_in_zip, 'r') as file_in_zip:
        model_bytes = file_in_zip.read()
        model_buffer = io.BytesIO(model_bytes)
        state_dict = torch.load(model_buffer, map_location=device)

model.load_state_dict(state_dict)
model.eval()
print(f"Model loaded from {args.zip_file} -> {args.model_in_zip} in memory.")

# -------------------------------
# 5. Ensure the image exists, run inference
# -------------------------------
if not os.path.exists(args.image_path):
    raise FileNotFoundError(f"Image file {args.image_path} does not exist.")

input_image = Image.open(args.image_path).convert("RGB")

transform = CustomResNetTransform(strong_aug=False)
predicted_label, garment_class = model.predict(input_image, transform, device=device)

print("Predicted label:", predicted_label)
print("Garment class:", garment_class)