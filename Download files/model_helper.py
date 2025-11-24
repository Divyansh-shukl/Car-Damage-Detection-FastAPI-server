# model_helper.py
import os
import logging
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

logger = logging.getLogger("uvicorn.error")  # Render's logs via uvicorn

class_names = [
    "Front Breakage", "Front Crushed", "Front Normal",
    "Rear Breakage", "Rear Crushed", "Rear Normal"
]

trained_model = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "saved_model.pth")


class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        # IMPORTANT: pass the Python None value (not the string 'None').
        # This constructs the ResNet architecture WITHOUT attempting to
        # download pretrained ImageNet weights.
        self.model = models.resnet50(weights=None)

        # Freeze all params, then unfreeze layer4 (if desired)
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace the final fully connected layer for our num_classes
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def predict(image_path: str):
    # Preprocess transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Validate image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # shape (1,3,224,224)

    global trained_model

    if trained_model is None:
        # Log model path and existence
        logger.info(f"MODEL DEBUG: Looking for model at: {MODEL_PATH}")
        logger.info(f"MODEL DEBUG: Exists? {os.path.exists(MODEL_PATH)}")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Make sure saved_model.pth is in the model/ folder and committed to the repo.")

        # choose device safely (Render is CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"MODEL DEBUG: Using device: {device}")

        # load state dict mapping to device to avoid CUDA/CPU mismatch
        state = torch.load(MODEL_PATH, map_location=device)

        # Build model (no pretrained weights)
        trained_model = CarClassifierResNet()
        # If the saved file is a whole model object (bad practice), torch.load could return a model.
        # We try to load state_dict; if state is a dict, load into model; otherwise, try to use it directly.
        if isinstance(state, dict):
            trained_model.load_state_dict(state)
        else:
            # If someone saved entire model, use it directly (less common).
            logger.warning("MODEL DEBUG: Saved file is not a state_dict; trying to use loaded object as model.")
            trained_model = state

        trained_model.to(device)
        trained_model.eval()
        logger.info("MODEL DEBUG: Model loaded and set to eval()")

    # Make sure tensor is on same device as model
    device = next(trained_model.parameters()).device
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = trained_model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        cls = class_names[predicted_class.item()]
        logger.info(f"PREDICTION DEBUG: predicted {cls}")
        return cls
