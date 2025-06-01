import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import os

# ---- Model Definition (must match training) ----
class PotatoCNN(nn.Module):
    def __init__(self, num_classes):
        super(PotatoCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),  # 256/8 = 32
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---- Streamlit App ----
st.set_page_config(
    page_title="Potato Disease Classification",
    page_icon=":potato:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Potato Disease Classification")

# Class names (must match your training order)
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Image preprocessing (must match training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load model
@st.cache_resource
def load_model():
    model = PotatoCNN(num_classes=3)
    model_path = os.path.join('saved_models', 'model.pth')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Image uploader
file = st.file_uploader("Upload a potato leaf image", type=["jpg", "jpeg", "png"])

def import_and_predict(image_data, model):
    image = ImageOps.fit(image_data, (256, 256), Image.Resampling.LANCZOS)
    img = transform(image).unsqueeze(0)  # [1, 3, 256, 256]
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, dim=1).numpy()[0]
        predicted_class = class_names[np.argmax(probabilities)]
        confidence = round(100 * np.max(probabilities), 2)
    return predicted_class, confidence

if file is None:
    st.info("Please upload a file.")
else:
    image = Image.open(file).convert("RGB")
    st.image(image, use_column_width=True)
    predicted_class, confidence = import_and_predict(image, model)
    st.success(f"Predicted label: {predicted_class}\nConfidence: {confidence}%")
