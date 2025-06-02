import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Parameters
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 15

# Use the full path for your dataset
DATA_DIR = r"D:\_programming\Potato-Disease-Classification-System-using-Convolutional-Neural-Networks-CNN\training\PlantVillage"
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes
num_classes = len(class_names)
print("Classes:", class_names)

# DataLoader
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Example batch shape
images, labels = next(iter(dataloader))
print("Batch shape:", images.shape)
print("Labels:", labels.numpy())

# Model definition
class PotatoCNN(nn.Module):
    def __init__(self, num_classes):
        super(PotatoCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(CHANNELS, 16, kernel_size=3, padding=1),
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
            nn.Linear(64 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8), 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = PotatoCNN(num_classes).to(device)
print(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataset)
    epoch_acc = correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

# Save model
save_path = os.path.join(SAVE_DIR, "model.pth")
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")
