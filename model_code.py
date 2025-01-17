import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

# Set the device
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Directories
base_dir = "/Users/aditya/Desktop/ChestScanProject/Data"
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

# Define data transformations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the ResNet-18 model pre-trained on ImageNet
resnet18_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# Modify the architecture
resnet18_model.fc = nn.Sequential(
    nn.Linear(512, 512),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, len(train_dataset.classes))  # Adjust for the number of classes
)

resnet18_model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# Send the model to the device
resnet18_model = resnet18_model.to(device)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.01)
optimizer = optim.SGD(resnet18_model.parameters(), lr=0.01, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6, verbose=True)

# Training loop
def train_model(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, epochs=30):
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = train_correct / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()

        val_loss /= len(valid_loader)
        val_accuracy = val_correct / len(valid_loader.dataset)

        # Update the learning rate scheduler
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Save the best model's state_dict
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'Best_resnet18_model.pth')

# Train the model
train_model(resnet18_model, train_loader, valid_loader, loss_fn, optimizer, scheduler, epochs=30)

# Evaluate on the test set
def evaluate_model(model, test_loader):
    model.eval()
    test_loss = 0
    test_correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = test_correct / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Load the best model's state_dict and evaluate
# Load the best model's state_dict safely
resnet18_model.load_state_dict(torch.load('Best_resnet18_model.pth', map_location=device, weights_only=True))
evaluate_model(resnet18_model, test_loader)

# Save the final model's state_dict
torch.save(resnet18_model.state_dict(), 'Resnet18_model_weights_ChestScan.pth')