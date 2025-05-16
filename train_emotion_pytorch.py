import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

def check_dataset_structure(root_dir):
    """Check if root_dir exists and contains subfolders with images."""
    import pathlib
    if not pathlib.Path(root_dir).is_dir():
        raise FileNotFoundError(f"Folder not found: {root_dir}")
    
    subfolders = [f for f in pathlib.Path(root_dir).iterdir() if f.is_dir()]
    if len(subfolders) == 0:
        raise RuntimeError(f"No class subfolders found in {root_dir}. "
                           "Expected folders like 'angry', 'happy', 'sad', 'neutral' containing images.")

def split_file_no_os(file_path, chunk_size_mb=25):
    chunk_size = chunk_size_mb * 1024 * 1024  # Convert MB to bytes
    try:
        with open(file_path, 'rb') as f:
            index = 1
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                with open(f"{file_path}.part{index}", 'wb') as chunk_file:
                    chunk_file.write(chunk)
                print(f"Created: {file_path}.part{index}")
                index += 1
        print("File splitting complete.")
    except Exception as e:
        print(f"Error during file splitting: {e}")

def main():
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset paths
    dataset_root = "D:/emotion detection"  # Update as needed
    train_dir = dataset_root + "/train"
    val_dir = dataset_root + "/val"

    # Check dataset structure
    check_dataset_structure(train_dir)
    check_dataset_structure(val_dir)

    # Transform pipeline
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485]*3, std=[0.229]*3)
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

    print(f"Classes found: {train_dataset.classes}")

    # DataLoader params
    use_cuda = torch.cuda.is_available()
    pin_memory = True if use_cuda else False
    num_workers = 4 if use_cuda else 0

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # Load pretrained MobileNetV2 model with new API
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, 4)  # 4 emotion classes
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Accuracy: {val_acc:.4f}")

    # Save model
    model_path = "emotion_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")

    # Split into 25MB chunks
    split_file_no_os(model_path)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
