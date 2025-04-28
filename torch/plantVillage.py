import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18, ResNet18_Weights

# Ruta al directorio principal de PlantVillage
data_dir = "/home/daniel/code/2025-phd-daniel-guerrero/dataset/PlantVillage"

# Transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Cargar conjunto completo
full_dataset = ImageFolder(data_dir, transform=transform)
class_names = full_dataset.classes
num_classes = len(class_names)

# Dividir en train, val y test
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Modelo ResNet18 con pesos preentrenados (y sin warnings)
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Congelar capas excepto la última
for param in model.parameters():
    param.requires_grad = False

# Reemplazar la última capa para clasificación
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Usar "cuda" también para GPUs AMD con ROCm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Usando dispositivo: {device}")

# Definir función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Entrenamiento del modelo
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_accuracy = correct_predictions / total_samples
    print(f'[Epoch {epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.4f}')

    # Validación
    model.eval()
    val_loss = 0.0
    val_correct_predictions = 0
    val_total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total_samples += labels.size(0)
            val_correct_predictions += (predicted == labels).sum().item()

    val_epoch_loss = val_loss / len(val_dataset)
    val_epoch_accuracy = val_correct_predictions / val_total_samples
    print(f'           Validation Loss: {val_epoch_loss:.4f} | Validation Accuracy: {val_epoch_accuracy:.4f}')

print("✅ Entrenamiento finalizado.")

# Evaluación en el conjunto de prueba
model.eval()
test_correct_predictions = 0
test_total_samples = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        test_total_samples += labels.size(0)
        test_correct_predictions += (predicted == labels).sum().item()

test_accuracy = test_correct_predictions / test_total_samples
print(f'📊 Precisión en el conjunto de prueba: {test_accuracy:.4f}')

# Guardar el modelo
torch.save(model.state_dict(), "modelo_plantvillage_rocm.pth")

# Para cargar el modelo después:
# loaded_model = resnet18(weights=None)
# loaded_model.fc = nn.Linear(loaded_model.fc.in_features, num_classes)
# loaded_model.load_state_dict(torch.load("modelo_plantvillage_rocm.pth", map_location=device))
# loaded_model.to(device)
