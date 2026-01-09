import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from morph_ml_sandbox.classifier import SimpleCNNClassifier
from morph_ml_sandbox.dataset import GalaxyDataset

SEED = 999

total_samples = 200
dataset = GalaxyDataset(
        num_samples=total_samples,
        image_size=(64, 64),
        noise_level=0.1,
        seed=SEED
    )
    
train_size = int(total_samples * 0.7)
val_size = int(total_samples * 0.15)
test_size = total_samples - train_size - val_size
# Split dataset
train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(SEED)
)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
model = SimpleCNNClassifier(input_channels=1, num_classes=3)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

model.eval()
    
#train model

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.00001
)
        
batch_images, batch_labels = next(iter(dataloader))
batch_images = batch_images.to(device)

with torch.no_grad():
    outputs = model(batch_images)
    predictions = torch.argmax(outputs, dim=1)

class_names = ['Elliptical', 'Spiral', 'Irregular']
    
print(f"\n   Sample predictions:")
for i in range(min(5, len(predictions))):
    true_label = class_names[batch_labels[i]]
    pred_label = class_names[predictions[i]]
    confidence = torch.softmax(outputs[i], dim=0).max().item()
    print(f"     Image {i+1}: True={true_label:12s} Pred={pred_label:12s} Conf={confidence:.2%}")