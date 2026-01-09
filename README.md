# morph-ml-sandbox
Make synthetic galaxy images and test ML classifiers/fitters on them, as practice for/benchmarking different methods. 

Files:
- `classifer.py`: types of morphology classifiers (CNN, Transformer)
- `dataset.py`: handles dataset loading for PyTorch
- `fitter.py`: Not implemented yet, will handle morphological fitting
- `train.py`: handles model training
- `utils.py`: TODO, but will hold common util functions

Example usage:

To run a simple classifier on 64 x 64 synthetic galaxy images, you can do the following:


```python
import torch
from torch.utils.data import DataLoader

from morph_ml_sandbox.classifier import SimpleCNNClassifier
from morph_ml_sandbox.dataset import GalaxyDataset
    
dataset = GalaxyDataset(
        num_samples=200,
        image_size=(64, 64),
        noise_level=0.1,
        seed=42
    )
    
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

model = SimpleCNNClassifier(input_channels=1, num_classes=3)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

model.eval()
    
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
```