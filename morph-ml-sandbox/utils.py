import numpy as np

def generate_galaxy_image(size=(64, 64), brightness=1.0, noise_level=0.1, normalize=False):
    """
    Generate a synthetic galaxy image.

    Parameters:
    - size: tuple, the dimensions of the image (height, width)
    - brightness: float, the peak brightness of the galaxy
    - noise_level: float, the standard deviation of Gaussian noise to add

    Returns:
    - image: 2D numpy array representing the galaxy image
    """
    y, x = np.indices(size)
    center = (size[0] // 2, size[1] // 2)
    radius = min(size) // 4

    # Create a simple Gaussian profile for the galaxy
    galaxy = brightness * np.exp(-((x - center[1])**2 + (y - center[0])**2) / (2 * (radius**2)))

    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, size)
    image = galaxy + noise

    # normalize the image to [0, 1]
    if normalize:
        image -= image.min()
        image /= image.max()

    # Clip values to be non-negative
    image = np.clip(image, 0, None)

    return image

def generate_batch_for_torch(batch_size, image_size=(64, 64), brightness=1.0, noise_level=0.1):
    """
    Generate a batch of synthetic galaxy images for PyTorch.

    Parameters:
    - batch_size: int, number of images in the batch
    - image_size: tuple, the dimensions of each image (height, width)
    - brightness: float, the peak brightness of the galaxies
    - noise_level: float, the standard deviation of Gaussian noise to add

    Returns:
    - batch: 4D numpy array with shape (batch_size, 1, height, width)
    """
    batch = np.zeros((batch_size, 1, image_size[0], image_size[1]), dtype=np.float32)
    
    for i in range(batch_size):
        image = generate_galaxy_image(size=image_size, brightness=brightness, noise_level=noise_level)
        batch[i, 0, :, :] = image

    return batch

def train_classifier(model, train_loader, num_epochs=10, learning_rate=0.001):
    """
    Train a PyTorch classifier model.

    Parameters:
    - model: PyTorch nn.Module, the classifier model to train
    - train_loader: DataLoader, the training data loader
    - num_epochs: int, number of epochs to train
    - learning_rate: float, learning rate for the optimizer

    Returns:
    - model: trained PyTorch nn.Module
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    return model


