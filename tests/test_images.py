from matplotlib import pyplot as plt
import numpy as np

from ml_sandbox.dataset import GalaxyDataset


def visualize_sample_images(dataset: GalaxyDataset, num_images: int = 9):
    """Visualize sample images from the GalaxyDataset.
    
    Args:
        dataset: Instance of GalaxyDataset
        num_images: Number of images to visualize (should be a perfect square)
    """
    assert int(np.sqrt(num_images))**2 == num_images, "num_images should be a perfect square"
    
    plt.figure(figsize=(8, 8))
    for i in range(num_images):
        choice = np.random.randint(0, len(dataset))
        image, label = dataset[choice]
        plt.subplot(int(np.sqrt(num_images)), int(np.sqrt(num_images)), i + 1)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(f"Label: {GalaxyDataset.MORPHOLOGY_NAMES[label]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Create a sample dataset
    sample_dataset = GalaxyDataset(num_samples=50, image_size=(128, 128), noise_level=0.05, seed=999)
    
    # Visualize sample images
    visualize_sample_images(sample_dataset, num_images=9)
