import numpy as np

def generate_galaxy_image(size=(64, 64), brightness=1.0, noise_level=0.1):
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

    # Clip values to be non-negative
    image = np.clip(image, 0, None)

    return image

