import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional


class GalaxyDataset(Dataset):
    """
    PyTorch Dataset for synthetic galaxy images with morphology labels.
    
    Morphologies:
    - 0: Elliptical (smooth, round/elliptical profile)
    - 1: Spiral (disk with spiral arms)
    - 2: Irregular (asymmetric, chaotic structure)
    """
    
    MORPHOLOGY_CLASSES = {
        'elliptical': 0,
        'spiral': 1,
        'irregular': 2
    }
    
    MORPHOLOGY_NAMES = {v: k for k, v in MORPHOLOGY_CLASSES.items()}
    
    def __init__(
        self,
        num_samples: int = 1000,
        image_size: Tuple[int, int] = (64, 64),
        noise_level: float = 0.1,
        morphology_distribution: Optional[dict] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the galaxy dataset.
        
        Args:
            num_samples: Total number of samples to generate
            image_size: Size of generated images (height, width)
            noise_level: Standard deviation of Gaussian noise
            morphology_distribution: Dict with morphology class names as keys and 
                                    probability fractions as values. If None, uniform distribution.
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.noise_level = noise_level
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Set morphology distribution
        if morphology_distribution is None:
            self.morphology_distribution = {
                'elliptical': 1/3,
                'spiral': 1/3,
                'irregular': 1/3
            }
        else:
            self.morphology_distribution = morphology_distribution
        
        # Generate labels for all samples
        self.labels = self._generate_labels()
    
    def _generate_labels(self) -> np.ndarray:
        """Generate random morphology labels based on distribution."""
        morphologies = list(self.morphology_distribution.keys())
        probabilities = list(self.morphology_distribution.values())
        
        morphology_names = np.random.choice(
            morphologies,
            size=self.num_samples,
            p=probabilities
        )
        
        labels = np.array([self.MORPHOLOGY_CLASSES[m] for m in morphology_names])
        return labels
    
    def _generate_elliptical(self) -> np.ndarray:
        """Generate an elliptical galaxy image."""
        y, x = np.indices(self.image_size)
        center = (self.image_size[0] // 2, self.image_size[1] // 2)
        
        # Random axis ratio (0.3 to 1.0)
        axis_ratio = np.random.uniform(0.3, 1.0)
        radius = min(self.image_size) // 4
        
        # Create elliptical profile
        r_x = (x - center[1]) / radius
        r_y = (y - center[0]) / (radius * axis_ratio)
        
        # Smooth Sérsic-like profile (n=4 for elliptical)
        galaxy = np.exp(-((r_x**2 + r_y**2)**0.25))
        
        return galaxy
    
    def _generate_spiral(self) -> np.ndarray:
        """Generate a spiral galaxy image."""
        y, x = np.indices(self.image_size)
        center = (self.image_size[0] // 2, self.image_size[1] // 2)
        
        # Convert to polar coordinates
        dx = x - center[1]
        dy = y - center[0]
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        
        radius = min(self.image_size) // 4
        
        # Disk component
        disk = np.exp(-(r / radius))
        
        # Spiral arms (2-4 arms)
        num_arms = np.random.randint(2, 5)
        spiral_phase = np.random.uniform(0, 2*np.pi)
        
        # Logarithmic spiral: theta_offset = a * log(r)
        a = np.random.uniform(0.2, 0.5)
        theta_offset = a * np.log(r + 1)
        
        # Create spiral pattern
        arm_pattern = np.cos(num_arms * (theta - theta_offset - spiral_phase))**2
        arm_modulation = 1.0 + 0.5 * arm_pattern
        
        galaxy = disk * arm_modulation
        
        return galaxy
    
    def _generate_irregular(self) -> np.ndarray:
        """Generate an irregular galaxy image."""
        y, x = np.indices(self.image_size)
        center = (self.image_size[0] // 2, self.image_size[1] // 2)
        
        # Create base irregular structure using multiple Gaussians at random positions
        galaxy = np.zeros(self.image_size)
        
        num_clumps = np.random.randint(3, 8)
        for _ in range(num_clumps):
            # Random positions for clumps
            cy = np.random.uniform(0.2 * self.image_size[0], 0.8 * self.image_size[0])
            cx = np.random.uniform(0.2 * self.image_size[1], 0.8 * self.image_size[1])
            
            # Random sizes
            sigma = np.random.uniform(3, 10)
            brightness = np.random.uniform(0.3, 1.0)
            
            # Add Gaussian clump
            clump = brightness * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
            galaxy += clump
        
        # Normalize and add random distortion
        galaxy /= (galaxy.max() + 1e-8)
        
        # Add asymmetric distortion
        distortion = np.random.normal(0, 0.1, self.image_size)
        galaxy = galaxy + 0.2 * distortion
        
        return galaxy
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a galaxy image and its morphology label.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image tensor, label)
        """
        label = self.labels[idx]
        morphology = self.MORPHOLOGY_NAMES[label]
        
        # Generate galaxy based on morphology
        if morphology == 'elliptical':
            galaxy = self._generate_elliptical()
        elif morphology == 'spiral':
            galaxy = self._generate_spiral()
        else:  # irregular
            galaxy = self._generate_irregular()
        
        # Add noise
        noise = np.random.normal(0, self.noise_level, self.image_size)
        image = galaxy + noise
        
        # Normalize to [0, 1]
        image = np.clip(image, 0, None)
        image = image / (image.max() + 1e-8)
        
        # Convert to torch tensor with channel dimension
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)
        
        return image_tensor, label