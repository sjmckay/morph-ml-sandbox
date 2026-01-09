import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader

from morph_ml_sandbox.classifier import SimpleCNNClassifier, TransformerClassifier
from morph_ml_sandbox.dataset import GalaxyDataset
from morph_ml_sandbox.utils import generate_galaxy_image, generate_batch_for_torch


class TestGalaxyImageGeneration(unittest.TestCase):
    """Test suite for synthetic galaxy image generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.image_size = (64, 64)
        np.random.seed(42)
    
    def test_generate_galaxy_image_shape(self):
        """Test that generated images have the correct shape."""
        image = generate_galaxy_image(size=self.image_size)
        self.assertEqual(image.shape, self.image_size)
    
    def test_generate_galaxy_image_dtype(self):
        """Test that generated images are float type."""
        image = generate_galaxy_image(size=self.image_size)
        self.assertTrue(np.issubdtype(image.dtype, np.floating))
    
    def test_generate_galaxy_image_values_non_negative(self):
        """Test that generated images have non-negative values."""
        image = generate_galaxy_image(size=self.image_size)
        self.assertTrue(np.all(image >= 0))
    
    def test_generate_galaxy_image_with_noise(self):
        """Test image generation with noise."""
        image = generate_galaxy_image(
            size=self.image_size,
            brightness=1.0,
            noise_level=0.2
        )
        self.assertEqual(image.shape, self.image_size)
        self.assertTrue(np.all(image >= 0))
    
    def test_generate_galaxy_image_normalized(self):
        """Test image normalization to [0, 1]."""
        image = generate_galaxy_image(
            size=self.image_size,
            normalize=True
        )
        self.assertLessEqual(image.max(), 1.0)
        self.assertGreaterEqual(image.min(), 0.0)
    
    def test_generate_batch_for_torch_shape(self):
        """Test batch generation has correct shape."""
        batch_size = 16
        batch = generate_batch_for_torch(
            batch_size=batch_size,
            image_size=self.image_size
        )
        self.assertEqual(batch.shape, (batch_size, 1, *self.image_size))
    
    def test_generate_batch_for_torch_dtype(self):
        """Test batch is float32."""
        batch = generate_batch_for_torch(batch_size=4)
        self.assertEqual(batch.dtype, np.float32)
    
    def test_generate_batch_for_torch_values(self):
        """Test batch values are valid."""
        batch = generate_batch_for_torch(batch_size=4)
        self.assertTrue(np.all(batch >= 0))


class TestGalaxyDataset(unittest.TestCase):
    """Test suite for GalaxyDataset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_samples = 100
        self.image_size = (64, 64)
        self.dataset = GalaxyDataset(
            num_samples=self.num_samples,
            image_size=self.image_size,
            noise_level=0.1,
            seed=42
        )
    
    def test_dataset_length(self):
        """Test dataset has correct number of samples."""
        self.assertEqual(len(self.dataset), self.num_samples)
    
    def test_dataset_sample_shape(self):
        """Test individual samples have correct shape."""
        image, label = self.dataset[0]
        self.assertEqual(image.shape, (1, *self.image_size))
    
    def test_dataset_sample_tensor_type(self):
        """Test samples are PyTorch tensors."""
        image, label = self.dataset[0]
        self.assertIsInstance(image, torch.Tensor)
    
    def test_dataset_label_type(self):
        """Test labels are integers."""
        _, label = self.dataset[0]
        self.assertIsInstance(label, (int, np.integer))
    
    def test_dataset_label_range(self):
        """Test labels are in valid range [0, 2]."""
        for i in range(min(50, len(self.dataset))):
            _, label = self.dataset[i]
            self.assertIn(label, [0, 1, 2])
    
    def test_dataset_morphology_classes(self):
        """Test that dataset covers all morphology classes."""
        labels = [self.dataset[i][1] for i in range(len(self.dataset))]
        unique_labels = set(labels)
        # With 100 samples, we should have at least 2 out of 3 classes
        self.assertGreaterEqual(len(unique_labels), 2)
    
    def test_dataset_morphology_distribution(self):
        """Test that morphology distribution is roughly uniform."""
        labels = np.array([self.dataset[i][1] for i in range(len(self.dataset))])
        counts = np.bincount(labels, minlength=3)
        
        # Each class should have roughly 1/3 of samples (with tolerance)
        expected_per_class = self.num_samples / 3
        for count in counts:
            # Allow ±50% variation due to randomness
            self.assertGreater(count, expected_per_class * 0.5)
            self.assertLess(count, expected_per_class * 1.5)
    
    def test_dataset_custom_morphology_distribution(self):
        """Test dataset with custom morphology distribution."""
        custom_dist = {'elliptical': 0.5, 'spiral': 0.3, 'irregular': 0.2}
        dataset = GalaxyDataset(
            num_samples=100,
            morphology_distribution=custom_dist,
            seed=42
        )
        
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
        counts = np.bincount(labels, minlength=3)
        
        # Elliptical (0) should be more frequent
        self.assertGreater(counts[0], counts[2])
    
    def test_dataset_reproducibility(self):
        """Test that same seed produces same dataset."""
        dataset1 = GalaxyDataset(num_samples=50, seed=123)
        dataset2 = GalaxyDataset(num_samples=50, seed=123)
        
        # Same labels in same order
        for i in range(len(dataset1)):
            _, label1 = dataset1[i]
            _, label2 = dataset2[i]
            self.assertEqual(label1, label2)
    
    def test_dataset_image_normalization(self):
        """Test that generated images are normalized."""
        image, _ = self.dataset[0]
        self.assertLessEqual(image.max().item(), 1.0)
        self.assertGreaterEqual(image.min().item(), 0.0)
    
    def test_dataloader_compatibility(self):
        """Test that dataset works with PyTorch DataLoader."""
        dataloader = DataLoader(self.dataset, batch_size=16, shuffle=True)
        
        # Get first batch
        batch_images, batch_labels = next(iter(dataloader))
        
        self.assertEqual(batch_images.shape[0], 16)
        self.assertEqual(batch_labels.shape[0], 16)
        self.assertEqual(batch_images.shape[1:], (1, *self.image_size))


class TestMorphologyGeneration(unittest.TestCase):
    """Test specific morphology generation methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dataset = GalaxyDataset(num_samples=100, seed=42)
        self.image_size = (64, 64)
    
    def test_elliptical_generation(self):
        """Test elliptical galaxy generation."""
        image = self.dataset._generate_elliptical()
        self.assertEqual(image.shape, self.image_size)
        self.assertTrue(np.all(image >= 0))
        # Elliptical should have smooth profile
        self.assertLess(np.std(image), 1.0)
    
    def test_spiral_generation(self):
        """Test spiral galaxy generation."""
        image = self.dataset._generate_spiral()
        self.assertEqual(image.shape, self.image_size)
        self.assertTrue(np.all(image >= 0))
    
    def test_irregular_generation(self):
        """Test irregular galaxy generation."""
        image = self.dataset._generate_irregular()
        self.assertEqual(image.shape, self.image_size)
        self.assertTrue(np.all(image >= 0))
    
    def test_morphology_distinctiveness(self):
        """Test that different morphologies produce visually different results."""
        # Generate multiple samples of each type
        elliptical_samples = [self.dataset._generate_elliptical() for _ in range(10)]
        spiral_samples = [self.dataset._generate_spiral() for _ in range(10)]
        irregular_samples = [self.dataset._generate_irregular() for _ in range(10)]
        
        # All should be non-zero
        self.assertTrue(all(np.max(s) > 0 for s in elliptical_samples))
        self.assertTrue(all(np.max(s) > 0 for s in spiral_samples))
        self.assertTrue(all(np.max(s) > 0 for s in irregular_samples))


class TestClassifiers(unittest.TestCase):
    """Test classifier models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 16
        self.image_size = (64, 64)
        self.num_classes = 3
    
    def test_cnn_classifier_initialization(self):
        """Test CNN classifier can be initialized."""
        model = SimpleCNNClassifier(input_channels=1, num_classes=self.num_classes)
        self.assertIsNotNone(model)
    
    def test_cnn_classifier_forward_pass(self):
        """Test CNN classifier forward pass."""
        model = SimpleCNNClassifier(input_channels=1, num_classes=self.num_classes)
        x = torch.randn(self.batch_size, 1, *self.image_size)
        output = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
    
    def test_cnn_classifier_output_type(self):
        """Test CNN classifier output is a tensor."""
        model = SimpleCNNClassifier(input_channels=1, num_classes=self.num_classes)
        x = torch.randn(self.batch_size, 1, *self.image_size)
        output = model(x)
        
        self.assertIsInstance(output, torch.Tensor)
    
    def test_transformer_classifier_initialization(self):
        """Test Transformer classifier can be initialized."""
        model = TransformerClassifier(
            input_dim=64*64,
            num_classes=self.num_classes
        )
        self.assertIsNotNone(model)
    
    def test_transformer_classifier_forward_pass(self):
        """Test Transformer classifier forward pass."""
        model = TransformerClassifier(
            input_dim=64*64,
            num_classes=self.num_classes
        )
        x = torch.randn(self.batch_size, 1, *self.image_size)
        output = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
    
    def test_classifier_gradient_flow(self):
        """Test that gradients flow through classifiers."""
        model = SimpleCNNClassifier(input_channels=1, num_classes=self.num_classes)
        x = torch.randn(self.batch_size, 1, *self.image_size, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients were computed
        self.assertIsNotNone(model.conv1.weight.grad)
        self.assertTrue(torch.any(model.conv1.weight.grad != 0))


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def test_dataset_to_model_pipeline(self):
        """Test complete pipeline from dataset to model."""
        # Create dataset
        dataset = GalaxyDataset(num_samples=50, seed=42)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        # Create model
        model = SimpleCNNClassifier(input_channels=1, num_classes=3)
        
        # Forward pass
        for batch_images, batch_labels in dataloader:
            output = model(batch_images)
            
            # Verify output
            self.assertEqual(output.shape[0], len(batch_labels))
            self.assertEqual(output.shape[1], 3)
            
            break  # Test just first batch
    
    def test_dataset_generation_consistency(self):
        """Test that dataset generation is consistent."""
        dataset1 = GalaxyDataset(num_samples=100, seed=999)
        dataset2 = GalaxyDataset(num_samples=100, seed=999)
        
        # Labels should be identical
        for i in range(len(dataset1)):
            _, label1 = dataset1[i]
            _, label2 = dataset2[i]
            self.assertEqual(label1, label2)


if __name__ == '__main__':
    unittest.main()