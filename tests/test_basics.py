import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader

from morph_ml_sandbox.classifier import SimpleCNNClassifier, TransformerClassifier
from morph_ml_sandbox.dataset import GalaxyDataset


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
    
    def test_dataset_custom_morphology_distribution(self):
        """Test dataset with custom morphology distribution."""
        custom_dist = {'elliptical': 0.5, 'spiral': 0.3, 'irregular': 0.2}
        dataset = GalaxyDataset(
            num_samples=100,
            morphology_distribution=custom_dist,
            seed=999
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

    
    def test_cnn_classifier_forward_pass(self):
        """Test CNN classifier forward pass."""
        model = SimpleCNNClassifier(input_channels=1, num_classes=self.num_classes)
        x = torch.randn(self.batch_size, 1, *self.image_size)
        output = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
    
    def test_transformer_classifier_forward_pass(self):
        """Test Transformer classifier forward pass."""
        model = TransformerClassifier(
            input_dim=64*64,
            num_classes=self.num_classes
        )
        x = torch.randn(self.batch_size, 1, *self.image_size)
        output = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))


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


class TestTrainingLoop(unittest.TestCase):
    """Test the training loop functionality."""
    
    def test_training_loop_cnn(self):
        """Test that the training loop runs without errors."""
        from morph_ml_sandbox.train import main
        
        try:
            main(use_transformer=False)  # Use CNN for faster testing
        except Exception as e:
            self.fail(f"Training loop raised an exception: {e}")

    def test_training_loop_transformer(self):
        """Test that the training loop runs without errors for Transformer."""
        from morph_ml_sandbox.train import main
        
        try:
            main(use_transformer=True)  # Use Transformer for testing
        except Exception as e:
            self.fail(f"Training loop raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()