import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
from pathlib import Path
import json
from typing import Dict, Tuple, List

from morph_ml_sandbox.classifier import SimpleCNNClassifier, TransformerClassifier
from morph_ml_sandbox.dataset import GalaxyDataset


class GalaxyMorphologyTrainer:
    """Complete training and evaluation pipeline for galaxy morphology classification."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: str = './checkpoints'
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to use ('cuda' or 'cpu')
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        return epoch_loss
    
    def validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float, float]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = running_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Calculate F1 score (macro average for multiclass)
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return val_loss, accuracy, f1
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Complete training loop with validation.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight decay
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc, val_f1 = self.validate(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Store history
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_accuracy'].append(val_acc)
            self.train_history['val_f1'].append(val_f1)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}] | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.4f} | "
                      f"Val F1: {val_f1:.4f}")
        
        print("Training complete!")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'history': self.train_history
        }
        
        filename = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, filename)
        
        if is_best:
            best_filename = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_filename)
            print(f"Saved best model at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_history = checkpoint.get('history', self.train_history)
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def evaluate(
        self,
        test_loader: DataLoader,
        class_names: List[str] = None
    ) -> Dict:
        """
        Comprehensive evaluation on test set.
        
        Args:
            test_loader: Test data loader
            class_names: Names of morphology classes
            
        Returns:
            Dictionary with evaluation metrics
        """
        if class_names is None:
            class_names = ['Elliptical', 'Spiral', 'Irregular']
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Overall Accuracy: {accuracy:.4f}\n")
        
        print("Confusion Matrix:")
        print(cm)
        print()
        
        print("Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        
        results = {
            'accuracy': float(accuracy),
            'confusion_matrix': cm.tolist(),
            'predictions': all_preds.tolist(),
            'labels': all_labels.tolist()
        }
        
        return results


def main():
    """Main training script."""
    # Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    TRAIN_SIZE = 3000
    VAL_SIZE = 500
    TEST_SIZE = 500
    IMAGE_SIZE = (64, 64)
    SEED = 42
    
    # Set random seed for reproducibility
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Create dataset
    print("Generating galaxy dataset...")
    dataset = GalaxyDataset(
        num_samples=TRAIN_SIZE + VAL_SIZE + TEST_SIZE,
        image_size=IMAGE_SIZE,
        noise_level=0.1,
        seed=SEED
    )
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [TRAIN_SIZE, VAL_SIZE, TEST_SIZE],
        generator=torch.Generator().manual_seed(SEED)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    print("Initializing SimpleCNN model...")
    model = SimpleCNNClassifier(input_channels=1, num_classes=3)
    
    # Train
    trainer = GalaxyMorphologyTrainer(model, checkpoint_dir='./checkpoints')
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE
    )
    
    # Evaluate
    print("\nEvaluating on test set...")
    trainer.load_checkpoint('./checkpoints/best_model.pt')
    results = trainer.evaluate(test_loader, class_names=['Elliptical', 'Spiral', 'Irregular'])
    
    # Save results
    results_file = Path('./checkpoints/results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")


if __name__ == '__main__':
    main()