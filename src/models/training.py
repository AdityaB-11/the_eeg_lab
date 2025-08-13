"""
Training utilities for self-supervised EEG foundation models.

This module provides training loops, evaluation metrics, and utilities
for comparative analysis of different self-supervised learning approaches.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Some features will be disabled.")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_type: str = "transformer"
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_epochs: int = 10
    weight_decay: float = 1e-5
    gradient_clip_val: float = 1.0
    
    # Self-supervised specific
    ssl_method: str = "contrastive"  # contrastive, masked, byol
    augmentation_strength: float = 0.1
    
    # Model architecture
    n_channels: int = 64
    seq_length: int = 1000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    
    # Logging and checkpointing
    log_every_n_steps: int = 50
    save_top_k: int = 3
    monitor_metric: str = "val_loss"
    patience: int = 20
    
    # Paths
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


class EEGDataset(Dataset):
    """Dataset class for EEG data."""
    
    def __init__(
        self, 
        data_path: str, 
        transform=None,
        target_transform=None,
        ssl_augmentations=None
    ):
        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform
        self.ssl_augmentations = ssl_augmentations
        
        # Load data (placeholder - implement based on your data format)
        self.data, self.labels = self._load_data()
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load EEG data from files."""
        # Placeholder implementation
        # In practice, load from EDF, MAT, or other formats
        if os.path.exists(self.data_path):
            # Load actual data
            pass
        else:
            # Generate synthetic data for demo
            n_samples = 1000
            data = np.random.randn(n_samples, 1000, 64)  # (samples, time, channels)
            labels = np.random.randint(0, 2, n_samples)  # Binary classification
            return data, labels
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        result = {
            'eeg': torch.FloatTensor(sample),
            'label': torch.LongTensor([label]) if isinstance(label, (int, np.integer)) else label
        }
        
        # Add augmented views for self-supervised learning
        if self.ssl_augmentations:
            aug1 = self.ssl_augmentations(sample)
            aug2 = self.ssl_augmentations(sample)
            result['eeg_aug1'] = torch.FloatTensor(aug1)
            result['eeg_aug2'] = torch.FloatTensor(aug2)
        
        return result


class FoundationModelTrainer(pl.LightningModule):
    """PyTorch Lightning trainer for foundation models."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.to_dict())
        
        # Initialize model
        if TORCH_AVAILABLE:
            from .foundation_models import create_foundation_model
            self.model = create_foundation_model(
                model_type=config.ssl_method,
                config={
                    'n_channels': config.n_channels,
                    'seq_length': config.seq_length,
                    'd_model': config.d_model,
                    'n_heads': config.n_heads,
                    'n_layers': config.n_layers
                }
            )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.test_results = {}
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, "train")
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, "val")
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, "test")
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def _compute_loss(self, batch, stage: str):
        """Compute loss based on SSL method."""
        if self.config.ssl_method == "contrastive":
            return self._contrastive_loss(batch)
        elif self.config.ssl_method == "masked":
            return self._masked_loss(batch)
        elif self.config.ssl_method == "byol":
            return self._byol_loss(batch)
        else:
            return self._supervised_loss(batch)
    
    def _contrastive_loss(self, batch):
        """Compute contrastive learning loss."""
        aug1 = batch['eeg_aug1']
        aug2 = batch['eeg_aug2']
        
        outputs = self.model(aug1, aug2)
        return outputs['contrastive_loss']
    
    def _masked_loss(self, batch):
        """Compute masked reconstruction loss."""
        eeg = batch['eeg']
        outputs = self.model(eeg)
        return outputs['reconstruction_loss']
    
    def _byol_loss(self, batch):
        """Compute BYOL loss."""
        aug1 = batch['eeg_aug1']
        aug2 = batch['eeg_aug2']
        
        outputs = self.model(aug1, aug2)
        return outputs['byol_loss']
    
    def _supervised_loss(self, batch):
        """Compute supervised classification loss."""
        eeg = batch['eeg']
        labels = batch['label']
        
        outputs = self.model(eeg)
        if 'classification_output' in outputs:
            return nn.CrossEntropyLoss()(outputs['classification_output'], labels.squeeze())
        else:
            return torch.tensor(0.0, requires_grad=True)
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.num_epochs,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        if hasattr(self.model, 'update_target_network'):
            self.model.update_target_network()


class ModelComparator:
    """Utility class for comparing different foundation models."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.results = {}
        self.models = {}
    
    def train_model(
        self, 
        model_name: str, 
        ssl_method: str,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, Any]:
        """Train a single model and return results."""
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}
        
        # Update config for this model
        config = TrainingConfig(**self.config.to_dict())
        config.ssl_method = ssl_method
        
        # Initialize trainer
        model = FoundationModelTrainer(config)
        
        # Setup logging
        logger = TensorBoardLogger(
            save_dir=config.log_dir,
            name=f"{model_name}_{ssl_method}"
        )
        
        # Setup callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(config.checkpoint_dir, model_name),
            filename=f"{ssl_method}_{{epoch:02d}}_{{val_loss:.2f}}",
            monitor=config.monitor_metric,
            save_top_k=config.save_top_k,
            mode="min"
        )
        
        early_stop_callback = EarlyStopping(
            monitor=config.monitor_metric,
            patience=config.patience,
            mode="min"
        )
        
        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=config.num_epochs,
            logger=logger,
            callbacks=[checkpoint_callback, early_stop_callback],
            gradient_clip_val=config.gradient_clip_val,
            log_every_n_steps=config.log_every_n_steps
        )
        
        # Train model
        start_time = datetime.now()
        trainer.fit(model, train_loader, val_loader)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Store results
        results = {
            'model_name': model_name,
            'ssl_method': ssl_method,
            'best_val_loss': trainer.callback_metrics.get('val_loss', float('inf')),
            'training_time': training_time,
            'best_model_path': checkpoint_callback.best_model_path,
            'final_lr': trainer.optimizers[0].param_groups[0]['lr']
        }
        
        self.results[f"{model_name}_{ssl_method}"] = results
        self.models[f"{model_name}_{ssl_method}"] = model
        
        return results
    
    def compare_methods(
        self,
        methods: List[str],
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader = None
    ) -> Dict[str, Any]:
        """Compare multiple self-supervised methods."""
        comparison_results = {
            'methods': methods,
            'individual_results': {},
            'comparison_metrics': {},
            'best_method': None
        }
        
        best_loss = float('inf')
        best_method = None
        
        for method in methods:
            print(f"Training with method: {method}")
            
            # Train model
            results = self.train_model(
                model_name="foundation_model",
                ssl_method=method,
                train_loader=train_loader,
                val_loader=val_loader
            )
            
            comparison_results['individual_results'][method] = results
            
            # Track best method
            val_loss = results.get('best_val_loss', float('inf'))
            if val_loss < best_loss:
                best_loss = val_loss
                best_method = method
        
        comparison_results['best_method'] = best_method
        
        # Compute comparison metrics
        self._compute_comparison_metrics(comparison_results)
        
        # Test best model if test loader provided
        if test_loader and best_method:
            test_results = self._evaluate_model(best_method, test_loader)
            comparison_results['test_results'] = test_results
        
        return comparison_results
    
    def _compute_comparison_metrics(self, results: Dict):
        """Compute metrics for method comparison."""
        methods = results['methods']
        individual_results = results['individual_results']
        
        # Training time comparison
        training_times = [individual_results[m]['training_time'] for m in methods]
        results['comparison_metrics']['avg_training_time'] = np.mean(training_times)
        results['comparison_metrics']['training_time_std'] = np.std(training_times)
        
        # Performance comparison
        val_losses = [individual_results[m]['best_val_loss'] for m in methods]
        results['comparison_metrics']['performance_ranking'] = sorted(
            zip(methods, val_losses), key=lambda x: x[1]
        )
    
    def _evaluate_model(self, method: str, test_loader: DataLoader) -> Dict:
        """Evaluate a trained model on test set."""
        model_key = f"foundation_model_{method}"
        if model_key not in self.models:
            return {"error": "Model not found"}
        
        model = self.models[model_key]
        model.eval()
        
        test_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                loss = model._compute_loss(batch, "test")
                test_loss += loss.item()
                num_batches += 1
        
        avg_test_loss = test_loss / num_batches if num_batches > 0 else 0
        
        return {
            'test_loss': avg_test_loss,
            'num_test_batches': num_batches
        }
    
    def save_comparison_results(self, filepath: str):
        """Save comparison results to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert results to JSON-serializable format
        serializable_results = {}
        for key, value in self.results.items():
            serializable_results[key] = {
                k: v for k, v in value.items() 
                if not isinstance(v, (torch.Tensor, pl.LightningModule))
            }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"Results saved to {filepath}")
    
    def generate_report(self) -> str:
        """Generate a comprehensive comparison report."""
        report = []
        report.append("=" * 80)
        report.append("SELF-SUPERVISED EEG FOUNDATION MODEL COMPARISON REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if not self.results:
            report.append("No results available. Run comparisons first.")
            return "\n".join(report)
        
        # Summary table
        report.append("SUMMARY OF RESULTS:")
        report.append("-" * 50)
        report.append(f"{'Method':<20} {'Val Loss':<12} {'Time (s)':<12}")
        report.append("-" * 50)
        
        for method, results in self.results.items():
            val_loss = results.get('best_val_loss', 'N/A')
            time = results.get('training_time', 'N/A')
            report.append(f"{method:<20} {val_loss:<12.4f} {time:<12.2f}")
        
        report.append("")
        
        # Best method
        best_method = min(self.results.items(), key=lambda x: x[1].get('best_val_loss', float('inf')))
        report.append(f"BEST PERFORMING METHOD: {best_method[0]}")
        report.append(f"Best Validation Loss: {best_method[1].get('best_val_loss', 'N/A'):.4f}")
        report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS:")
        report.append("-" * 50)
        for method, results in self.results.items():
            report.append(f"\n{method.upper()}:")
            for key, value in results.items():
                if key != 'model_name':
                    report.append(f"  {key}: {value}")
        
        return "\n".join(report)


def create_ssl_augmentations(strength: float = 0.1):
    """Create augmentation pipeline for self-supervised learning."""
    if not TORCH_AVAILABLE:
        return None
    
    from .foundation_models import EEGAugmentations
    
    def augment(x):
        # Apply multiple augmentations
        x = EEGAugmentations.gaussian_noise(torch.FloatTensor(x), strength)
        x = EEGAugmentations.time_masking(x, strength)
        x = EEGAugmentations.channel_dropout(x, strength * 0.5)
        return x.numpy()
    
    return augment


# Utility functions for model evaluation
def evaluate_downstream_task(
    pretrained_model: nn.Module,
    task_data: DataLoader,
    n_classes: int,
    freeze_backbone: bool = True
) -> Dict[str, float]:
    """Evaluate pretrained model on downstream classification task."""
    if not TORCH_AVAILABLE:
        return {"error": "PyTorch not available"}
    
    # Add classification head
    if freeze_backbone:
        for param in pretrained_model.parameters():
            param.requires_grad = False
    
    # Create classifier
    classifier = nn.Sequential(
        pretrained_model,
        nn.Linear(pretrained_model.d_model, n_classes)
    )
    
    # Train classifier (simplified)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    classifier.train()
    for batch in task_data:
        eeg = batch['eeg']
        labels = batch['label']
        
        optimizer.zero_grad()
        outputs = classifier(eeg)
        
        if isinstance(outputs, dict):
            outputs = outputs['global_representation']
        
        loss = criterion(outputs, labels.squeeze())
        loss.backward()
        optimizer.step()
    
    # Evaluate
    classifier.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in task_data:
            eeg = batch['eeg']
            labels = batch['label']
            
            outputs = classifier(eeg)
            if isinstance(outputs, dict):
                outputs = outputs['global_representation']
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'total_samples': total,
        'correct_predictions': correct
    }
