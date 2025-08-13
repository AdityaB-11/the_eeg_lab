"""
Self-Supervised Foundation Models for EEG Analysis

This module contains implementations of various self-supervised learning approaches
for universal EEG representation learning and disease prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class EEGTransformer(nn.Module):
    """
    Transformer-based foundation model for EEG analysis.
    
    This model uses self-attention mechanisms to capture temporal and spatial
    dependencies in EEG signals for universal representation learning.
    """
    
    def __init__(
        self,
        n_channels: int = 64,
        seq_length: int = 1000,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 5000,
        n_classes: int = None
    ):
        super(EEGTransformer, self).__init__()
        
        self.n_channels = n_channels
        self.seq_length = seq_length
        self.d_model = d_model
        self.n_classes = n_classes
        
        # Channel embedding
        self.channel_embedding = nn.Linear(n_channels, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Output layers
        self.layer_norm = nn.LayerNorm(d_model)
        
        # For self-supervised pretraining
        self.ssl_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # For downstream classification (if specified)
        if n_classes is not None:
            self.classification_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, n_classes)
            )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the EEG Transformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, n_channels)
            mask: Optional attention mask
            
        Returns:
            Dictionary containing various outputs
        """
        batch_size, seq_length, n_channels = x.shape
        
        # Channel embedding
        x = self.channel_embedding(x)  # (batch_size, seq_length, d_model)
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        encoded = self.layer_norm(encoded)
        
        # Global representation (mean pooling)
        if mask is not None:
            # Masked mean pooling
            mask_expanded = mask.unsqueeze(-1).expand_as(encoded)
            encoded_masked = encoded.masked_fill(mask_expanded, 0)
            global_repr = encoded_masked.sum(dim=1) / (~mask).sum(dim=1, keepdim=True)
        else:
            global_repr = encoded.mean(dim=1)  # (batch_size, d_model)
        
        outputs = {
            'sequence_output': encoded,  # Full sequence representations
            'global_representation': global_repr,  # Global EEG representation
        }
        
        # Self-supervised learning head
        ssl_output = self.ssl_head(global_repr)
        outputs['ssl_output'] = ssl_output
        
        # Classification head (if available)
        if hasattr(self, 'classification_head'):
            class_output = self.classification_head(global_repr)
            outputs['classification_output'] = class_output
        
        return outputs


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class ContrastiveEEGModel(nn.Module):
    """
    Contrastive learning model for EEG self-supervised pretraining.
    
    Based on SimCLR approach adapted for EEG signals.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        projection_dim: int = 128,
        temperature: float = 0.1
    ):
        super(ContrastiveEEGModel, self).__init__()
        
        self.backbone = backbone
        self.temperature = temperature
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(backbone.d_model, backbone.d_model),
            nn.ReLU(),
            nn.Linear(backbone.d_model, projection_dim)
        )
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for contrastive learning.
        
        Args:
            x1, x2: Augmented views of the same EEG signal
            
        Returns:
            Dictionary with representations and contrastive loss
        """
        # Get representations from backbone
        repr1 = self.backbone(x1)['global_representation']
        repr2 = self.backbone(x2)['global_representation']
        
        # Project to contrastive space
        z1 = F.normalize(self.projection_head(repr1), dim=1)
        z2 = F.normalize(self.projection_head(repr2), dim=1)
        
        # Compute contrastive loss
        loss = self.contrastive_loss(z1, z2)
        
        return {
            'representation_1': repr1,
            'representation_2': repr2,
            'projection_1': z1,
            'projection_2': z2,
            'contrastive_loss': loss
        }
    
    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss (NT-Xent)."""
        batch_size = z1.shape[0]
        
        # Concatenate representations
        z = torch.cat([z1, z2], dim=0)  # (2*batch_size, projection_dim)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature
        
        # Create labels for positive pairs
        labels = torch.cat([torch.arange(batch_size) + batch_size,
                           torch.arange(batch_size)]).to(z.device)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


class MaskedEEGModel(nn.Module):
    """
    Masked language modeling approach for EEG signals.
    
    Similar to BERT but adapted for continuous EEG time series.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        mask_ratio: float = 0.15,
        mask_token_id: int = -1
    ):
        super(MaskedEEGModel, self).__init__()
        
        self.backbone = backbone
        self.mask_ratio = mask_ratio
        self.mask_token_id = mask_token_id
        
        # Reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(backbone.d_model, backbone.d_model),
            nn.ReLU(),
            nn.Linear(backbone.d_model, backbone.n_channels)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with masked reconstruction.
        
        Args:
            x: Input EEG tensor of shape (batch_size, seq_length, n_channels)
            
        Returns:
            Dictionary with original, masked inputs and reconstructions
        """
        batch_size, seq_length, n_channels = x.shape
        
        # Create random mask
        mask = torch.rand(batch_size, seq_length) < self.mask_ratio
        mask = mask.to(x.device)
        
        # Apply mask to input
        x_masked = x.clone()
        x_masked[mask] = self.mask_token_id
        
        # Get representations
        outputs = self.backbone(x_masked, mask=mask)
        sequence_output = outputs['sequence_output']
        
        # Reconstruct masked positions
        reconstructed = self.reconstruction_head(sequence_output)
        
        # Compute reconstruction loss only on masked positions
        recon_loss = F.mse_loss(
            reconstructed[mask], 
            x[mask], 
            reduction='mean'
        )
        
        return {
            'original': x,
            'masked_input': x_masked,
            'reconstructed': reconstructed,
            'mask': mask,
            'reconstruction_loss': recon_loss,
            'global_representation': outputs['global_representation']
        }


class EEGBYOLModel(nn.Module):
    """
    Bootstrap Your Own Latent (BYOL) model adapted for EEG signals.
    
    Self-supervised learning without negative samples.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        projection_dim: int = 256,
        prediction_dim: int = 256,
        tau: float = 0.996
    ):
        super(EEGBYOLModel, self).__init__()
        
        self.tau = tau
        
        # Online network
        self.online_backbone = backbone
        self.online_projector = nn.Sequential(
            nn.Linear(backbone.d_model, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        self.online_predictor = nn.Sequential(
            nn.Linear(projection_dim, prediction_dim),
            nn.BatchNorm1d(prediction_dim),
            nn.ReLU(),
            nn.Linear(prediction_dim, projection_dim)
        )
        
        # Target network (momentum updated)
        self.target_backbone = self._copy_network(backbone)
        self.target_projector = self._copy_network(self.online_projector)
        
        # Disable gradients for target network
        for param in self.target_backbone.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
    
    def _copy_network(self, network: nn.Module) -> nn.Module:
        """Create a copy of the network."""
        import copy
        return copy.deepcopy(network)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for BYOL training.
        
        Args:
            x1, x2: Two augmented views of the same EEG signal
            
        Returns:
            Dictionary with predictions, targets, and loss
        """
        # Online network forward pass
        online_repr1 = self.online_backbone(x1)['global_representation']
        online_repr2 = self.online_backbone(x2)['global_representation']
        
        online_proj1 = self.online_projector(online_repr1)
        online_proj2 = self.online_projector(online_repr2)
        
        online_pred1 = self.online_predictor(online_proj1)
        online_pred2 = self.online_predictor(online_proj2)
        
        # Target network forward pass
        with torch.no_grad():
            target_repr1 = self.target_backbone(x1)['global_representation']
            target_repr2 = self.target_backbone(x2)['global_representation']
            
            target_proj1 = self.target_projector(target_repr1)
            target_proj2 = self.target_projector(target_repr2)
        
        # Compute BYOL loss
        loss1 = self._regression_loss(online_pred1, target_proj2.detach())
        loss2 = self._regression_loss(online_pred2, target_proj1.detach())
        loss = (loss1 + loss2) / 2
        
        return {
            'online_prediction_1': online_pred1,
            'online_prediction_2': online_pred2,
            'target_projection_1': target_proj1,
            'target_projection_2': target_proj2,
            'byol_loss': loss
        }
    
    def _regression_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute normalized regression loss."""
        pred_norm = F.normalize(pred, dim=1)
        target_norm = F.normalize(target, dim=1)
        return 2 - 2 * (pred_norm * target_norm).sum(dim=1).mean()
    
    def update_target_network(self):
        """Update target network with momentum."""
        for online_param, target_param in zip(
            self.online_backbone.parameters(), 
            self.target_backbone.parameters()
        ):
            target_param.data = self.tau * target_param.data + (1 - self.tau) * online_param.data
        
        for online_param, target_param in zip(
            self.online_projector.parameters(), 
            self.target_projector.parameters()
        ):
            target_param.data = self.tau * target_param.data + (1 - self.tau) * online_param.data


class EEGAugmentations:
    """
    Data augmentation techniques for EEG signals in self-supervised learning.
    """
    
    @staticmethod
    def time_masking(x: torch.Tensor, mask_ratio: float = 0.1) -> torch.Tensor:
        """Apply random time masking."""
        batch_size, seq_length, n_channels = x.shape
        mask_length = int(seq_length * mask_ratio)
        
        x_aug = x.clone()
        for i in range(batch_size):
            start_idx = torch.randint(0, seq_length - mask_length + 1, (1,)).item()
            x_aug[i, start_idx:start_idx + mask_length, :] = 0
        
        return x_aug
    
    @staticmethod
    def channel_dropout(x: torch.Tensor, dropout_ratio: float = 0.1) -> torch.Tensor:
        """Apply random channel dropout."""
        batch_size, seq_length, n_channels = x.shape
        n_drop = int(n_channels * dropout_ratio)
        
        x_aug = x.clone()
        for i in range(batch_size):
            drop_channels = torch.randperm(n_channels)[:n_drop]
            x_aug[i, :, drop_channels] = 0
        
        return x_aug
    
    @staticmethod
    def gaussian_noise(x: torch.Tensor, noise_std: float = 0.01) -> torch.Tensor:
        """Add Gaussian noise."""
        noise = torch.randn_like(x) * noise_std
        return x + noise
    
    @staticmethod
    def time_shift(x: torch.Tensor, max_shift: int = 10) -> torch.Tensor:
        """Apply random time shifts."""
        batch_size, seq_length, n_channels = x.shape
        
        x_aug = x.clone()
        for i in range(batch_size):
            shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
            if shift > 0:
                x_aug[i, shift:, :] = x[i, :-shift, :]
                x_aug[i, :shift, :] = 0
            elif shift < 0:
                x_aug[i, :shift, :] = x[i, -shift:, :]
                x_aug[i, shift:, :] = 0
        
        return x_aug
    
    @staticmethod
    def frequency_masking(x: torch.Tensor, mask_ratio: float = 0.1) -> torch.Tensor:
        """Apply frequency domain masking."""
        # Convert to frequency domain
        x_fft = torch.fft.fft(x, dim=1)
        
        # Apply masking in frequency domain
        freq_length = x_fft.shape[1]
        mask_length = int(freq_length * mask_ratio)
        
        x_fft_aug = x_fft.clone()
        start_idx = torch.randint(0, freq_length - mask_length + 1, (1,)).item()
        x_fft_aug[:, start_idx:start_idx + mask_length, :] = 0
        
        # Convert back to time domain
        x_aug = torch.fft.ifft(x_fft_aug, dim=1).real
        
        return x_aug


def create_foundation_model(
    model_type: str = "transformer",
    config: Dict = None
) -> nn.Module:
    """
    Factory function to create different types of foundation models.
    
    Args:
        model_type: Type of model ("transformer", "contrastive", "masked", "byol")
        config: Model configuration dictionary
        
    Returns:
        Initialized foundation model
    """
    if config is None:
        config = {
            'n_channels': 64,
            'seq_length': 1000,
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 6
        }
    
    if model_type == "transformer":
        return EEGTransformer(**config)
    
    elif model_type == "contrastive":
        backbone = EEGTransformer(**config)
        return ContrastiveEEGModel(backbone)
    
    elif model_type == "masked":
        backbone = EEGTransformer(**config)
        return MaskedEEGModel(backbone)
    
    elif model_type == "byol":
        backbone = EEGTransformer(**config)
        return EEGBYOLModel(backbone)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
