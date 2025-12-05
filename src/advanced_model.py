"""
Advanced Deep Learning Model for NMR Measurement Quality Index (MQI) Prediction

This module implements an advanced architecture with:
1. Attention mechanisms to focus on important features
2. Residual connections for better gradient flow
3. Multi-scale feature extraction
4. Channel-wise and spatial attention
5. Advanced regularization techniques
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Optional
import math

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results")

class ChannelAttention(nn.Module):
    """
    Channel Attention Module (inspired by CBAM).
    
    This module learns to emphasize important channels (features) and
    suppress less useful ones, allowing the network to focus on the most
    informative feature maps.
    """
    
    def __init__(self, channels: int, reduction_ratio: int = 8):
        """
        Initialize channel attention module.
        
        Args:
            channels: Number of input channels
            reduction_ratio: Reduction ratio for the bottleneck layer
        """
        super(ChannelAttention, self).__init__()
        
        # Shared MLP for both max and average pooled features
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            #nn.ReLU(),
            nn.LeakyReLU(0.01),
            nn.Linear(channels // reduction_ratio, channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Attention-weighted tensor of same shape
        """
        batch, channels, _, _ = x.size()
        
        # Global average pooling
        avg_pool = torch.mean(x, dim=[2, 3])  # (batch, channels)
        
        # Global max pooling
        max_pool = torch.max(x.view(batch, channels, -1), dim=2)[0]  # (batch, channels)
        
        # Apply shared MLP to both pooled features
        avg_out = self.fc(avg_pool)
        max_out = self.fc(max_pool)
        
        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out)
        
        # Reshape for broadcasting
        attention = attention.view(batch, channels, 1, 1)
        
        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (inspired by CBAM).
    
    This module learns to emphasize important spatial locations,
    helping the network focus on relevant regions of the input.
    """
    
    def __init__(self, kernel_size: int = 7):
        """
        Initialize spatial attention module.
        
        Args:
            kernel_size: Size of the convolutional kernel
        """
        super(SpatialAttention, self).__init__()
        
        # Convolutional layer to generate spatial attention map
        self.conv = nn.Conv2d(
            2,  # Concatenated max and avg pooled channels
            1,  # Output single attention map
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Attention-weighted tensor of same shape
        """
        # Channel-wise average and max pooling
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # (batch, 1, height, width)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # (batch, 1, height, width)
        
        # Concatenate along channel dimension
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (batch, 2, height, width)
        
        # Generate attention map
        attention = self.sigmoid(self.conv(pooled))  # (batch, 1, height, width)
        
        return x * attention


class ResidualBlock(nn.Module):
    """
    Residual Block with Channel and Spatial Attention.
    
    This block combines:
    - Convolutional layers for feature extraction
    - Batch normalization for stable training
    - Attention mechanisms for feature refinement
    - Residual connection for improved gradient flow
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Initialize residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for the first convolution
        """
        super(ResidualBlock, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Attention modules
        self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention()
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.LeakyReLU(0.01)#nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with residual connection applied
        """
        identity = x
        
        # Main path
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply attention
        out = self.channel_attention(out)
        out = self.spatial_attention(out)
        
        # Add shortcut
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out


class MultiScaleFeatureExtractor(nn.Module):
    """
    Multi-scale feature extraction module.
    
    This module extracts features at different scales using parallel
    convolutional paths with different kernel sizes, capturing both
    fine-grained and coarse patterns in the NMR data.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize multi-scale feature extractor.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (per scale)
        """
        super(MultiScaleFeatureExtractor, self).__init__()
        
        # Different kernel sizes for different scales
        self.scale1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.scale3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.scale5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        
        # Batch normalization for each scale
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.bn5 = nn.BatchNorm2d(out_channels)
        
        # Fusion layer to combine multi-scale features
        self.fusion = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
        self.bn_fusion = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.LeakyReLU(0.01)#nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract and fuse multi-scale features.
        
        Args:
            x: Input tensor
            
        Returns:
            Fused multi-scale features
        """
        # Extract features at different scales
        s1 = self.relu(self.bn1(self.scale1(x)))
        s3 = self.relu(self.bn3(self.scale3(x)))
        s5 = self.relu(self.bn5(self.scale5(x)))
        
        # Concatenate features from all scales
        multi_scale = torch.cat([s1, s3, s5], dim=1)
        
        # Fuse multi-scale features
        fused = self.relu(self.bn_fusion(self.fusion(multi_scale)))
        
        return fused


class AdvancedNMRNet(nn.Module):
    """
    Advanced Neural Network for NMR MQI Prediction.
    
    Key features:
    1. Multi-scale feature extraction to capture patterns at different scales
    2. Residual blocks with attention for better feature learning
    3. Channel and spatial attention mechanisms
    4. Deep architecture with skip connections
    5. Advanced regularization (dropout, batch norm, weight decay)
    
    This architecture is designed to:
    - Better capture complex relationships in NMR data
    - Handle the unique structure of real/imaginary channels
    - Focus on relevant features through attention
    - Enable training of deeper networks through residual connections
    """
    
    def __init__(self, dropout_rate: float = 0.1):
        """
        Initialize the advanced NMR network.
        
        Args:
            dropout_rate: Dropout probability for regularization
        """
        super(AdvancedNMRNet, self).__init__()
        
        # Initial feature extraction with multi-scale approach
        # Input: (batch, 2, 54, 12)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01)#nn.ReLU()
        )
        
        # Multi-scale feature extraction
        self.multi_scale = MultiScaleFeatureExtractor(32, 32)
        
        # Residual blocks with increasing channels
        # These blocks progressively extract higher-level features
        self.res_block1 = ResidualBlock(32, 64)
        self.res_block2 = ResidualBlock(64, 128)
        self.res_block3 = ResidualBlock(128, 256)
        
        # Additional convolutional layers for deeper feature extraction
        self.conv_deep = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),#nn.ReLU(),
            ChannelAttention(256),
            SpatialAttention()
        )
        
        # Global pooling to aggregate spatial information
        # This reduces (batch, 256, 54, 12) to (batch, 256, 1, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Feature fusion and regression head
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 2, 256),  # 2x because we concat avg and max pooling
            #nn.ReLU(),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            #nn.ReLU(),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate / 2),  # Lower dropout near output
            nn.Linear(64, 1)  # Final regression output
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """
        Initialize network weights using appropriate strategies.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for Conv layers (good for ReLU)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the advanced network.
        
        Args:
            x: Input tensor of shape (batch, 2, 54, 12)
            
        Returns:
            Predicted MQI values of shape (batch, 1)
        """
        # Initial feature extraction
        x = self.initial_conv(x)  # (batch, 32, 54, 12)
        
        # Multi-scale features
        x = self.multi_scale(x)  # (batch, 32, 54, 12)
        
        # Residual blocks with attention
        x = self.res_block1(x)  # (batch, 64, 54, 12)
        x = self.res_block2(x)  # (batch, 128, 54, 12)
        x = self.res_block3(x)  # (batch, 256, 54, 12)
        
        # Deep convolutional features with attention
        x = self.conv_deep(x)  # (batch, 256, 54, 12)
        
        # Global pooling (both avg and max for richer representation)
        avg_pool = self.global_avg_pool(x).view(x.size(0), -1)  # (batch, 256)
        max_pool = self.global_max_pool(x).view(x.size(0), -1)  # (batch, 256)
        
        # Concatenate pooled features
        x = torch.cat([avg_pool, max_pool], dim=1)  # (batch, 512)
        
        # Regression head
        x = self.fc_layers(x)  # (batch, 1)
        
        return x


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Stops training when validation loss hasn't improved for a specified
    number of epochs (patience).
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better), 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


def train_advanced_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    use_early_stopping: bool = True
) -> dict:
    """
    Train the advanced model with sophisticated training techniques.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Maximum number of training epochs
        learning_rate: Initial learning rate
        device: Device to use for training
        use_early_stopping: Whether to use early stopping
        
    Returns:
        Dictionary containing training history
    """
    model = model.to(device)
    
    # Huber loss is more robust to outliers than MSE
    criterion = nn.SmoothL1Loss()  # Also known as Huber loss
    
    # AdamW optimizer with weight decay (better regularization than Adam)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Cosine annealing learning rate scheduler with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=15, min_delta=1e-4)
    
    # Track training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        # Track learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        # Update learning rate
        scheduler.step()
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, '
                  f'LR: {current_lr:.6f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(RESULTS_PATH, 'best_advanced_model.pth'))
            print(f'  → New best model saved (val_loss: {best_val_loss:.4f})')
        
        # Early stopping check
        if use_early_stopping and early_stopping(avg_val_loss):
            print(f'\nEarly stopping triggered at epoch {epoch+1}')
            print(f'Best validation loss: {best_val_loss:.4f}')
            break
    
    if not early_stopping.early_stop:
        print(f'\nTraining completed. Best validation loss: {best_val_loss:.4f}')
    
    return history


def evaluate_advanced_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Comprehensive evaluation of the advanced model.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        device: Device to use for evaluation
        
    Returns:
        Tuple of (MSE, MAE, R2, predictions, targets)
    """
    model = model.to(device)
    model.eval()
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            predictions.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
    
    # Concatenate all batches
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
    # R² score (coefficient of determination)
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f'\nAdvanced Model Test Results:')
    print(f'MSE: {mse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'RMSE: {np.sqrt(mse):.4f}')
    print(f'R² Score: {r2:.4f}')
    
    return mse, mae, r2, predictions, targets


# Example usage
if __name__ == "__main__":
    from baseline_model import NMRDataset
    from torch.utils.data import random_split
    from utils import set_seed

    # Set all random seeds for reproducibility
    set_seed(42)

    # Generate synthetic data (replace with actual data)
    
    num_samples = 4200
    echo_samples = np.random.randn(num_samples, 2, 54, 12).astype(np.float32)
    mqi_values = np.random.rand(num_samples).astype(np.float32)
    
    # Create dataset
    dataset = NMRDataset(echo_samples, mqi_values)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset split:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Initialize advanced model
    model = AdvancedNMRNet(dropout_rate=0.3)
    
    # Print model architecture
    print(f"\nAdvanced Model Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\nStarting training with advanced techniques...")
    history = train_advanced_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        learning_rate=0.001,
        use_early_stopping=True
    )
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(RESULTS_PATH, 'best_advanced_model.pth')))
    
    # Evaluate on test set
    mse, mae, r2, predictions, targets = evaluate_advanced_model(model, test_loader)
