"""
Baseline Models for NMR Measurement Quality Index (MQI) Prediction

This module implements multiple baseline approaches:
1. Minimal CNN - Lightweight Deep Learning
2. Original CNN - Previous baseline for comparison

Includes feature engineering utilities for classical ML models.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from typing import Tuple, Optional, Dict, List
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import src.utils as ut

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "results")


# ============================================================================
# FEATURE ENGINEERING UTILITIES
# ============================================================================

class FeatureExtractor:
    """
    Extract engineered features from NMR echo samples.

    Converts raw (2, 54, 12) data to interpretable statistical features
    that work better with classical ML models like XGBoost and GP.
    """

    def __init__(self, use_pca: bool = False, n_pca_components: int = 50):
        """
        Initialize feature extractor.

        Args:
            use_pca: Whether to apply PCA for dimensionality reduction
            n_pca_components: Number of PCA components to keep
        """
        self.use_pca = use_pca
        self.n_pca_components = n_pca_components
        self.pca = None
        self.scaler = None

    def extract_statistical_features(self, echo_samples: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from NMR echo samples.

        For each bin (12 total) and each channel (real/imaginary):
        - Mean, std, min, max
        - Peak value and location
        - Signal energy
        - Kurtosis (peakedness)

        Args:
            echo_samples: Array of shape (N, 2, 54, 12)

        Returns:
            Features of shape (N, n_features)
        """
        N = echo_samples.shape[0]
        features = []

        for i in range(N):
            sample_features = []

            # Process each channel (real, imaginary)
            for ch in range(2):
                channel_data = echo_samples[i, ch, :, :]  # Shape: (54, 12)

                # Process each bin
                for bin_idx in range(12):
                    bin_data = channel_data[:, bin_idx]  # Shape: (54,)

                    # Basic statistics
                    sample_features.append(np.mean(bin_data))
                    sample_features.append(np.std(bin_data))
                    sample_features.append(np.min(bin_data))
                    sample_features.append(np.max(bin_data))

                    # Peak information
                    sample_features.append(np.max(np.abs(bin_data)))
                    sample_features.append(np.argmax(np.abs(bin_data)) / 54.0)  # Normalized position

                    # Signal energy
                    sample_features.append(np.sum(bin_data ** 2))

                    # Kurtosis (peakedness)
                    if np.std(bin_data) > 1e-8:
                        kurtosis = np.mean((bin_data - np.mean(bin_data)) ** 4) / (np.std(bin_data) ** 4)
                    else:
                        kurtosis = 0.0
                    sample_features.append(kurtosis)

                # Global channel statistics
                sample_features.append(np.mean(channel_data))
                sample_features.append(np.std(channel_data))
                sample_features.append(np.sum(channel_data ** 2))  # Total energy

            # Cross-channel features
            real_channel = echo_samples[i, 0, :, :]
            imag_channel = echo_samples[i, 1, :, :]

            # Correlation between real and imaginary
            corr = np.corrcoef(real_channel.flatten(), imag_channel.flatten())[0, 1]
            sample_features.append(corr if not np.isnan(corr) else 0.0)

            # Magnitude statistics
            magnitude = np.sqrt(real_channel**2 + imag_channel**2)
            sample_features.append(np.mean(magnitude))
            sample_features.append(np.std(magnitude))
            sample_features.append(np.max(magnitude))

            features.append(sample_features)

        features = np.array(features)
        print(f"Extracted {features.shape[1]} statistical features")
        return features

    def extract_raw_features(self, echo_samples: np.ndarray) -> np.ndarray:
        """
        Simple flattening of raw echo samples.

        Args:
            echo_samples: Array of shape (N, 2, 54, 12)

        Returns:
            Flattened features of shape (N, 1296)
        """
        N = echo_samples.shape[0]
        features = echo_samples.reshape(N, -1)
        print(f"Flattened to {features.shape[1]} raw features")
        return features

    def fit_transform(self, echo_samples: np.ndarray, method: str = 'statistical') -> np.ndarray:
        """
        Extract features and optionally apply PCA.

        Args:
            echo_samples: Array of shape (N, 2, 54, 12)
            method: 'statistical' or 'raw'

        Returns:
            Transformed features
        """
        # Extract features
        if method == 'statistical':
            features = self.extract_statistical_features(echo_samples)
        elif method == 'raw':
            features = self.extract_raw_features(echo_samples)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Apply PCA if requested
        if self.use_pca:
            self.pca = PCA(n_components=min(self.n_pca_components, features.shape[1]))
            features = self.pca.fit_transform(features)
            variance_explained = np.sum(self.pca.explained_variance_ratio_)
            print(f"PCA: {features.shape[1]} components explain {variance_explained*100:.2f}% variance")

        return features

    def transform(self, echo_samples: np.ndarray, method: str = 'statistical') -> np.ndarray:
        """
        Transform features using fitted parameters (for validation/test sets).

        Args:
            echo_samples: Array of shape (N, 2, 54, 12)
            method: 'statistical' or 'raw'

        Returns:
            Transformed features
        """
        # Extract features
        if method == 'statistical':
            features = self.extract_statistical_features(echo_samples)
        elif method == 'raw':
            features = self.extract_raw_features(echo_samples)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Apply PCA if it was fitted
        if self.use_pca and self.pca is not None:
            features = self.pca.transform(features)

        return features


# ============================================================================
# MODEL 1: MINIMAL CNN (LIGHTWEIGHT DEEP LEARNING)
# ============================================================================

class MinimalCNN(nn.Module):
    """
    Minimal CNN model - 10x smaller than baseline.

    Only ~5-10K parameters (vs 250K baseline, 1.2M advanced).
    Better parameter-to-sample ratio for small datasets.

    IMPROVED VERSION with fixes for range compression issue:
    - Sigmoid output activation to constrain predictions to [0, 1]
    - ELU activation instead of LeakyReLU (better for regression)
    - Lower dropout (0.3 instead of 0.5)
    - Better initialization
    """

    def __init__(self, dropout_rate: float = 0.3, use_output_activation: bool = True):
        """
        Initialize minimal CNN.

        Args:
            dropout_rate: Dropout probability (default: 0.3, lower than before)
            use_output_activation: Whether to use sigmoid on output (default: True)
        """
        super(MinimalCNN, self).__init__()

        self.use_output_activation = use_output_activation

        # Very simple convolutional layers
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Minimal fully connected layers
        self.fc1 = nn.Linear(32, 16)
        self.bn_fc1 = nn.BatchNorm1d(16)  # Added: BN for FC layer
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(16, 1)

        # ELU activation - better for regression than LeakyReLU
        # Smooth everywhere, allows negative values, faster convergence
        self.activation = nn.ELU()

        # Initialize weights properly for regression
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for better regression performance."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Conv layers
        x = self.activation(self.bn1(self.conv1(x)))  # (batch, 16, 54, 12)
        x = self.activation(self.bn2(self.conv2(x)))  # (batch, 32, 54, 12)

        # Global pooling
        x = self.global_pool(x)  # (batch, 32, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 32)

        # FC layers with batch normalization
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)

        # BEST FIX: No activation function, just clipping
        # Let the network learn the full range naturally
        # Batch normalization prevents extreme values during training
        x = torch.clamp(x, 0, 1)

        return x


# ============================================================================
# MODEL 4: ORIGINAL BASELINE CNN (FOR COMPARISON)
# ============================================================================

class BaselineCNN(nn.Module):
    """
    Original baseline CNN model for MQI prediction.

    Architecture:
    - Simple 2D convolutional layers to extract spatial features
    - Global average pooling to reduce dimensions
    - Fully connected layers for regression

    IMPROVED VERSION with fixes for range compression issue:
    - Sigmoid output activation to constrain predictions to [0, 1]
    - ELU activation instead of LeakyReLU (better for regression)
    - Better initialization
    """

    def __init__(self, dropout_rate: float = 0.3, use_output_activation: bool = True):
        """
        Initialize the baseline CNN model.

        Args:
            dropout_rate: Dropout probability for regularization (default: 0.3)
            use_output_activation: Whether to use sigmoid on output (default: True)
        """
        super(BaselineCNN, self).__init__()

        self.use_output_activation = use_output_activation

        # Convolutional layers to extract features from the 2D structure (54 x 12)
        # Input: (batch, 2, 54, 12)
        self.conv1 = nn.Conv2d(
            in_channels=2,      # Real and imaginary channels
            out_channels=32,    # First feature extraction layer
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)

        # Second convolutional layer
        # Shape: (batch, 32, 54, 12)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)

        # Third convolutional layer
        # Shape: (batch, 64, 54, 12)
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)

        # Global average pooling to reduce spatial dimensions
        # This averages across the spatial dimensions (54 x 12)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(128, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)  # Added: BN for FC layer
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(64, 32)
        self.bn_fc2 = nn.BatchNorm1d(32)  # Added: BN for FC layer
        self.dropout2 = nn.Dropout(dropout_rate)

        # Output layer - single value (MQI)
        self.fc3 = nn.Linear(32, 1)

        # ELU activation - better for regression than LeakyReLU
        self.activation = nn.ELU()

        # Initialize weights properly for regression
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for better regression performance."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, 2, 54, 12)

        Returns:
            Predicted MQI values of shape (batch, 1)
        """
        # Convolutional feature extraction
        x = self.activation(self.bn1(self.conv1(x)))  # (batch, 32, 54, 12)
        x = self.activation(self.bn2(self.conv2(x)))  # (batch, 64, 54, 12)
        x = self.activation(self.bn3(self.conv3(x)))  # (batch, 128, 54, 12)

        # Global average pooling
        x = self.global_pool(x)  # (batch, 128, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (batch, 128)

        # Fully connected layers with batch normalization
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.activation(x)
        x = self.dropout2(x)

        # Output layer
        x = self.fc3(x)

        # BEST FIX: No activation function, just clipping
        # Let the network learn the full range naturally
        # Batch normalization prevents extreme values during training
        x = torch.clamp(x, 0, 1)

        return x


# ============================================================================
# PYTORCH DATASET (UNCHANGED)
# ============================================================================

class NMRDataset(Dataset):
    """
    PyTorch Dataset for NMR echo samples and their corresponding MQI values.

    The NMR data has shape (2, 54, 12) where:
    - 2 channels: real and imaginary components
    - 54 points per bin
    - 12 bins
    """

    def __init__(self, echo_samples: np.ndarray, mqi_values: np.ndarray):
        """
        Initialize the dataset.

        Args:
            echo_samples: Array of shape (N, 2, 54, 12) containing NMR echo samples
            mqi_values: Array of shape (N,) containing MQI target values
        """
        self.echo_samples = torch.FloatTensor(echo_samples)
        self.mqi_values = torch.FloatTensor(mqi_values).unsqueeze(1)  # Shape: (N, 1)

    def __len__(self) -> int:
        return len(self.echo_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.echo_samples[idx], self.mqi_values[idx]


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_minimal_cnn(
    model: MinimalCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.0005,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    use_huber_loss: bool = True
) -> dict:
    """
    Train the minimal CNN model.

    Args:
        model: MinimalCNN model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer (default: 0.0005, lower than before)
        device: Device to use for training
        use_huber_loss: Use Huber loss instead of MSE (default: True)

    Returns:
        Dictionary containing training history
    """
    print("\n" + "="*80)
    print("TRAINING MINIMAL CNN MODEL")
    print("="*80)

    model = model.to(device)

    # Loss and optimizer
    # Huber loss (SmoothL1Loss) is less sensitive to outliers than MSE
    # This prevents the model from being too conservative
    if use_huber_loss:
        criterion = nn.SmoothL1Loss()  # Huber loss
        print("Using Huber Loss (SmoothL1Loss) - robust to outliers")
    else:
        criterion = nn.MSELoss()
        print("Using MSE Loss")

    # Lower learning rate + higher weight decay for bounded regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training history
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(RESULTS_PATH, 'best_minimal_cnn.pth'))

    print(f'\n✓ Training completed. Best validation loss: {best_val_loss:.4f}')

    return history


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.0005,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    use_huber_loss: bool = True
) -> dict:
    """
    Train the baseline CNN model with improved loss function.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer (default: 0.0005, lower than before)
        device: Device to use for training ('cuda' or 'cpu')
        use_huber_loss: Use Huber loss instead of MSE (default: True)

    Returns:
        Dictionary containing training history (losses)
    """
    model = model.to(device)

    # Loss function for regression
    # Huber loss is less sensitive to outliers and prevents extreme predictions
    if use_huber_loss:
        criterion = nn.SmoothL1Loss()  # Huber loss
        print("Using Huber Loss (SmoothL1Loss) - robust to outliers")
    else:
        criterion = nn.MSELoss()
        print("Using MSE Loss")

    # Adam optimizer with higher weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Learning rate scheduler to reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5#, verbose=True
    )

    # Track training history
    history = {
        'train_loss': [],
        'val_loss': []
    }

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Calculate loss
            loss = criterion(output, target)

            # Backward pass
            loss.backward()

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

        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)

        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(RESULTS_PATH, 'best_baseline_model.pth'))

    print(f'\nTraining completed. Best validation loss: {best_val_loss:.4f}')

    return history


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    model_name: str = "CNN Model",
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Evaluate the model on test data.

    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        model_name: Name of the model for display
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
    predictions = np.concatenate(predictions, axis=0).flatten()
    targets = np.concatenate(targets, axis=0).flatten()

    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))

    # R² score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print(f"\n{'='*80}")
    print(f"{model_name.upper()} TEST RESULTS")
    print(f"{'='*80}")
    print(f'MSE:  {mse:.4f}')
    print(f'MAE:  {mae:.4f}')
    print(f'RMSE: {np.sqrt(mse):.4f}')
    print(f'R²:   {r2:.4f}')
    print(f"{'='*80}\n")

    return mse, mae, r2, predictions, targets


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_prediction_analysis(
    predictions: np.ndarray,
    targets: np.ndarray,
    model_name: str = "CNN Model",
    save_path: str = None
):
    """
    Create comprehensive prediction analysis plots for a single CNN model.

    Args:
        predictions: Model predictions
        targets: Ground truth values
        model_name: Name of the model for plot titles
        save_path: Path to save the plot (optional)
    """
    import matplotlib.pyplot as plt

    if save_path is None:
        save_path = os.path.join(RESULTS_PATH, f'{model_name.lower().replace(" ", "_")}_prediction_analysis.png')

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Flatten arrays for consistency
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Calculate errors
    errors = predictions - targets
    abs_errors = np.abs(errors)

    # Color scheme
    color = '#1f77b4'  # Blue

    # ========== Plot 1: Predicted vs Actual ==========
    axes[0, 0].scatter(targets, predictions, alpha=0.6, s=30, color=color, edgecolors='black', linewidth=0.5)
    axes[0, 0].plot([targets.min(), targets.max()],
                    [targets.min(), targets.max()],
                    'r--', linewidth=2.5, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual MQI', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Predicted MQI', fontsize=12, fontweight='bold')
    axes[0, 0].set_title(f'{model_name}: Predicted vs Actual', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    # Add R² annotation
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}',
                    transform=axes[0, 0].transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ========== Plot 2: Residual Plot ==========
    axes[0, 1].scatter(targets, errors, alpha=0.6, s=30, color=color, edgecolors='black', linewidth=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2.5)
    axes[0, 1].set_xlabel('Actual MQI', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Residual (Predicted - Actual)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title(f'{model_name}: Residual Plot', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Add mean and std annotation
    axes[0, 1].text(0.05, 0.95,
                    f'Mean Error: {np.mean(errors):.4f}\nStd Error: {np.std(errors):.4f}',
                    transform=axes[0, 1].transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ========== Plot 3: Error Distribution ==========
    axes[1, 0].hist(errors, bins=50, alpha=0.7, color=color, edgecolor='black', linewidth=1.2)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='Zero Error')
    axes[1, 0].axvline(x=np.mean(errors), color='darkblue', linestyle=':', linewidth=2.5,
                       label=f'Mean: {np.mean(errors):.4f}')
    axes[1, 0].set_xlabel('Prediction Error', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1, 0].set_title(f'{model_name}: Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # ========== Plot 4: Error Statistics ==========
    axes[1, 1].boxplot([abs_errors], labels=[model_name],
                       patch_artist=True,
                       boxprops=dict(facecolor=color, alpha=0.7),
                       medianprops=dict(color='red', linewidth=2),
                       widths=0.5)
    axes[1, 1].set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
    axes[1, 1].set_title(f'{model_name}: Absolute Error Statistics', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # Add statistics text
    stats_text = f'Min: {np.min(abs_errors):.4f}\n'
    stats_text += f'Q1: {np.percentile(abs_errors, 25):.4f}\n'
    stats_text += f'Median: {np.median(abs_errors):.4f}\n'
    stats_text += f'Q3: {np.percentile(abs_errors, 75):.4f}\n'
    stats_text += f'Max: {np.max(abs_errors):.4f}\n'
    stats_text += f'Mean: {np.mean(abs_errors):.4f}'

    axes[1, 1].text(1.3, 0.5, stats_text,
                    transform=axes[1, 1].transData,
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n{model_name} prediction analysis plot saved to: {save_path}")
    plt.close()


def plot_training_history(
    history: dict,
    model_name: str = "CNN Model",
    save_path: str = None
):
    """
    Plot training and validation loss curves.

    Args:
        history: Dictionary containing 'train_loss' and 'val_loss' lists
        model_name: Name of the model for plot title
        save_path: Path to save the plot (optional)
    """
    import matplotlib.pyplot as plt

    if save_path is None:
        save_path = os.path.join(RESULTS_PATH, f'{model_name.lower().replace(" ", "_")}_training_history.png')

    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(history['train_loss']) + 1)

    ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)

    # Mark the best validation loss
    best_val_idx = np.argmin(history['val_loss'])
    best_val_loss = history['val_loss'][best_val_idx]
    ax.plot(best_val_idx + 1, best_val_loss, 'r*', markersize=15,
            label=f'Best Val Loss: {best_val_loss:.4f} (Epoch {best_val_idx + 1})')

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name}: Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"{model_name} training history plot saved to: {save_path}")
    plt.close()


def plot_models_comparison(
    models_results: dict,
    save_path: str = None
):
    """
    Create comprehensive comparison plots for multiple CNN models.

    Args:
        models_results: Dictionary with model names as keys and results dicts as values
                       Each result dict should contain: 'predictions', 'targets', 'metrics'
        save_path: Path to save the plot (optional)
    """
    import matplotlib.pyplot as plt

    if save_path is None:
        save_path = os.path.join(RESULTS_PATH, 'cnn_models_comparison.png')

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    model_names = list(models_results.keys())

    # ========== Plot 1: Predicted vs Actual (All Models) ==========
    for idx, (model_name, results) in enumerate(models_results.items()):
        preds = results['predictions'].flatten()
        targs = results['targets'].flatten()
        color = colors[idx % len(colors)]

        axes[0, 0].scatter(targs, preds, alpha=0.5, s=20, color=color,
                          label=model_name, edgecolors='black', linewidth=0.3)

    # Add perfect prediction line
    all_targets = np.concatenate([r['targets'].flatten() for r in models_results.values()])
    axes[0, 0].plot([all_targets.min(), all_targets.max()],
                    [all_targets.min(), all_targets.max()],
                    'k--', linewidth=2.5, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual MQI', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Predicted MQI', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Predicted vs Actual (All Models)', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # ========== Plot 2: MSE Comparison ==========
    mse_values = [results['metrics']['mse'] for results in models_results.values()]
    bars = axes[0, 1].bar(range(len(model_names)), mse_values,
                          color=colors[:len(model_names)], alpha=0.7, edgecolor='black', linewidth=1.2)
    axes[0, 1].set_xticks(range(len(model_names)))
    axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 1].set_ylabel('MSE', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Mean Squared Error Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, mse_values):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # ========== Plot 3: MAE Comparison ==========
    mae_values = [results['metrics']['mae'] for results in models_results.values()]
    bars = axes[0, 2].bar(range(len(model_names)), mae_values,
                          color=colors[:len(model_names)], alpha=0.7, edgecolor='black', linewidth=1.2)
    axes[0, 2].set_xticks(range(len(model_names)))
    axes[0, 2].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 2].set_ylabel('MAE', fontsize=12, fontweight='bold')
    axes[0, 2].set_title('Mean Absolute Error Comparison', fontsize=14, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, mae_values):
        height = bar.get_height()
        axes[0, 2].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # ========== Plot 4: R² Comparison ==========
    r2_values = [results['metrics']['r2'] for results in models_results.values()]
    bars = axes[1, 0].bar(range(len(model_names)), r2_values,
                          color=colors[:len(model_names)], alpha=0.7, edgecolor='black', linewidth=1.2)
    axes[1, 0].set_xticks(range(len(model_names)))
    axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1, 0].set_ylabel('R² Score', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylim([0, 1.0])
    axes[1, 0].axhline(y=0.8, color='g', linestyle='--', linewidth=2, alpha=0.5, label='Good threshold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].legend(fontsize=9)

    # Add value labels on bars
    for bar, val in zip(bars, r2_values):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # ========== Plot 5: Error Distribution Comparison ==========
    for idx, (model_name, results) in enumerate(models_results.items()):
        preds = results['predictions'].flatten()
        targs = results['targets'].flatten()
        errors = preds - targs
        color = colors[idx % len(colors)]

        axes[1, 1].hist(errors, bins=30, alpha=0.5, color=color,
                       label=model_name, edgecolor='black', linewidth=0.8)

    axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='Zero Error')
    axes[1, 1].set_xlabel('Prediction Error', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Error Distribution Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # ========== Plot 6: Model Parameters & Training Time ==========
    if 'params' in list(models_results.values())[0]['metrics']:
        param_counts = [results['metrics']['params'] for results in models_results.values()]
        train_times = [results['metrics']['training_time'] for results in models_results.values()]

        ax1 = axes[1, 2]
        ax2 = ax1.twinx()

        x = np.arange(len(model_names))
        width = 0.35

        bars1 = ax1.bar(x - width/2, [p/1000 for p in param_counts], width,
                       label='Parameters (K)', color='steelblue', alpha=0.7, edgecolor='black')
        bars2 = ax2.bar(x + width/2, train_times, width,
                       label='Training Time (s)', color='coral', alpha=0.7, edgecolor='black')

        ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Parameters (Thousands)', fontsize=11, fontweight='bold', color='steelblue')
        ax2.set_ylabel('Training Time (seconds)', fontsize=11, fontweight='bold', color='coral')
        ax1.set_title('Model Complexity & Training Time', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax2.tick_params(axis='y', labelcolor='coral')
        ax1.grid(True, alpha=0.3, axis='y')

        # Add legends
        ax1.legend(loc='upper left', fontsize=9)
        ax2.legend(loc='upper right', fontsize=9)
    else:
        # If no params info, just show a summary table
        axes[1, 2].axis('off')
        summary_text = "Model Performance Summary\n\n"
        for model_name, results in models_results.items():
            metrics = results['metrics']
            summary_text += f"{model_name}:\n"
            summary_text += f"  MSE:  {metrics['mse']:.4f}\n"
            summary_text += f"  MAE:  {metrics['mae']:.4f}\n"
            summary_text += f"  R²:   {metrics['r2']:.4f}\n\n"

        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='top', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nCNN models comparison plot saved to: {save_path}")
    plt.close()


def run_all():
    """
    Main function to train and evaluate both CNN models with comprehensive visualizations.

    Returns:
        Dictionary containing results for both models
    """
    from src.utils import set_seed
    import time

    SEED = 42
    set_seed(SEED)

    # Load and split data
    train_data, val_data, test_data, norm_stats = ut.get_data_splits_normalized(
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42
    )

    X_train = train_data[0]
    y_train = train_data[1]
    X_val = val_data[0]
    y_val = val_data[1]
    X_test = test_data[0]
    y_test = test_data[1]

    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")

    # Create PyTorch datasets
    train_dataset = NMRDataset(X_train, y_train)
    val_dataset = NMRDataset(X_val, y_val)
    test_dataset = NMRDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # ========== MODEL 1: MINIMAL CNN ==========
    print("\n" + "="*80)
    print("MODEL 1: MINIMAL CNN")
    print("="*80)

    minimal_cnn = MinimalCNN(dropout_rate=0.5)
    minimal_cnn_params = sum(p.numel() for p in minimal_cnn.parameters())
    print(f"Minimal CNN parameters: {minimal_cnn_params:,}")

    min_start = time.time()
    minimal_history = train_minimal_cnn(minimal_cnn, train_loader, val_loader, num_epochs=50)
    minimal_cnn.load_state_dict(torch.load(os.path.join(RESULTS_PATH, 'best_minimal_cnn.pth')))
    min_mse, min_mae, min_r2, min_preds, min_targets = evaluate_model(
        minimal_cnn, test_loader, model_name="Minimal CNN"
    )
    min_time = time.time() - min_start

    # Generate visualizations for Minimal CNN
    plot_prediction_analysis(min_preds, min_targets, model_name="Minimal CNN")
    plot_training_history(minimal_history, model_name="Minimal CNN")

    # ========== MODEL 2: BASELINE CNN ==========
    print("\n" + "="*80)
    print("MODEL 2: BASELINE CNN")
    print("="*80)

    baseline_cnn = BaselineCNN(dropout_rate=0.3)
    baseline_cnn_params = sum(p.numel() for p in baseline_cnn.parameters())
    print(f"Baseline CNN parameters: {baseline_cnn_params:,}")

    base_start = time.time()
    baseline_history = train_model(baseline_cnn, train_loader, val_loader, num_epochs=50)
    baseline_cnn.load_state_dict(torch.load(os.path.join(RESULTS_PATH, 'best_baseline_model.pth')))
    base_mse, base_mae, base_r2, base_preds, base_targets = evaluate_model(
        baseline_cnn, test_loader, model_name="Baseline CNN"
    )
    base_time = time.time() - base_start

    # Generate visualizations for Baseline CNN
    plot_prediction_analysis(base_preds, base_targets, model_name="Baseline CNN")
    plot_training_history(baseline_history, model_name="Baseline CNN")

    # ========== FINAL SUMMARY ==========
    print("\n" + "="*80)
    print("FINAL COMPARISON - BOTH CNN MODELS")
    print("="*80)
    print(f"{'Model':<20} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'R²':<12} {'Time (s)':<12}")
    print("-"*80)
    print(f"{'Minimal CNN':<20} {min_mse:<12.4f} {min_mae:<12.4f} {np.sqrt(min_mse):<12.4f} {min_r2:<12.4f} {min_time:<12.2f}")
    print(f"{'Baseline CNN':<20} {base_mse:<12.4f} {base_mae:<12.4f} {np.sqrt(base_mse):<12.4f} {base_r2:<12.4f} {base_time:<12.2f}")
    print("="*80)

    # ========== COMPREHENSIVE COMPARISON PLOT ==========
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE COMPARISON PLOTS")
    print("="*80)

    # Prepare results dictionary for comparison plot
    models_results = {
        'Minimal CNN': {
            'predictions': min_preds,
            'targets': min_targets,
            'metrics': {
                'mse': min_mse,
                'mae': min_mae,
                'rmse': np.sqrt(min_mse),
                'r2': min_r2,
                'training_time': min_time,
                'params': minimal_cnn_params
            }
        },
        'Baseline CNN': {
            'predictions': base_preds,
            'targets': base_targets,
            'metrics': {
                'mse': base_mse,
                'mae': base_mae,
                'rmse': np.sqrt(base_mse),
                'r2': base_r2,
                'training_time': base_time,
                'params': baseline_cnn_params
            }
        }
    }

    # Generate comprehensive comparison plot
    plot_models_comparison(models_results)

    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETED")
    print("="*80)

    return models_results
    