"""
Baseline Models for NMR Measurement Quality Index (MQI) Prediction

This module implements multiple baseline approaches:
1. XGBoost (Gradient Boosting) - Classical ML
2. Gaussian Process Regression - Bayesian ML
3. Minimal CNN - Lightweight Deep Learning
4. Original CNN - Previous baseline for comparison

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

# Try to import optional dependencies
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not installed. Install with: pip install xgboost")

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False
    warnings.warn("Scikit-learn GP not available. Install with: pip install scikit-learn")


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
# MODEL 1: XGBOOST (GRADIENT BOOSTING)
# ============================================================================

class XGBoostModel:
    """
    XGBoost model for MQI prediction.

    Gradient boosting ensemble - excellent for small datasets.
    Works with flattened or engineered features.
    """

    def __init__(
        self,
        max_depth: int = 6,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        use_feature_engineering: bool = True,
        feature_method: str = 'statistical'
    ):
        """
        Initialize XGBoost model.

        Args:
            max_depth: Maximum tree depth
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            subsample: Row subsample ratio
            colsample_bytree: Column subsample ratio
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            use_feature_engineering: Use statistical features vs raw flattening
            feature_method: 'statistical' or 'raw'
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")

        self.model = xgb.XGBRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            n_jobs=-1
        )

        self.use_feature_engineering = use_feature_engineering
        self.feature_method = feature_method
        self.feature_extractor = FeatureExtractor(use_pca=False)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            verbose: bool = True):
        """
        Train XGBoost model.

        Args:
            X_train: Training data (N, 2, 54, 12)
            y_train: Training targets (N,)
            X_val: Validation data (optional)
            y_val: Validation targets (optional)
            verbose: Print progress
        """
        # Extract features
        if self.use_feature_engineering:
            X_train_feat = self.feature_extractor.fit_transform(X_train, method=self.feature_method)
            eval_set = None
            if X_val is not None and y_val is not None:
                X_val_feat = self.feature_extractor.transform(X_val, method=self.feature_method)
                eval_set = [(X_val_feat, y_val)]
        else:
            X_train_feat = X_train.reshape(len(X_train), -1)
            eval_set = None
            if X_val is not None and y_val is not None:
                X_val_feat = X_val.reshape(len(X_val), -1)
                eval_set = [(X_val_feat, y_val)]

        # Train
        self.model.fit(
            X_train_feat, y_train,
            eval_set=eval_set,
            verbose=verbose
        )

        if verbose:
            print(f"\n✓ XGBoost training complete")
            print(f"  Features used: {X_train_feat.shape[1]}")
            print(f"  Trees built: {self.model.n_estimators}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input data (N, 2, 54, 12)

        Returns:
            Predictions (N,)
        """
        if self.use_feature_engineering:
            X_feat = self.feature_extractor.transform(X, method=self.feature_method)
        else:
            X_feat = X.reshape(len(X), -1)

        return self.model.predict(X_feat)

    def get_feature_importance(self, top_n: int = 20) -> Dict:
        """
        Get feature importance scores.

        Args:
            top_n: Number of top features to return

        Returns:
            Dictionary with feature indices and importance scores
        """
        importance = self.model.feature_importances_
        top_indices = np.argsort(importance)[-top_n:][::-1]

        return {
            'indices': top_indices,
            'scores': importance[top_indices]
        }


# ============================================================================
# MODEL 2: GAUSSIAN PROCESS REGRESSION
# ============================================================================

class GaussianProcessModel:
    """
    Gaussian Process Regression for MQI prediction.

    Bayesian model that provides uncertainty estimates.
    Best with PCA-reduced features for computational efficiency.
    """

    def __init__(
        self,
        use_pca: bool = True,
        n_pca_components: int = 50,
        use_feature_engineering: bool = True,
        feature_method: str = 'statistical',
        kernel_length_scale: float = 1.0,
        noise_level: float = 0.1
    ):
        """
        Initialize Gaussian Process model.

        Args:
            use_pca: Apply PCA dimensionality reduction
            n_pca_components: Number of PCA components
            use_feature_engineering: Use statistical features
            feature_method: 'statistical' or 'raw'
            kernel_length_scale: RBF kernel length scale
            noise_level: Noise level (alpha)
        """
        if not GP_AVAILABLE:
            raise ImportError("Scikit-learn GP not available. Install with: pip install scikit-learn")

        # Kernel: RBF + noise
        kernel = ConstantKernel(1.0) * RBF(length_scale=kernel_length_scale) + \
                 WhiteKernel(noise_level=noise_level)

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            random_state=42,
            normalize_y=True
        )

        self.use_pca = use_pca
        self.use_feature_engineering = use_feature_engineering
        self.feature_method = feature_method
        self.feature_extractor = FeatureExtractor(
            use_pca=use_pca,
            n_pca_components=n_pca_components
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = True):
        """
        Train Gaussian Process model.

        Args:
            X_train: Training data (N, 2, 54, 12)
            y_train: Training targets (N,)
            verbose: Print progress
        """
        # Extract features
        if self.use_feature_engineering:
            X_train_feat = self.feature_extractor.fit_transform(X_train, method=self.feature_method)
        else:
            X_train_feat = X_train.reshape(len(X_train), -1)
            if self.use_pca:
                self.feature_extractor.pca = PCA(n_components=min(50, X_train_feat.shape[1]))
                X_train_feat = self.feature_extractor.pca.fit_transform(X_train_feat)

        if verbose:
            print(f"Training GP with {X_train_feat.shape[1]} features...")
            print("This may take a few minutes...")

        # Train
        self.model.fit(X_train_feat, y_train)

        if verbose:
            print(f"\n✓ GP training complete")
            print(f"  Kernel: {self.model.kernel_}")
            print(f"  Log-marginal-likelihood: {self.model.log_marginal_likelihood():.4f}")

    def predict(self, X: np.ndarray, return_std: bool = False):
        """
        Make predictions with optional uncertainty estimates.

        Args:
            X: Input data (N, 2, 54, 12)
            return_std: Return standard deviation (uncertainty)

        Returns:
            Predictions (N,) or (predictions, std) if return_std=True
        """
        if self.use_feature_engineering:
            X_feat = self.feature_extractor.transform(X, method=self.feature_method)
        else:
            X_feat = X.reshape(len(X), -1)
            if self.use_pca and self.feature_extractor.pca is not None:
                X_feat = self.feature_extractor.pca.transform(X_feat)

        return self.model.predict(X_feat, return_std=return_std)


# ============================================================================
# MODEL 3: MINIMAL CNN (LIGHTWEIGHT DEEP LEARNING)
# ============================================================================

class MinimalCNN(nn.Module):
    """
    Minimal CNN model - 10x smaller than baseline.

    Only ~5-10K parameters (vs 250K baseline, 1.2M advanced).
    Better parameter-to-sample ratio for small datasets.
    """

    def __init__(self, dropout_rate: float = 0.5):
        """
        Initialize minimal CNN.

        Args:
            dropout_rate: Dropout probability (high for regularization)
        """
        super(MinimalCNN, self).__init__()

        # Very simple convolutional layers
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Minimal fully connected layers
        self.fc1 = nn.Linear(32, 16)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(16, 1)

        self.activation = nn.LeakyReLU(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Conv layers
        x = self.activation(self.bn1(self.conv1(x)))  # (batch, 16, 54, 12)
        x = self.activation(self.bn2(self.conv2(x)))  # (batch, 32, 54, 12)

        # Global pooling
        x = self.global_pool(x)  # (batch, 32, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 32)

        # FC layers
        x = self.activation(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)

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

    This baseline uses standard CNN operations without advanced techniques,
    making it a good starting point for comparison.
    """

    def __init__(self, dropout_rate: float = 0.3):
        """
        Initialize the baseline CNN model.

        Args:
            dropout_rate: Dropout probability for regularization (default: 0.3)
        """
        super(BaselineCNN, self).__init__()

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
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Output layer - single value (MQI)
        self.fc3 = nn.Linear(32, 1)

        # Activation function
        self.relu = nn.LeakyReLU(0.01)  # Allows small negative gradients

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, 2, 54, 12)

        Returns:
            Predicted MQI values of shape (batch, 1)
        """
        # Convolutional feature extraction
        x = self.relu(self.bn1(self.conv1(x)))  # (batch, 32, 54, 12)
        x = self.relu(self.bn2(self.conv2(x)))  # (batch, 64, 54, 12)
        x = self.relu(self.bn3(self.conv3(x)))  # (batch, 128, 54, 12)

        # Global average pooling
        x = self.global_pool(x)  # (batch, 128, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (batch, 128)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)

        x = self.relu(self.fc2(x))
        x = self.dropout2(x)

        # Output (no activation for regression)
        x = self.fc3(x)

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

def train_xgboost_model(
    model: XGBoostModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None
) -> Dict:
    """Train XGBoost model."""
    print("\n" + "="*80)
    print("TRAINING XGBOOST MODEL")
    print("="*80)

    model.fit(X_train, y_train, X_val, y_val, verbose=True)

    # Get feature importance
    importance = model.get_feature_importance(top_n=10)
    print(f"\nTop 10 important features:")
    for idx, (feat_idx, score) in enumerate(zip(importance['indices'], importance['scores'])):
        print(f"  {idx+1}. Feature {feat_idx}: {score:.4f}")

    return {'model': model}


def train_gp_model(
    model: GaussianProcessModel,
    X_train: np.ndarray,
    y_train: np.ndarray
) -> Dict:
    """Train Gaussian Process model."""
    print("\n" + "="*80)
    print("TRAINING GAUSSIAN PROCESS MODEL")
    print("="*80)

    model.fit(X_train, y_train, verbose=True)

    return {'model': model}


def train_minimal_cnn(
    model: MinimalCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> dict:
    """
    Train the minimal CNN model.

    Args:
        model: MinimalCNN model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to use for training

    Returns:
        Dictionary containing training history
    """
    print("\n" + "="*80)
    print("TRAINING MINIMAL CNN MODEL")
    print("="*80)

    model = model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

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
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> dict:
    """
    Train the baseline CNN model with mean squared error loss.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to use for training ('cuda' or 'cpu')

    Returns:
        Dictionary containing training history (losses)
    """
    model = model.to(device)

    # Loss function for regression
    criterion = nn.MSELoss()

    # Adam optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

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

def evaluate_classical_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model"
) -> Tuple[float, float, float, np.ndarray]:
    """
    Evaluate XGBoost or GP model.

    Returns:
        Tuple of (MSE, MAE, R2, predictions)
    """
    print(f"\n{'='*80}")
    print(f"{model_name.upper()} TEST RESULTS")
    print(f"{'='*80}")

    # Get predictions
    if isinstance(model, GaussianProcessModel):
        predictions, std = model.predict(X_test, return_std=True)
        print(f"Mean prediction uncertainty (std): {np.mean(std):.4f}")
    else:
        predictions = model.predict(X_test)

    # Calculate metrics
    mse = np.mean((predictions - y_test) ** 2)
    mae = np.mean(np.abs(predictions - y_test))

    # R² score
    ss_res = np.sum((y_test - predictions) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print(f'MSE:  {mse:.4f}')
    print(f'MAE:  {mae:.4f}')
    print(f'RMSE: {np.sqrt(mse):.4f}')
    print(f'R²:   {r2:.4f}')
    print(f"{'='*80}\n")

    return mse, mae, r2, predictions


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate the model on test data.

    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        device: Device to use for evaluation

    Returns:
        Tuple of (MSE, MAE, predictions, targets)
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

    print(f'\nTest Results:')
    print(f'MSE: {mse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'RMSE: {np.sqrt(mse):.4f}')

    return mse, mae, predictions, targets


def run_all():
    from src.utils import set_seed
    import time

    SEED = 42
    set_seed(SEED)

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

    # ========== MODEL 1: XGBOOST ==========
    if XGBOOST_AVAILABLE:
        xgb_start = time.time()
        xgb_model = XGBoostModel(
            max_depth=6,
            n_estimators=300,
            learning_rate=0.05,
            use_feature_engineering=True,
            feature_method='statistical'
        )
        train_xgboost_model(xgb_model, X_train, y_train, X_val, y_val)
        xgb_mse, xgb_mae, xgb_r2, xgb_preds = evaluate_classical_model(
            xgb_model, X_test, y_test, "XGBoost"
        )
        xgb_time = time.time() - xgb_start

    # ========== MODEL 2: GAUSSIAN PROCESS ==========
    if GP_AVAILABLE:
        gp_start = time.time()
        gp_model = GaussianProcessModel(
            use_pca=True,
            n_pca_components=50,
            use_feature_engineering=True,
            feature_method='statistical'
        )
        train_gp_model(gp_model, X_train, y_train)
        gp_mse, gp_mae, gp_r2, gp_preds = evaluate_classical_model(
            gp_model, X_test, y_test, "Gaussian Process"
        )
        gp_time = time.time() - gp_start

    # ========== MODEL 3: MINIMAL CNN ==========
    # Create PyTorch datasets
    train_dataset = NMRDataset(X_train, y_train)
    val_dataset = NMRDataset(X_val, y_val)
    test_dataset = NMRDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    minimal_cnn = MinimalCNN(dropout_rate=0.5)
    minimal_cnn_params = sum(p.numel() for p in minimal_cnn.parameters())
    print(f"\nMinimal CNN parameters: {minimal_cnn_params:,}")

    min_start = time.time()
    train_minimal_cnn(minimal_cnn, train_loader, val_loader, num_epochs=50)
    minimal_cnn.load_state_dict(torch.load(os.path.join(RESULTS_PATH, 'best_minimal_cnn.pth')))
    min_mse, min_mae, min_preds, min_targets = evaluate_model(minimal_cnn, test_loader)
    min_time = time.time() - min_start

    # ========== MODEL 4: ORIGINAL BASELINE CNN ==========
    baseline_cnn = BaselineCNN(dropout_rate=0.3)
    print(f"\nBaseline CNN parameters: {sum(p.numel() for p in baseline_cnn.parameters()):,}")

    train_model(baseline_cnn, train_loader, val_loader, num_epochs=50)
    baseline_cnn.load_state_dict(torch.load(os.path.join(RESULTS_PATH, 'best_baseline_model.pth')))
    base_mse, base_mae, base_preds, base_targets = evaluate_model(baseline_cnn, test_loader)

    # ========== SUMMARY ==========
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(f"{'Model':<20} {'MSE':<12} {'MAE':<12} {'RMSE':<12}")
    print("-"*80)
    if XGBOOST_AVAILABLE:
        print(f"{'XGBoost':<20} {xgb_mse:<12.4f} {xgb_mae:<12.4f} {np.sqrt(xgb_mse):<12.4f}")
    if GP_AVAILABLE:
        print(f"{'Gaussian Process':<20} {gp_mse:<12.4f} {gp_mae:<12.4f} {np.sqrt(gp_mse):<12.4f}")
    print(f"{'Minimal CNN':<20} {min_mse:<12.4f} {min_mae:<12.4f} {np.sqrt(min_mse):<12.4f}")
    print(f"{'Baseline CNN':<20} {base_mse:<12.4f} {base_mae:<12.4f} {np.sqrt(base_mse):<12.4f}")
    print("="*80)

    # ========== GENERATE BASELINE MODELS COMPARISON ==========
    if XGBOOST_AVAILABLE and GP_AVAILABLE:

        print("\n" + "="*80)
        print("GENERATING BASELINE MODELS COMPARISON")
        print("="*80)

        # Prepare metrics dictionaries for the three baseline models
        xgboost_metrics = {
            'mse': xgb_mse,
            'mae': xgb_mae,
            'rmse': np.sqrt(xgb_mse),
            'r2': xgb_r2,
            'training_time': xgb_time,
            'n_features': xgb_model.feature_extractor.extract_statistical_features(X_train[:1]).shape[1]
        }

        gp_metrics = {
            'mse': gp_mse,
            'mae': gp_mae,
            'rmse': np.sqrt(gp_mse),
            'r2': gp_r2,
            'training_time': gp_time,
            'n_features': gp_model.feature_extractor.extract_statistical_features(X_train[:1]).shape[1]
        }

        minimal_cnn_metrics = {
            'mse': min_mse,
            'mae': min_mae,
            'rmse': np.sqrt(min_mse),
            'training_time': min_time,
            'params': minimal_cnn_params
        }

        # Calculate R² for Minimal CNN if not already calculated
        ss_res = np.sum((min_targets.flatten() - min_preds.flatten()) ** 2)
        ss_tot = np.sum((min_targets.flatten() - np.mean(min_targets.flatten())) ** 2)
        min_r2 = 1 - (ss_res / ss_tot)
        minimal_cnn_metrics['r2'] = min_r2

        return xgb_preds, gp_preds, min_preds, y_test, \
               xgboost_metrics, gp_metrics, minimal_cnn_metrics
                      
    else:
        if not XGBOOST_AVAILABLE:
            print("\n⚠️  XGBoost not available - skipping baseline models comparison")
        if not GP_AVAILABLE:
            print("\n⚠️  Gaussian Process not available - skipping baseline models comparison")
    
    