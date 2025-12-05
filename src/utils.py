import numpy as np
import os, sys
from scipy import fft, io
import torch
import random

from sklearn.decomposition import PCA

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_DATA_PATH = os.path.join(DATA_PATH, "TrainingData_vol2")
VALID_DATA_PATH = os.path.join(DATA_PATH, "ValidationData_vol2")

DATA_LEN = 42 # number of different speed / displacement combinations
SAMPLE_LEN = 100 # number of different samples within each combination


def set_seed(seed=42):

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # PyTorch CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_raw_data(feature_type, target_type):
    """
    Load raw NMR data from .mat files WITHOUT normalization.
    This should be used when you want to normalize properly after splitting.

    Returns:
        tuple: (echo_samples, mqi_values) both as numpy arrays
            - echo_samples: shape (4200, 2, 54, 12) - NOT normalized
            - mqi_values: shape (4200,)
    """

    condition = True

    mat_contents = {}
    for idx in range(DATA_LEN):

        file_idx = f"0{idx+1}" if idx < 9 else idx+1
        filename = os.path.join(TRAIN_DATA_PATH, f"case0{file_idx}_result.mat")
        mat_contents[idx] = io.loadmat(filename)

    valid_data_samples = [(19, val) for val in range(11)]
    valid_data_samples.extend([(20, val) for val in range(9)])
    valid_data_samples.extend([(22, 0), (24, 0), (28, 0), (32, 0)])
    idx = DATA_LEN
    for case_idx, segment_idx in valid_data_samples:

        seg_file_idx = f"0{segment_idx}" if segment_idx < 10 else f"{segment_idx}"
        filename = os.path.join(VALID_DATA_PATH, f"case0{case_idx}_segment{seg_file_idx}_result.mat")
        mat_contents[idx] = io.loadmat(filename)
        idx += 1

    if feature_type == "echo_sample":
        DATA_IDX = 1
    elif feature_type == "cpmg" or feature_type == "fft_cpmg":
        DATA_IDX = 0
    else:
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    all_features = []
    all_targets = []

    for idx_1 in range(len(mat_contents)):
                
        # use condition
        #v = mat_contents[idx_1]['v'][0][0]
        #if v > 1600: # eliminate very high speeds
        #    continue

        # use condition
        #d = mat_contents[idx_1]['d'][0][0]
        #if d > 30: # eliminate very high displacements
        #    continue


        for idx_2 in range(SAMPLE_LEN):
            mix_sample = mat_contents[idx_1]['mixed'][0][idx_2]
            all_features.append(mix_sample[DATA_IDX])

            if target_type == "mqi":
                all_targets.append(mix_sample[2][0][0])
            elif target_type == "speed":
                all_targets.append(mat_contents[idx_1]['v'][0][0])
            elif target_type == "displacement":
                all_targets.append(mat_contents[idx_1]['d'][0][0])

    del mat_contents  # Free memory

    n = len(all_targets)

    # Convert to numpy and flatten
    if feature_type == "echo_sample":
        input_data = np.array(all_features).reshape(n, 54, 12)
        input_data_real = input_data.real
        input_data_imag = input_data.imag
        input_data_ri = np.array([input_data_real, input_data_imag]).reshape(n, 2, 54, 12)
    elif feature_type == "cpmg":
        input_data = np.array(all_features).reshape(n, -1)
        input_data_real = input_data.real
        input_data_imag = input_data.imag
        input_data_ri = np.array([input_data_real, input_data_imag]).reshape(n, 2, -1)
    elif feature_type == "fft_cpmg":
        input_data = np.array(all_features).reshape(n, -1)
        
        new_data = np.diff(input_data, axis=1)
        new_data = fft.fft(new_data)

        input_data_ri = np.array([new_data.real, new_data.imag]).reshape(n, 2, -1)

    else:
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    target_values = np.array(all_targets)

    del all_features  # Free memory

    print(f"Sample shape: {input_data_ri.shape}")
    print(f"Target values shape: {target_values.shape}")
    print(f"\nRaw data range: [{input_data_ri.min():.4f}, {input_data_ri.max():.4f}]")
    print(f"Target values range: [{target_values.min():.4f}, {target_values.max():.4f}]")

    return input_data_ri, target_values


def get_data_splits_normalized(train_ratio=0.7, val_ratio=0.15, seed=42,
                               feature_type: str = "echo_sample", target_type: str = "mqi"):
    """
    Steps:
    1. Load raw data
    2. Shuffle data with fixed seed
    3. Split into train/val/test
    4. Compute normalization statistics on training set
    5. Apply those statistics to val/test sets

    Can use as features either "echo_sample" or "cpmg"
    Can use as target either "mqi" or "speed" or "displacement"

    Args:
        train_ratio: Proportion for training set (default: 0.7)
        val_ratio: Proportion for validation set (default: 0.15)
        seed: Random seed for reproducibility (default: 42)
        feature_type: Type of feature to use ("echo_sample" or "cpmg")
        target_type: Type of target to use ("mqi", "speed", or "displacement")

    Returns:
        tuple: (train_data, val_data, test_data, norm_stats)
            - train_data: tuple (echo_samples_train, mqi_train)
            - val_data: tuple (echo_samples_val, mqi_val)
            - test_data: tuple (echo_samples_test, mqi_test)
            - norm_stats: dict with 'mean' and 'std' for reference
    """
    
    # Step 1: Load raw data
    feature_vals, target_vals = load_raw_data(feature_type, target_type)

    # Step 2: Shuffle data
    print("Shuffling data...")
    
    np.random.seed(seed)
    shuffle_indices = np.random.permutation(len(feature_vals))
    feature_vals = feature_vals[shuffle_indices]
    target_vals = target_vals[shuffle_indices]
    
    # Step 3: Split into train/val/test
    print("Splitting train/val/test...")
    
    n_samples = len(feature_vals)
    n_train = int(train_ratio * n_samples)
    n_val = int(val_ratio * n_samples)
    n_test = n_samples - n_train - n_val

    feat_train = feature_vals[:n_train]
    target_train = target_vals[:n_train]

    feat_val = feature_vals[n_train:n_train+n_val]
    target_val = target_vals[n_train:n_train+n_val]
    
    feat_test = feature_vals[n_train+n_val:]
    target_test = target_vals[n_train+n_val:]

    # Step 4: Compute normalization statistics ONLY on training set
    # This computes individual channel statistics for real and imaginary channels
    print("Computing normalization statistics on training set...")
    if feature_type == "echo_sample": # shape (N, 2, 54, 12)
        feat_mean = feat_train.mean(axis=(0, 2, 3), keepdims=True)
        feat_std = feat_train.std(axis=(0, 2, 3), keepdims=True)

        print(f"Training set statistics:")
        print(f"  Channel 0 (Real):      mean={feat_mean[0,0,0,0]:.4f}, std={feat_std[0,0,0,0]:.4f}")
        print(f"  Channel 1 (Imaginary): mean={feat_mean[0,1,0,0]:.4f}, std={feat_std[0,1,0,0]:.4f}\n")
        
    elif feature_type == "cpmg" or feature_type == "fft_cpmg": # shape (N, 2, 2048)
        feat_mean = feat_train.mean(axis=(0, 2), keepdims=True)
        feat_std = feat_train.std(axis=(0, 2), keepdims=True)
        
        print(f"Training set statistics:")
        print(f"  Channel 0 (Real):      mean={feat_mean[0,0,0]:.4f}, std={feat_std[0,0,0]:.4f}")
        print(f"  Channel 1 (Imaginary): mean={feat_mean[0,1,0]:.4f}, std={feat_std[0,1,0]:.4f}\n")

    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    # Step 5: Apply normalization to all sets using TRAINING statistics
    print("Applying normalization using training set statistics...")
    
    feat_train_norm = (feat_train - feat_mean) / (feat_std)
    feat_val_norm = (feat_val - feat_mean) / (feat_std)
    feat_test_norm = (feat_test - feat_mean) / (feat_std)

    # Verify normalization worked on training set
    train_mean_check = feat_train_norm.mean()
    train_std_check = feat_train_norm.std()
    print(f"\nTraining set verification:")
    print(f"  Mean: {train_mean_check:.6e} (should be ≈ 0)")
    print(f"  Std:  {train_std_check:.4f} (should be ≈ 1)")

    if abs(train_mean_check) > 1e-6:
        print(" WARNING: Mean is not close to 0!")
    if abs(train_std_check - 1.0) > 0.1:
        print(" WARNING: Std is not close to 1!")

    # Package results
    train_data = (feat_train_norm, target_train)
    val_data = (feat_val_norm, target_val)
    test_data = (feat_test_norm, target_test)
    norm_stats = {'mean': feat_mean, 'std': feat_std}

    return train_data, val_data, test_data, norm_stats


def verify_split_after_shuffle(train_dataset, val_dataset, test_dataset):
    
    train_targets = train_dataset[1]
    val_targets = val_dataset[1]
    test_targets = test_dataset[1]

    print(f"\n{'Split':<10} {'N':<8} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-"*60)
    print(f"{'Train':<10} {len(train_targets):<8} {train_targets.mean():<10.4f} "
          f"{train_targets.std():<10.4f} {train_targets.min():<10.4f} {train_targets.max():<10.4f}")
    print(f"{'Val':<10} {len(val_targets):<8} {val_targets.mean():<10.4f} "
          f"{val_targets.std():<10.4f} {val_targets.min():<10.4f} {val_targets.max():<10.4f}")
    print(f"{'Test':<10} {len(test_targets):<8} {test_targets.mean():<10.4f} "
          f"{test_targets.std():<10.4f} {test_targets.min():<10.4f} {test_targets.max():<10.4f}")

    # Check if distributions are similar
    mean_diff_val = abs(val_targets.mean() - train_targets.mean()) / train_targets.mean() * 100
    mean_diff_test = abs(test_targets.mean() - train_targets.mean()) / train_targets.mean() * 100
    print(f"\nDifference from train mean:")
    print(f"  Val:  {mean_diff_val:.2f}%")
    print(f"  Test: {mean_diff_test:.2f}%")

    if mean_diff_val > 20 or mean_diff_test > 20:
        print("\n WARNING: Large difference in means - shuffle may not have worked!")
    else:
        print("\n Splits look good - similar distributions")


class FeatureExtractor:
    """
    Extract engineered features from NMR samples.

    Converts raw (2, 54, 12) data to interpretable statistical features
    that work better with classical ML models like XGBoost.
    """

    def __init__(self, use_pca: bool = False, n_pca_components: int = 50,
                 use_rfecv: bool = False, feature_type: str = "echo_sample"):
        """
        Initialize feature extractor.

        Args:
            use_pca: Whether to apply PCA for dimensionality reduction
            n_pca_components: Number of PCA components to keep
        """
        self.use_pca = use_pca
        self.n_pca_components = n_pca_components
        self.use_rfecv = use_rfecv
        self.feature_type = feature_type

        self.rfecv = None
        self.pca = None
        self.scaler = None

    def extract_statistical_features_echo(self, echo_samples: np.ndarray) -> np.ndarray:
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

    def extract_statistical_features_cpmg(self, cpmg_samples: np.ndarray) -> np.ndarray:
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
        N = cpmg_samples.shape[0]
        features = []

        for i in range(N):
            sample_features = []

            # Process each channel (real, imaginary)
            for ch in range(2):
                channel_data = cpmg_samples[i, ch, :]
    
                # Basic statistics
                sample_features.append(np.mean(channel_data))
                sample_features.append(np.std(channel_data))
                sample_features.append(np.min(channel_data))
                sample_features.append(np.max(channel_data))

                # Peak information
                sample_features.append(np.max(np.abs(channel_data)))
                #sample_features.append(np.argmax(np.abs(channel_data)) / 54.0)  # Normalized position
                # Signal energy
                sample_features.append(np.sum(channel_data ** 2))

                # Kurtosis (peakedness)
                if np.std(channel_data) > 1e-8:
                    kurtosis = np.mean((channel_data - np.mean(channel_data)) ** 4) / (np.std(channel_data) ** 4)
                else:
                    kurtosis = 0.0
                sample_features.append(kurtosis)

                # Global channel statistics
                sample_features.append(np.mean(channel_data))
                sample_features.append(np.std(channel_data))
                sample_features.append(np.sum(channel_data ** 2))  # Total energy

            # Cross-channel features
            real_channel = cpmg_samples[i, 0, :]
            imag_channel = cpmg_samples[i, 1, :]

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
    
    def extract_fft_features(self, samples: np.ndarray) -> np.ndarray:
        """
        Extract FFT features from samples.
        """
        #real_part = np.diff(samples.real.flatten())
        #imag_part = np.diff(samples.imag.flatten())
        #new_data = np.empty(samples.real.shape, dtype=np.complex64)
        #new_data.real = real_part
        #new_data.imag = imag_part

        new_data = np.diff(samples, axis=2)
        new_data = fft.fft(new_data)
        
        N = samples.shape[0]
        features = new_data.reshape(N, -1)

        return features

    def extract_raw_features(self, samples: np.ndarray) -> np.ndarray:
        """
        Simple flattening of raw echo samples.

        Args:
            samples: Array of shape (N, 2, 54, 12)

        Returns:
            Flattened features of shape (N, 1296)
        """

        N = samples.shape[0]
        features = samples.reshape(N, -1)
        print(f"Flattened to {features.shape[1]} raw features")
        return features
        
    def fit_transform(self, samples: np.ndarray, method: str = 'statistical') -> np.ndarray:
        """
        Extract features and optionally apply PCA.

        Args:
            echo_sample: Array of shape (N, 2, 54, 12)
            method: 'statistical' or 'raw'

        Returns:
            Transformed features
        """
        # Extract features
        if method == 'statistical' and self.feature_type == "echo_sample":
            features = self.extract_statistical_features_echo(samples)
        elif method == "statistical" and (self.feature_type == "cpmg" or self.feature_type == "fft_cpmg"):
            features = self.extract_statistical_features_cpmg(samples)
        elif method == 'raw':
            features = self.extract_raw_features(samples)
        else:
            raise ValueError(f"Unknown method: {method} or {self.feature_type}")

        # Apply PCA if requested
        if self.use_pca:
            self.pca = PCA(n_components=min(self.n_pca_components, features.shape[1]))
            features = self.pca.fit_transform(features)
            variance_explained = np.sum(self.pca.explained_variance_ratio_)
            print(f"PCA: {features.shape[1]} components explain {variance_explained*100:.2f}% variance")

        return features

    def transform(self, samples: np.ndarray, method: str = 'statistical') -> np.ndarray:
        """
        Transform features using fitted parameters (for validation/test sets).

        Args:
            samples: Array of shape (N, 2, 54, 12)
            method: 'statistical' or 'raw'

        Returns:
            Transformed features
        """
        # Extract features
        if method == 'statistical' and self.feature_type == "echo_sample":
            features = self.extract_statistical_features_echo(samples)
        elif method == "statistical" and (self.feature_type == "cpmg" or self.feature_type == "fft_cpmg"):
            features = self.extract_statistical_features_cpmg(samples)
        elif method == 'raw':
            features = self.extract_raw_features(samples)
        else:
            raise ValueError(f"Unknown method: {method} or {self.feature_type}")

        # Apply PCA if it was fitted
        if self.use_pca and self.pca is not None:
            features = self.pca.transform(features)

        return features