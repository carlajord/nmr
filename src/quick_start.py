"""
Quick Start Guide - NMR MQI Prediction
=======================================

This script demonstrates how to quickly get started with the models.
"""

import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from baseline_model import BaselineCNN, NMRDataset, train_model, evaluate_model
from advanced_model import AdvancedNMRNet, train_advanced_model, evaluate_advanced_model
from model_comparison import run_complete_comparison
from torch.utils.data import DataLoader, random_split

from utils import set_seed, get_data_splits_normalized

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results")
BATCH_SIZE = 64

def example_1_train_baseline_only():
    """
    Example 1: Train only the baseline model with proper data handling
    """
    print("="*80)
    print("EXAMPLE 1: Training Baseline Model")
    print("="*80)

    # Set seeds for reproducibility
    set_seed(42)

    # Load data with proper normalization (NO leakage!)
    train_data, val_data, test_data, norm_stats = get_data_splits_normalized(
        train_ratio=0.7, val_ratio=0.15, seed=42
    )

    # Unpack data
    echo_train, mqi_train = train_data
    echo_val, mqi_val = val_data
    echo_test, mqi_test = test_data

    # Create PyTorch datasets
    from torch.utils.data import TensorDataset
    train_dataset = TensorDataset(
        torch.FloatTensor(echo_train),
        torch.FloatTensor(mqi_train).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(echo_val),
        torch.FloatTensor(mqi_val).unsqueeze(1)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(echo_test),
        torch.FloatTensor(mqi_test).unsqueeze(1)
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize and train model
    model = BaselineCNN(dropout_rate=0.3)  # Increased dropout
    print(f"\nTraining baseline model with {sum(p.numel() for p in model.parameters()):,} parameters")

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        learning_rate=0.001
    )

    # Load best model and evaluate
    model.load_state_dict(torch.load(os.path.join(RESULTS_PATH, 'best_baseline_model.pth')))
    mse, mae, preds, targets = evaluate_model(model, test_loader)

    print(f"\n✓ Baseline model training complete!")
    print(f"  MSE: {mse:.4f}, MAE: {mae:.4f}")


def example_2_train_advanced_only():
    """
    Example 2: Train only the advanced model with proper data handling
    """
    print("="*80)
    print("EXAMPLE 2: Training Advanced Model")
    print("="*80)

    # Set seeds for reproducibility
    set_seed(42)

    # Load data with proper normalization (NO leakage!)
    train_data, val_data, test_data, norm_stats = get_data_splits_normalized(
        train_ratio=0.7, val_ratio=0.15, seed=42
    )

    # Unpack data
    echo_train, mqi_train = train_data
    echo_val, mqi_val = val_data
    echo_test, mqi_test = test_data

    # Create PyTorch datasets
    from torch.utils.data import TensorDataset
    train_dataset = TensorDataset(
        torch.FloatTensor(echo_train),
        torch.FloatTensor(mqi_train).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(echo_val),
        torch.FloatTensor(mqi_val).unsqueeze(1)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(echo_test),
        torch.FloatTensor(mqi_test).unsqueeze(1)
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize and train model with higher dropout
    model = AdvancedNMRNet(dropout_rate=0.4)  # Increased dropout for regularization
    print(f"\nTraining advanced model with {sum(p.numel() for p in model.parameters()):,} parameters")

    history = train_advanced_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        learning_rate=0.001,
        use_early_stopping=True
    )

    # Load best model and evaluate
    model.load_state_dict(torch.load(os.path.join(RESULTS_PATH, 'best_advanced_model.pth')))
    mse, mae, r2, preds, targets = evaluate_advanced_model(model, test_loader)

    print(f"\n✓ Advanced model training complete!")
    print(f"  MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")


def example_3_compare_both_models():


    run_complete_comparison(
        batch_size=BATCH_SIZE,
        baseline_epochs=50,
        advanced_epochs=100,
        seed=42,  # For reproducibility
        train_ratio=0.7,
        val_ratio=0.15
    )

    print("\n✓ Complete comparison finished!")

def example_4_inference_with_trained_model():
    """
    Example 4: Use a trained model for inference
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Inference with Trained Model")
    print("="*80)
    
    # First train a model (or load existing one)
    num_samples = 100  # Small dataset for quick demo
    echo_samples = np.random.randn(num_samples, 2, 54, 12).astype(np.float32)
    mqi_values = np.random.rand(num_samples).astype(np.float32)
    
    dataset = NMRDataset(echo_samples, mqi_values)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Quick training
    model = BaselineCNN(dropout_rate=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    model.train()
    for epoch in range(5):  # Just 5 epochs for demo
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    # Now use for inference
    model.eval()
    
    # New sample for prediction
    new_sample = np.random.randn(1, 2, 54, 12).astype(np.float32)
    new_sample_tensor = torch.FloatTensor(new_sample)
    
    with torch.no_grad():
        prediction = model(new_sample_tensor)
    
    print(f"\nPredicted MQI: {prediction.item():.4f}")
    print("✓ Inference complete!")


if __name__ == "__main__":
    
    # example_1_train_baseline_only()
    # example_2_train_advanced_only()
    example_3_compare_both_models()  # This is the most comprehensive
    # example_4_inference_with_trained_model()
