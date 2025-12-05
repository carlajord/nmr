"""
XGBoost Model for NMR Measurement Quality Index (MQI) Prediction

This module implements XGBoost (Gradient Boosting) - a classical ML approach
with feature engineering for NMR echo sample analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from sklearn.decomposition import PCA
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import time

from src.utils import set_seed
import src.utils as ut

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results_reg")
SEED = 42

class XGBoostModel:
    """
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
        feature_method: str = 'raw',
        use_pca: bool = False,
        feature_type: str = "echo_sample"
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
       
        self.model = xgb.XGBRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=SEED,
            n_jobs=-1,
            early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
            device='gpu'
        )

        self.use_feature_engineering = use_feature_engineering
        self.feature_method = feature_method
        self.feature_type = feature_type
        self.feature_extractor = ut.FeatureExtractor(use_pca=use_pca, feature_type=feature_type)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            verbose: bool = True):
        """
        Train XGBoost model.

        Args:
            X_train: Training data flattened (N, features)
            y_train: Training targets (N,)
            X_val: Validation data (optional)
            y_val: Validation targets (optional)
            verbose: Print progress
        """
        # Extract features
        if self.use_feature_engineering or self.feature_method == 'statistical':
            X_train_feat = self.feature_extractor.fit_transform(X_train, method=self.feature_method)
            eval_set = None
            if X_val is not None and y_val is not None:
                X_val_feat = self.feature_extractor.transform(X_val, method=self.feature_method)
                eval_set = [(X_train_feat, y_train), (X_val_feat, y_val)]
        else:
            X_train_feat = X_train.reshape(len(X_train), -1)
            eval_set = None
            if X_val is not None and y_val is not None:
                X_val_feat = X_val.reshape(len(X_val), -1)
                eval_set = [(X_train_feat, y_train), (X_val_feat, y_val)]

        # Train
        self.model.fit(
            X_train_feat, y_train,
            eval_set=eval_set,
            verbose=verbose
        )

        if verbose:
            print(f"\nXGBoost training complete")
            print(f"  Features used: {X_train_feat.shape[1]}")
            print(f"  Trees built: {self.model.n_estimators}")
        
        results = self.model.evals_result()
        epochs = len(results['validation_0']['rmse'])
        x_axis = range(0, epochs)
        ig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
        ax.plot(x_axis, results['validation_1']['rmse'], label='Validation')
        ax.legend()
        plt.ylabel('RMSE')
        plt.title('XGBoost RMSE')

        save_path = os.path.join(RESULTS_PATH, 'xgboost_regression_learning_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input data (N, features)

        Returns:
            Predictions (N,)
        """
        if self.use_feature_engineering or self.feature_method == 'statistical':
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


def train_xgboost_model(
    model: XGBoostModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None
) -> Dict:
    """Train XGBoost regression model."""
    
    model.fit(X_train, y_train, X_val, y_val, verbose=True)

    # Get feature importance
    importance = model.get_feature_importance(top_n=10)
    print(f"\nTop 10 important features:")
    for idx, (feat_idx, score) in enumerate(zip(importance['indices'], importance['scores'])):
        print(f"  {idx+1}. Feature {feat_idx}: {score:.4f}")

    return {'model': model}


def evaluate_xgboost_model(
    model: XGBoostModel,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[float, float, float, np.ndarray]:
    
    # Get predictions
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


def plot_xgboost_prediction_analysis(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: str = os.path.join(RESULTS_PATH, 'xgb_reg_analysis.png')
):
    """
    Create prediction analysis plots for XGBoost Regression model.

    Args:
        predictions: Predictions from XGBoost Regression model
        targets: Ground truth values
        save_path: Path to save the plot
    """
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
    axes[0, 0].set_xlabel('Actual', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Predicted', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('XGBoost: Predicted vs Actual', fontsize=14, fontweight='bold')
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
    axes[0, 1].set_xlabel('Actual', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Residual (Predicted - Actual)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('XGBoost: Residual Plot', fontsize=14, fontweight='bold')
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
    axes[1, 0].set_title('XGBoost: Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # ========== Plot 4: Error Statistics ==========
    # Box plot and violin plot combined view
    axes[1, 1].boxplot([abs_errors], labels=['XGBoost'],
                       patch_artist=True,
                       boxprops=dict(facecolor=color, alpha=0.7),
                       medianprops=dict(color='red', linewidth=2),
                       widths=0.5)
    axes[1, 1].set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('XGBoost: Absolute Error Statistics', fontsize=14, fontweight='bold')
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
    print(f"\nXGBoost prediction analysis plot saved to: {save_path}")
    plt.close()


# ============================================================================
# HYPERPARAMETER OPTIMIZATION
# ============================================================================

def optimize_xgboost_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 100,
    use_feature_engineering: bool = True,
    feature_method: str = 'statistical',
    use_pca: bool = False,
    timeout: int = None,
    seed: int = SEED,
    feature_type: str = 'echo_sample',
    plot_suffix: str = ''
) -> Dict:
    """
    Optimize XGBoost hyperparameters using Optuna.

    This function uses Bayesian optimization (TPE sampler) to find the best
    hyperparameters for the XGBoost model based on validation set performance.

    Args:
        X_train: Training data (N, 2, 54, 12)
        y_train: Training targets (N,)
        X_val: Validation data (M, 2, 54, 12)
        y_val: Validation targets (M,)
        n_trials: Number of optimization trials (default: 100)
        use_feature_engineering: Use statistical features vs raw flattening
        feature_method: 'statistical' or 'raw'
        use_pca: Whether to use PCA for dimensionality reduction
        timeout: Time limit in seconds (None for no limit)
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - best_params: Best hyperparameters found
            - best_score: Best validation RMSE achieved
            - study: Optuna study object for further analysis
            - best_model: Trained model with best parameters
    """

    print("\n" + "="*80)
    print("XGBOOST HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("="*80)
    print(f"Configuration:")
    print(f"  Number of trials: {n_trials}")
    print(f"  Feature engineering: {use_feature_engineering}")
    print(f"  Feature method: {feature_method}")
    print(f"  Use PCA: {use_pca}")
    print(f"  Timeout: {timeout if timeout else 'None'}")
    print(f"  Random seed: {seed}")
    print("="*80)

    def objective(trial):
        """
        Objective function for Optuna to minimize.
        Returns validation RMSE.
        """
        # Sample hyperparameters
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'n_estimators': 1000,  # Fixed to a high number with early stopping
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True)
        }

        # Create and train model with suggested parameters
        model = XGBoostModel(
            max_depth=params['max_depth'],
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            colsample_bytree=params['colsample_bytree'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            use_feature_engineering=use_feature_engineering,
            feature_method=feature_method,
            use_pca=use_pca,
            feature_type=feature_type
        )

        # Train model (suppress verbose output)
        model.fit(X_train, y_train, X_val, y_val, verbose=False)

        # Evaluate on validation set
        val_predictions = model.predict(X_val)
        val_rmse = np.sqrt(np.mean((val_predictions - y_val) ** 2))

        return val_rmse

    # Create Optuna study
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        study_name='xgb_reg_opt'
    )

    # Suppress Optuna's verbose output
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Run optimization
    print("\nStarting optimization...")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )

    # Get best parameters
    best_params = study.best_params
    best_score = study.best_value

    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    print(f"Best validation RMSE: {best_score:.6f}")
    print(f"\nBest hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print("="*80)

    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    best_model = XGBoostModel(
        max_depth=best_params['max_depth'],
        n_estimators=1000,
        learning_rate=best_params['learning_rate'],
        colsample_bytree=best_params['colsample_bytree'],
        reg_alpha=best_params['reg_alpha'],
        reg_lambda=best_params['reg_lambda'],
        use_feature_engineering=use_feature_engineering,
        feature_method=feature_method,
        use_pca=use_pca,
        feature_type=feature_type
    )

    best_model.fit(X_train, y_train, X_val, y_val, verbose=True)

    # Generate optimization history plot
    _plot_optimization_history(study, save_path=os.path.join(RESULTS_PATH, f'opt_reg_history{plot_suffix}.png'))

    return {
        'best_params': best_params,
        'best_score': best_score,
        'study': study,
        'best_model': best_model
    }


def _plot_optimization_history(
    study,
    save_path: str = os.path.join(RESULTS_PATH, 'opt_reg_history.png')
):
    """
    Plot Optuna optimization history.

    Args:
        study: Optuna study object
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Get trial data
    trials = study.trials
    trial_numbers = [t.number for t in trials]
    trial_values = [t.value for t in trials if t.value is not None]
    valid_trial_numbers = [t.number for t in trials if t.value is not None]

    # Plot 1: Optimization history
    axes[0, 0].plot(valid_trial_numbers, trial_values, 'b-', alpha=0.6, linewidth=1)
    axes[0, 0].scatter(valid_trial_numbers, trial_values, c=trial_values,
                       cmap='viridis', s=30, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Add best value line
    best_values = [study.best_value if i in valid_trial_numbers else None
                   for i in range(len(trial_numbers))]
    best_values_clean = []
    current_best = float('inf')
    for i, val in enumerate([t.value for t in trials]):
        if val is not None and val < current_best:
            current_best = val
        best_values_clean.append(current_best if current_best != float('inf') else None)

    valid_best = [(i, v) for i, v in enumerate(best_values_clean) if v is not None]
    if valid_best:
        axes[0, 0].plot([i for i, _ in valid_best], [v for _, v in valid_best],
                        'r--', linewidth=2, label=f'Best: {study.best_value:.6f}')

    axes[0, 0].set_xlabel('Trial', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Validation RMSE', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Optimization History', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Parameter importances
    try:
        importances = optuna.importance.get_param_importances(study)
        params = list(importances.keys())
        values = list(importances.values())

        axes[0, 1].barh(params, values, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Importance', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Hyperparameter Importance', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='x')
    except Exception as e:
        axes[0, 1].text(0.5, 0.5, f'Importance plot unavailable:\n{str(e)}',
                        ha='center', va='center', transform=axes[0, 1].transAxes)

    # Plot 3: Distribution of best parameter (learning_rate)
    learning_rates = [t.params.get('learning_rate') for t in trials if t.value is not None]
    if learning_rates:
        axes[1, 0].hist(learning_rates, bins=30, alpha=0.7, color='green',
                        edgecolor='black', linewidth=1.2)
        axes[1, 0].axvline(x=study.best_params['learning_rate'], color='r',
                          linestyle='--', linewidth=2.5,
                          label=f"Best: {study.best_params['learning_rate']:.4f}")
        axes[1, 0].set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Learning Rate Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Plot 4: Distribution of best parameter (max_depth)
    max_depths = [t.params.get('max_depth') for t in trials if t.value is not None]
    if max_depths:
        axes[1, 1].hist(max_depths, bins=range(3, 12), alpha=0.7, color='orange',
                        edgecolor='black', linewidth=1.2)
        axes[1, 1].axvline(x=study.best_params['max_depth'], color='r',
                          linestyle='--', linewidth=2.5,
                          label=f"Best: {study.best_params['max_depth']}")
        axes[1, 1].set_xlabel('Max Depth', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Max Depth Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Optimization history plot saved to: {save_path}")
    plt.close()


def run(feature_type, target_type, feature_method,
        use_pca=False, use_feature_engineering=True
        ) -> Tuple[XGBoostModel, np.ndarray, Dict, Tuple[np.ndarray, np.ndarray]]:
    
    ut.set_seed(SEED)

    feature_type = feature_type #"cpmg", "echo_sample", "fft_cpmg"
    target_type = target_type # "mqi", "displacement" or "speed"
    feature_method = feature_method #'statistical' or "raw"
    use_pca = use_pca
    use_feature_engineering = use_feature_engineering

    # Load and split data
    train_data, val_data, test_data, norm_stats = ut.get_data_splits_normalized(
        train_ratio=0.7,
        val_ratio=0.15,
        seed=SEED,
        feature_type=feature_type,
        target_type=target_type
    )

    ut.verify_split_after_shuffle(train_data, val_data, test_data)

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

    # Initialize XGBoost model
    start_time = time.time()
    xgb_model = XGBoostModel(
        max_depth=5,
        n_estimators=1000,
        learning_rate=0.05,
        use_feature_engineering=use_feature_engineering,
        feature_method=feature_method,
        use_pca=use_pca,
        feature_type=feature_type
    )

    # Train model
    train_xgboost_model(xgb_model, X_train, y_train, X_val, y_val)

    # Evaluate model
    #mse, mae, r2, predictions = evaluate_xgboost_model(xgb_model, X_test, y_test)
    # Evaluate model
    mse_train, mae_train, r2_train, predictions_train = evaluate_xgboost_model(xgb_model, X_train, y_train)
    mse_val, mae_val, r2_val, predictions_val = evaluate_xgboost_model(xgb_model, X_val, y_val)
    mse_test, mae_test, r2_test, predictions_test = evaluate_xgboost_model(xgb_model, X_test, y_test)

    training_time = time.time() - start_time
    
    if feature_method == "statistical":
        if feature_type == "echo_sample":
            n_features = xgb_model.feature_extractor.extract_statistical_features_echo(X_train[:1]).shape[1]
        elif feature_type == "cpmg" or feature_type == "fft_cpmg":
            n_features = xgb_model.feature_extractor.extract_statistical_features_cpmg(X_train[:1]).shape[1]
        else:
            raise ValueError(f"Unsupported feature_type: {feature_type}")
    elif feature_method == "raw":
        n_features = X_train.shape[1]
    else:
        raise ValueError(f"Unsupported feature_method: {feature_method}")

    # Prepare metrics dictionary
    metrics = {
        'mse': mse_test,
        'mae': mae_test,
        'rmse': np.sqrt(mse_test),
        'r2': r2_test,
        'training_time': training_time,
        'n_features': n_features
    }

    print(f"\nTotal training and evaluation time: {training_time:.2f} seconds")

    # Generate prediction analysis plots
    #plot_xgboost_prediction_analysis(predictions, y_test)
    plot_xgboost_prediction_analysis(predictions_train, y_train,
                                     save_path=os.path.join(RESULTS_PATH, 'xgboost_regression_analysis_train.png'))
    plot_xgboost_prediction_analysis(predictions_val, y_val,
                                     save_path=os.path.join(RESULTS_PATH, 'xgboost_regression_analysis_val.png'))
    plot_xgboost_prediction_analysis(predictions_test, y_test,
                                     save_path=os.path.join(RESULTS_PATH, 'xgboost_regression_analysis_test.png'))
    

    return xgb_model, predictions_test, metrics, (X_test, y_test)

def run_opt_reg(
        feature_type: str = 'echo_sample',
        target_type: str = 'mqi',
        n_trials: int = 100,
        use_feature_engineering: bool = True,
        feature_method: str = 'raw',
        use_pca: bool = False,
        timeout: int = None,
        plot_suffix: str = ''
    ):
    """
    Run XGBoost with hyperparameter optimization.

    This function performs the complete workflow:
    1. Load and split data
    2. Optimize hyperparameters using Optuna
    3. Evaluate best model on test set
    4. Generate visualizations

    Args:
        n_trials: Number of optimization trials (default: 100)
        use_feature_engineering: Use statistical features vs raw flattening
        feature_method: 'statistical' or 'raw'
        use_pca: Whether to use PCA for dimensionality reduction
        timeout: Time limit in seconds (None for no limit)

    Returns:
        Dictionary containing:
            - best_model: Trained model with best parameters
            - best_params: Best hyperparameters found
            - predictions: Test set predictions
            - metrics: Performance metrics on test set
            - test_data: Test data tuple (X_test, y_test)
            - optimization_results: Full optimization results from Optuna
    """
    
    set_seed(SEED)

    # Load and split data
    print("\nLoading and splitting data...")
    train_data, val_data, test_data, norm_stats = ut.get_data_splits_normalized(
        train_ratio=0.7,
        val_ratio=0.15,
        seed=SEED,
        feature_type=feature_type,
        target_type=target_type
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

    # Run hyperparameter optimization
    start_time = time.time()
    optimization_results = optimize_xgboost_hyperparameters(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_trials=n_trials,
        use_feature_engineering=use_feature_engineering,
        feature_method=feature_method,
        use_pca=use_pca,
        timeout=timeout,
        seed=SEED,
        feature_type=feature_type,
        plot_suffix=plot_suffix
    )

    best_model = optimization_results['best_model']
    best_params = optimization_results['best_params']

    # Evaluate model
    mse_train, mae_train, r2_train, predictions_train = evaluate_xgboost_model(best_model, X_train, y_train)
    mse_val, mae_val, r2_val, predictions_val = evaluate_xgboost_model(best_model, X_val, y_val)
    mse_test, mae_test, r2_test, predictions_test = evaluate_xgboost_model(best_model, X_test, y_test)

    total_time = time.time() - start_time

    # Prepare metrics dictionary
    metrics = {
        'feature_type': feature_type,
        'target_type': target_type,
        'mse': mse_test,
        'mae': mae_test,
        'rmse': np.sqrt(mse_test),
        'r2': r2_test,
        'optimization_time': total_time,
        'best_validation_rmse': optimization_results['best_score'],
        'n_trials': n_trials
    }

    file_name = os.path.join(RESULTS_PATH, f"metrics{plot_suffix}.txt")
    with open(file_name, "w") as f:
        f.write(metrics.__str__())

    print(f"Content written to '{file_name}' in write mode.")

    print(f"\n" + "="*80)
    print("OPTIMIZATION SUMMARY")
    print("="*80)
    print(f"Total optimization time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Number of trials: {n_trials}")
    print(f"Best validation RMSE: {optimization_results['best_score']:.6f}")
    print(f"Test RMSE: {metrics['rmse']:.6f}")
    print(f"Test R²: {metrics['r2']:.6f}")
    print("="*80)

    best_model.model.save_model(os.path.join(RESULTS_PATH, f'xgb_reg{plot_suffix}.json'))
    
    # Generate prediction analysis plots
    plot_xgboost_prediction_analysis(predictions_train, y_train,
                                     save_path=os.path.join(RESULTS_PATH, f'opt_reg_train{plot_suffix}.png'))
    plot_xgboost_prediction_analysis(predictions_val, y_val,
                                     save_path=os.path.join(RESULTS_PATH, f'opt_reg_val{plot_suffix}.png'))
    plot_xgboost_prediction_analysis(predictions_test, y_test,
                                     save_path=os.path.join(RESULTS_PATH, f'opt_reg_test{plot_suffix}.png'))
    return {
        'best_model': best_model,
        'best_params': best_params,
        'predictions': predictions_test,
        'metrics': metrics,
        'test_data': (X_test, y_test),
        'optimization_results': optimization_results
    }
