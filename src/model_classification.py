"""
XGBoost Model for NMR Measurement Quality Index (MQI) Classification

This module implements XGBoost (Gradient Boosting) for multi-class classification
of NMR measurement quality with feature engineering for echo sample analysis.

Classification Categories:
- Class 0: Low quality (MQI < 0.5)
- Class 1: Medium quality (0.5 <= MQI <= 0.7)
- Class 2: High quality (MQI > 0.7)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import warnings

import time
import src.utils as ut

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results_class")
SEED = 42

def mqi_to_class_labels(mqi_values: np.ndarray) -> np.ndarray:
    """
    Convert continuous MQI values to class labels.

    Class 0: MQI < 0.5 (Low quality)
    Class 1: 0.5 <= MQI <= 0.7 (Medium quality)
    Class 2: MQI > 0.7 (High quality)

    Args:
        mqi_values: Array of continuous MQI values

    Returns:
        Array of class labels (0, 1, or 2)
    """
    labels = np.zeros(len(mqi_values), dtype=int)
    labels[mqi_values < 0.5] = 0  # Low quality
    labels[(mqi_values >= 0.5) & (mqi_values <= 0.7)] = 1  # Medium quality
    labels[mqi_values > 0.7] = 2  # High quality
    return labels


def get_class_distribution(labels: np.ndarray) -> Dict:
    """
    Get the distribution of classes.

    Args:
        labels: Array of class labels

    Returns:
        Dictionary with class counts and percentages
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    distribution = {}
    class_names = {0: "Low (MQI < 0.5)", 1: "Medium (0.5 <= MQI <= 0.7)", 2: "High (MQI > 0.7)"}

    for cls, count in zip(unique, counts):
        distribution[cls] = {
            'count': count,
            'percentage': (count / total) * 100,
            'name': class_names.get(cls, f"Class {cls}")
        }

    return distribution



class XGBoostModel:
    """
    XGBoost model for MQI quality classification.

    Multi-class classification into three categories:
    - Class 0: Low quality (MQI < 0.5)
    - Class 1: Medium quality (0.5 <= MQI <= 0.7)
    - Class 2: High quality (MQI > 0.7)

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
        feature_method: str = 'statistical',
        use_pca: bool = False,
        feature_type: str = 'echo_sample'
    ):
        """
        Initialize XGBoost classification model.

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
            use_pca: Whether to use PCA for dimensionality reduction
        """

        self.model = xgb.XGBClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=SEED,
            n_jobs=-1,
            num_class=3,
            objective='multi:softmax',
            eval_metric='mlogloss',
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
        Train XGBoost classification model.

        Args:
            X_train: Training data flattened (N, features)
            y_train: Training class labels (N,) - values should be 0, 1, or 2
            X_val: Validation data (optional)
            y_val: Validation class labels (optional)
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make class predictions.

        Args:
            X: Input data (N, features)

        Returns:
            Predicted class labels (N,) - values will be 0, 1, or 2
        """
        if self.use_feature_engineering or self.feature_method == 'statistical':
            X_feat = self.feature_extractor.transform(X, method=self.feature_method)
        else:
            X_feat = X.reshape(len(X), -1)

        return self.model.predict(X_feat)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input data (N, 2, 54, 12)

        Returns:
            Class probabilities (N, 3) for each of the 3 classes
        """
        if self.use_feature_engineering or self.feature_method == 'statistical':
            X_feat = self.feature_extractor.transform(X, method=self.feature_method)
        else:
            X_feat = X.reshape(len(X), -1)

        return self.model.predict_proba(X_feat)

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
    """Train XGBoost classification model."""
    
    # Show class distribution
    print("\nTraining set class distribution:")
    train_dist = get_class_distribution(y_train)
    for cls in sorted(train_dist.keys()):
        info = train_dist[cls]
        print(f"  {info['name']}: {info['count']} samples ({info['percentage']:.1f}%)")

    if y_val is not None:
        print("\nValidation set class distribution:")
        val_dist = get_class_distribution(y_val)
        for cls in sorted(val_dist.keys()):
            info = val_dist[cls]
            print(f"  {info['name']}: {info['count']} samples ({info['percentage']:.1f}%)")

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
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    
    # Show test set class distribution
    print("\nTest set class distribution:")
    test_dist = get_class_distribution(y_test)
    for cls in sorted(test_dist.keys()):
        info = test_dist[cls]
        print(f"  {info['name']}: {info['count']} samples ({info['percentage']:.1f}%)")

    # Get predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predictions, average='weighted', zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)

    # Print overall metrics
    print(f"\n{'='*80}")
    print(f'Accuracy:  {accuracy:.4f}')
    print(f'Precision: {precision:.4f} (weighted)')
    print(f'Recall:    {recall:.4f} (weighted)')
    print(f'F1-Score:  {f1:.4f} (weighted)')
    print(f"{'='*80}")

    # Print per-class metrics
    print(f"\nPer-Class Metrics:")
    class_names = ["Low (MQI < 0.5)", "Medium (0.5 <= MQI <= 0.7)", "High (MQI > 0.7)"]
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_test, predictions, average=None, zero_division=0
    )

    for i, class_name in enumerate(class_names):
        if i < len(precision_per_class):
            print(f"\n  {class_name}:")
            print(f"    Precision: {precision_per_class[i]:.4f}")
            print(f"    Recall:    {recall_per_class[i]:.4f}")
            print(f"    F1-Score:  {f1_per_class[i]:.4f}")
            print(f"    Support:   {support_per_class[i]}")

    # Print confusion matrix
    print(f"\nConfusion Matrix:")
    print(f"{'':>10} {'Predicted':^30}")
    print(f"{'':>10} {'Low':>10} {'Medium':>10} {'High':>10}")
    print(f"{'Actual':>10}")
    for i, class_name in enumerate(["Low", "Medium", "High"]):
        if i < cm.shape[0]:
            row_str = f"{class_name:>10}"
            for j in range(min(3, cm.shape[1])):
                row_str += f"{cm[i, j]:>10}"
            print(row_str)

    print(f"{'='*80}\n")

    return accuracy, precision, recall, f1, predictions, cm


def plot_xgboost_prediction_analysis(
    predictions: np.ndarray,
    targets: np.ndarray,
    confusion_mat: np.ndarray = None,
    save_path: str = os.path.join(RESULTS_PATH, 'xgb_class_analysis.png')
):
    """
    Create classification analysis plots for XGBoost model.

    Args:
        predictions: Predicted class labels from XGBoost model
        targets: Ground truth class labels
        confusion_mat: Confusion matrix (optional, will be computed if not provided)
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Flatten arrays for consistency
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Calculate confusion matrix if not provided
    if confusion_mat is None:
        confusion_mat = confusion_matrix(targets, predictions)

    class_names = ['Low\n(MQI < 0.5)', 'Medium\n(0.5 ≤ MQI ≤ 0.7)', 'High\n(MQI > 0.7)']
    class_short = ['Low', 'Medium', 'High']

    # ========== Plot 1: Confusion Matrix Heatmap ==========
    im = axes[0, 0].imshow(confusion_mat, cmap='Blues', aspect='auto')
    axes[0, 0].set_xticks(np.arange(len(class_short)))
    axes[0, 0].set_yticks(np.arange(len(class_short)))
    axes[0, 0].set_xticklabels(class_short)
    axes[0, 0].set_yticklabels(class_short)
    axes[0, 0].set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Actual Class', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    # Add text annotations
    for i in range(len(class_short)):
        for j in range(len(class_short)):
            if i < confusion_mat.shape[0] and j < confusion_mat.shape[1]:
                text_color = 'white' if confusion_mat[i, j] > confusion_mat.max() / 2 else 'black'
                axes[0, 0].text(j, i, str(confusion_mat[i, j]),
                              ha="center", va="center", color=text_color, fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=axes[0, 0])

    # ========== Plot 2: Class Distribution Comparison ==========
    x_pos = np.arange(len(class_short))
    width = 0.35

    actual_counts = np.bincount(targets.astype(int), minlength=3)
    pred_counts = np.bincount(predictions.astype(int), minlength=3)

    axes[0, 1].bar(x_pos - width/2, actual_counts, width, label='Actual',
                   alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.2)
    axes[0, 1].bar(x_pos + width/2, pred_counts, width, label='Predicted',
                   alpha=0.8, color='orange', edgecolor='black', linewidth=1.2)

    axes[0, 1].set_xlabel('Class', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Class Distribution: Actual vs Predicted', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(class_short)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # ========== Plot 3: Per-Class Accuracy ==========
    per_class_accuracy = []
    for i in range(len(class_short)):
        if i < confusion_mat.shape[0]:
            class_total = np.sum(confusion_mat[i, :])
            if class_total > 0:
                class_correct = confusion_mat[i, i]
                per_class_accuracy.append(class_correct / class_total)
            else:
                per_class_accuracy.append(0)
        else:
            per_class_accuracy.append(0)

    colors_acc = ['#d62728' if acc < 0.5 else '#ff7f0e' if acc < 0.7 else '#2ca02c'
                  for acc in per_class_accuracy]

    bars = axes[1, 0].bar(class_short, per_class_accuracy, alpha=0.8,
                          color=colors_acc, edgecolor='black', linewidth=1.2)
    axes[1, 0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Class', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylim([0, 1.0])
    #axes[1, 0].axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='50% threshold')
    #axes[1, 0].axhline(y=0.7, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='70% threshold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].legend(fontsize=10)

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, per_class_accuracy)):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{acc:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # ========== Plot 4: Prediction Correctness ==========
    correct = (predictions == targets)
    correct_counts = [np.sum(correct), np.sum(~correct)]
    labels_correct = ['Correct', 'Incorrect']
    colors_pie = ['#2ca02c', '#d62728']

    wedges, texts, autotexts = axes[1, 1].pie(correct_counts, labels=labels_correct, autopct='%1.1f%%',
                                               colors=colors_pie, startangle=90,
                                               textprops={'fontsize': 12, 'fontweight': 'bold'},
                                               wedgeprops={'edgecolor': 'black', 'linewidth': 1.2})

    axes[1, 1].set_title(f'Overall Accuracy: {np.mean(correct):.2%}\n({np.sum(correct)}/{len(correct)} samples)',
                        fontsize=14, fontweight='bold')

    # Make percentage text white for better visibility
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nXGBoost classification analysis plot saved to: {save_path}")
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
    metric: str = 'f1',
    feature_type: str = 'echo_sample',
    plot_suffix: str = ''
) -> Dict:
    """
    Optimize XGBoost hyperparameters using Optuna for classification.

    This function uses Bayesian optimization (TPE sampler) to find the best
    hyperparameters for the XGBoost classifier based on validation set performance.

    Args:
        X_train: Training data (N, 2, 54, 12)
        y_train: Training class labels (N,) - values should be 0, 1, or 2
        X_val: Validation data (M, 2, 54, 12)
        y_val: Validation class labels (M,)
        n_trials: Number of optimization trials (default: 100)
        use_feature_engineering: Use statistical features vs raw flattening
        feature_method: 'statistical' or 'raw'
        use_pca: Whether to use PCA for dimensionality reduction
        timeout: Time limit in seconds (None for no limit)
        seed: Random seed for reproducibility
        metric: Metric to optimize ('accuracy' or 'f1'). Default: 'f1'

    Returns:
        Dictionary containing:
            - best_params: Best hyperparameters found
            - best_score: Best validation score achieved
            - study: Optuna study object for further analysis
            - best_model: Trained model with best parameters
    """

    print("\n" + "="*80)
    print("XGBOOST CLASSIFICATION HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("="*80)
    print(f"Configuration:")
    print(f"  Number of trials: {n_trials}")
    print(f"  Feature engineering: {use_feature_engineering}")
    print(f"  Feature method: {feature_method}")
    print(f"  Use PCA: {use_pca}")
    print(f"  Optimization metric: {metric}")
    print(f"  Timeout: {timeout if timeout else 'None'}")
    print(f"  Random seed: {seed}")
    print("="*80)

    def objective(trial):
        """
        Objective function for Optuna to maximize.
        Returns validation accuracy or F1 score (negated for minimization).
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

        if metric == 'accuracy':
            score = accuracy_score(y_val, val_predictions)
        elif metric == 'f1':
            _, _, score, _ = precision_recall_fscore_support(
                y_val, val_predictions, average='weighted', zero_division=0
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Return negative score for minimization (Optuna minimizes by default)
        return -score

    # Create Optuna study (maximize since we return negative score)
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        direction='minimize',  # We minimize negative score, which is equivalent to maximizing score
        sampler=sampler,
        study_name='xgb_class_opt'
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
    best_score = -study.best_value  # Convert back to positive score

    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    metric_name = 'Accuracy' if metric == 'accuracy' else 'F1-Score (weighted)'
    print(f"Best validation {metric_name}: {best_score:.6f}")
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
    _plot_optimization_history(study, metric=metric, save_path=os.path.join(RESULTS_PATH, f'opt_class_history{plot_suffix}.png'))

    return {
        'best_params': best_params,
        'best_score': best_score,
        'study': study,
        'best_model': best_model
    }


def _plot_optimization_history(
    study,
    metric: str = 'f1',
    save_path: str = os.path.join(RESULTS_PATH, 'opt_class_history.png')
):
    """
    Plot Optuna optimization history for classification.

    Args:
        study: Optuna study object
        metric: Metric name being optimized ('accuracy' or 'f1')
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Get trial data (convert negative values back to positive)
    trials = study.trials
    trial_numbers = [t.number for t in trials]
    trial_values = [-t.value for t in trials if t.value is not None]  # Negate back to positive
    valid_trial_numbers = [t.number for t in trials if t.value is not None]

    metric_name = 'Accuracy' if metric == 'accuracy' else 'F1-Score'

    # Plot 1: Optimization history
    axes[0, 0].plot(valid_trial_numbers, trial_values, 'b-', alpha=0.6, linewidth=1)
    axes[0, 0].scatter(valid_trial_numbers, trial_values, c=trial_values,
                       cmap='viridis_r', s=30, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Add best value line
    best_values_clean = []
    current_best = -float('inf')
    for i, val in enumerate([-t.value for t in trials]):
        if val is not None and val > current_best:
            current_best = val
        best_values_clean.append(current_best if current_best != -float('inf') else None)

    valid_best = [(i, v) for i, v in enumerate(best_values_clean) if v is not None]
    if valid_best:
        axes[0, 0].plot([i for i, _ in valid_best], [v for _, v in valid_best],
                        'r--', linewidth=2, label=f'Best: {-study.best_value:.6f}')

    axes[0, 0].set_xlabel('Trial', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel(f'Validation {metric_name}', fontsize=12, fontweight='bold')
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


def run(
    feature_type: str,
    target_type: str,
    feature_method: str,
    use_pca: bool = False,
    use_feature_engineering: bool = True
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
        seed=SEED
    )

    X_train = train_data[0]
    y_train_continuous = train_data[1]
    X_val = val_data[0]
    y_val_continuous = val_data[1]
    X_test = test_data[0]
    y_test_continuous = test_data[1]

    # Convert continuous MQI values to class labels
    y_train = mqi_to_class_labels(y_train_continuous)
    y_val = mqi_to_class_labels(y_val_continuous)
    y_test = mqi_to_class_labels(y_test_continuous)

    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")

    # Initialize XGBoost classification model
    start_time = time.time()
    xgb_model = XGBoostModel(
        max_depth=5,
        n_estimators=1000,
        learning_rate=0.05,
        use_feature_engineering=True,
        feature_method='raw',  # or 'statistical'
        use_pca=False
    )

    # Train model
    train_xgboost_model(xgb_model, X_train, y_train, X_val, y_val)

    # Evaluate model
    accuracy_train, precision_train, recall_train, f1_train, predictions_train, cm_train = evaluate_xgboost_model(xgb_model, X_train, y_train)
    accuracy_val, precision_val, recall_val, f1_val, predictions_val, cm_val = evaluate_xgboost_model(xgb_model, X_val, y_val)
    accuracy_test, precision_test, recall_test, f1_test, predictions_test, cm_test = evaluate_xgboost_model(xgb_model, X_test, y_test)

    training_time = time.time() - start_time

    # Prepare metrics dictionary
    metrics = {
        'accuracy': accuracy_test,
        'precision': precision_test,
        'recall': recall_test,
        'f1': f1_test,
        'training_time': training_time,
        'confusion_matrix': cm_test
    }

    print(f"\nTotal training and evaluation time: {training_time:.2f} seconds")

    # Generate prediction analysis plots
    #plot_xgboost_prediction_analysis(predictions, y_test, cm)
    plot_xgboost_prediction_analysis(predictions_train, y_train,
                                     save_path=os.path.join(RESULTS_PATH, 'xgboost_prediction_analysis_train.png'))
    plot_xgboost_prediction_analysis(predictions_val, y_val,
                                     save_path=os.path.join(RESULTS_PATH, 'xgboost_prediction_analysis_val.png'))
    plot_xgboost_prediction_analysis(predictions_test, y_test,
                                     save_path=os.path.join(RESULTS_PATH, 'xgboost_prediction_analysis_test.png'))
    
    return xgb_model, predictions_test, metrics, (X_test, y_test)


def run_opt_class(
        feature_type: str = 'echo_sample',
        target_type: str = 'mqi',
        n_trials: int = 100,
        use_feature_engineering: bool = True,
        feature_method: str = 'raw',
        use_pca: bool = False,
        timeout: int = None,
        metric: str = 'f1',
        plot_suffix: str = ''
    ):
    """
    Run XGBoost classification with hyperparameter optimization.

    This function performs the complete workflow:
    1. Load and split data
    2. Convert MQI values to class labels
    3. Optimize hyperparameters using Optuna
    4. Evaluate best model on test set
    5. Generate visualizations

    Args:
        n_trials: Number of optimization trials (default: 100)
        use_feature_engineering: Use statistical features vs raw flattening
        feature_method: 'statistical' or 'raw'
        use_pca: Whether to use PCA for dimensionality reduction
        timeout: Time limit in seconds (None for no limit)
        metric: Metric to optimize ('accuracy' or 'f1'). Default: 'f1'

    Returns:
        Dictionary containing:
            - best_model: Trained model with best parameters
            - best_params: Best hyperparameters found
            - predictions: Test set predictions
            - metrics: Performance metrics on test set
            - test_data: Test data tuple (X_test, y_test)
            - optimization_results: Full optimization results from Optuna
    """
    from src.utils import set_seed
    import time
    
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
    y_train_continuous = train_data[1]
    X_val = val_data[0]
    y_val_continuous = val_data[1]
    X_test = test_data[0]
    y_test_continuous = test_data[1]

    # Convert continuous MQI values to class labels
    y_train = mqi_to_class_labels(y_train_continuous)
    y_val = mqi_to_class_labels(y_val_continuous)
    y_test = mqi_to_class_labels(y_test_continuous)

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
        metric=metric,
        feature_type=feature_type,
        plot_suffix=plot_suffix
    )

    best_model = optimization_results['best_model']
    best_params = optimization_results['best_params']

    # Evaluate on test set
    print("\n" + "="*80)
    print("EVALUATING OPTIMIZED MODEL ON TEST SET")
    print("="*80)
    accuracy_train, precision_train, recall_train, f1_train, predictions_train, cm_train = evaluate_xgboost_model(best_model, X_train, y_train)
    accuracy_val, precision_val, recall_val, f1_val, predictions_val, cm_val = evaluate_xgboost_model(best_model, X_val, y_val)
    accuracy_test, precision_test, recall_test, f1_test, predictions_test, cm_test = evaluate_xgboost_model(best_model, X_test, y_test)

    total_time = time.time() - start_time

    # Prepare metrics dictionary
    metric_name = 'accuracy' if metric == 'accuracy' else 'f1'
    metrics = {
        'accuracy': accuracy_test,
        'precision': precision_test,
        'recall': recall_test,
        'f1': f1_test,
        'optimization_time': total_time,
        f'best_validation_{metric_name}': optimization_results['best_score'],
        'n_trials': n_trials,
        'confusion_matrix': cm_test
    }

    file_name = f"metrics{plot_suffix}.txt"
    with open(file_name, "w") as f:
        f.write(metrics.__str__())

    print(f"\n" + "="*80)
    print("OPTIMIZATION SUMMARY")
    print("="*80)
    print(f"Total optimization time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Number of trials: {n_trials}")
    metric_display = 'Accuracy' if metric == 'accuracy' else 'F1-Score'
    print(f"Best validation {metric_display}: {optimization_results['best_score']:.6f}")
    print(f"Test Accuracy: {accuracy_test:.6f}")
    print(f"Test F1-Score: {f1_test:.6f}")
    print("="*80)

    best_model.model.save_model(os.path.join(RESULTS_PATH, f'xgb_class{plot_suffix}.json'))
    
    # Generate prediction analysis plots
    plot_xgboost_prediction_analysis(predictions_train, y_train, cm_train,
                                     save_path=os.path.join(RESULTS_PATH, f'opt_class_train{plot_suffix}.png'))
    plot_xgboost_prediction_analysis(predictions_val, y_val, cm_val,
                                     save_path=os.path.join(RESULTS_PATH, f'opt_class_val{plot_suffix}.png'))
    plot_xgboost_prediction_analysis(predictions_test, y_test, cm_test,
                                     save_path=os.path.join(RESULTS_PATH, f'opt_class_test{plot_suffix}.png'))
    return {
        'best_model': best_model,
        'best_params': best_params,
        'predictions': predictions_test,
        'metrics': metrics,
        'test_data': (X_test, y_test),
        'optimization_results': optimization_results
    }
