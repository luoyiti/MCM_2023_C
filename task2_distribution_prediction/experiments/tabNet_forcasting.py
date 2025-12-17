"""
TabNet Regression Model for Wordle Difficulty Prediction
Predicts autoencoder_value using various word features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from pytorch_tabnet.tab_model import TabNetRegressor

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# 1. Load and Prepare Data
# ============================================================================

def load_data(file_path: str) -> pd.DataFrame:
    """Load and return the dataset."""
    df = pd.read_csv(file_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare feature matrix X and target vector y.
    Select comprehensive features for predicting autoencoder_value based on feature engineering.
    """
    # Define feature columns based on comprehensive feature engineering categories
    feature_cols = [
        # 使用特征 (Basic Usage Features)
        'Zipf-value',             # Zipf频率值
        
        # 字母组合特征 (Letter Combination Features)
        'letter_entropy',        # 字母熵
        
        # 反馈特征 (Feedback Features)
        'feedback_entropy',      # 反馈熵
        
        # 音节特征 (Syllable Features)
        # 'num_consonants',        # 辅音字母数量
        'max_consecutive_vowels',      # 最大连续元音数
        # 'max_consecutive_consonants',  # 最大连续辅音数
        # 'starts_with_vowel',      # 是否以元音开头
        # 'ends_with_vowel',         # 是否以元音结尾
        
        # 字母频率特征 (Letter Frequency Features)
        'letter_freq_mean',      # 字母平均频率
        # 'letter_freq_min',        # 字母最小频率
        # 'letter_commonness',      # 字母常见度
        'scrabble_score',        # Scrabble分数
        'has_common_suffix',     # 是否有常见后缀
        # 'has_common_prefix',      # 是否有常见前缀
        'num_rare_letters',        # 罕见字母数量
        
        # 位置特征 (Position Features)
        'position_rarity',        # 位置稀有度
        # 'positional_freq_mean',   # 位置频率均值
        'positional_freq_min',    # 位置频率最小值
        # 'positional_fit',         # 位置拟合度
        # 'position_self_entropy',            # 位置自熵
        # 'position_self_entropy_2_letters',  # 两字母位置熵
        'hamming_neighbors',       # 汉明邻居数量
        
        # 键盘距离特征 (Keyboard Distance Features)
        'keyboard_distance',       # 键盘输入距离
        
        # 语义特征 (Semantic Features)
        # 'semantic_neighbors_count',    # 语义邻居数量
        # 'semantic_density',            # 语义密度
        'semantic_distance',           # 语义距离（余弦相似度）
        # 'semantic_distance_to_center',  # 到语义中心的距离
        
        # 仿真模拟特征 - 随机策略 (Simulation Features - Random Strategy)
        # 'mean_simulate_random',   # 随机策略平均尝试次数
        '1_try_simulate_random',  # 随机策略1次尝试概率
        '2_try_simulate_random',  # 随机策略2次尝试概率
        '3_try_simulate_random',  # 随机策略3次尝试概率
        '4_try_simulate_random',  # 随机策略4次尝试概率
        '5_try_simulate_random',  # 随机策略5次尝试概率
        '6_try_simulate_random',  # 随机策略6次尝试概率
        '7_try_simulate_random',  # 随机策略7次尝试概率
        
        # 仿真模拟特征 - 频率策略 (Simulation Features - Frequency Strategy)
        # 'mean_simulate_freq',     # 频率策略平均尝试次数
        '1_try_simulate_freq',    # 频率策略1次尝试概率
        '2_try_simulate_freq',    # 频率策略2次尝试概率
        '3_try_simulate_freq',    # 频率策略3次尝试概率
        '4_try_simulate_freq',    # 频率策略4次尝试概率
        '5_try_simulate_freq',    # 频率策略5次尝试概率
        '6_try_simulate_freq',    # 频率策略6次尝试概率
        '7_try_simulate_freq',    # 频率策略7次尝试概率
        
        # 仿真模拟特征 - 熵策略 (Simulation Features - Entropy Strategy)
        # 'mean_simulate_entropy',  # 熵策略平均尝试次数
        '1_try_simulate_entropy', # 熵策略1次尝试概率
        '2_try_simulate_entropy', # 熵策略2次尝试概率
        '3_try_simulate_entropy', # 熵策略3次尝试概率
        '4_try_simulate_entropy', # 熵策略4次尝试概率
        '5_try_simulate_entropy', # 熵策略5次尝试概率
        '6_try_simulate_entropy', # 熵策略6次尝试概率
        '7_try_simulate_entropy',  # 熵策略7次尝试概率
        
        # 强化学习特征 - 低训练水平 (RL Features - Low Training)
        # 'rl_expected_steps_low_training',   # 低训练水平期望步数
        'rl_1_try_low_training',            # 低训练水平1次尝试概率
        'rl_2_try_low_training',            # 低训练水平2次尝试概率
        'rl_3_try_low_training',            # 低训练水平3次尝试概率
        'rl_4_try_low_training',            # 低训练水平4次尝试概率
        'rl_5_try_low_training',            # 低训练水平5次尝试概率
        'rl_6_try_low_training',            # 低训练水平6次尝试概率
        'rl_7_try_low_training',            # 低训练水平7次尝试概率
        
        # 强化学习特征 - 高训练水平 (RL Features - High Training)
        # 'rl_expected_steps_high_training',   # 高训练水平期望步数
        'rl_1_try_high_training',           # 高训练水平1次尝试概率
        'rl_2_try_high_training',           # 高训练水平2次尝试概率
        'rl_3_try_high_training',           # 高训练水平3次尝试概率
        'rl_4_try_high_training',           # 高训练水平4次尝试概率
        'rl_5_try_high_training',           # 高训练水平5次尝试概率
        'rl_6_try_high_training',           # 高训练水平6次尝试概率
        'rl_7_try_high_training',           # 高训练水平7次尝试概率
        
        # 强化学习特征 - 少量训练水平 (RL Features - Little Training)
        # 'rl_expected_steps_little_training', # 少量训练水平期望步数
        'rl_1_try_little_training',          # 少量训练水平1次尝试概率
        'rl_2_try_little_training',          # 少量训练水平2次尝试概率
        'rl_3_try_little_training',          # 少量训练水平3次尝试概率
        'rl_4_try_little_training',          # 少量训练水平4次尝试概率
        'rl_5_try_little_training',          # 少量训练水平5次尝试概率
        'rl_6_try_little_training',          # 少量训练水平6次尝试概率
        'rl_7_try_little_training',          # 少量训练水平7次尝试概率
    
    ]
    
    # Filter to only include columns that exist in the dataframe
    available_features = [col for col in feature_cols if col in df.columns]
    missing_features = [col for col in feature_cols if col not in df.columns]
    
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
    
    print(f"Using {len(available_features)} features")
    
    target_col = 'autoencoder_value'
    
    X = df[available_features].copy()
    y = df[target_col].copy()
    
    return X, y, available_features


# ============================================================================
# 2. Data Preprocessing
# ============================================================================

def preprocess_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Preprocess features: handle missing values, scale, and convert to numpy arrays.
    """
    # Fill missing values with median
    X = X.fillna(X.median(numeric_only=True))
    
    # Scale features for better TabNet performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to float32 for TabNet
    X_np = X_scaled.astype(np.float32)
    y_np = y.to_numpy().reshape(-1, 1).astype(np.float32)
    
    print(f"Feature matrix shape: {X_np.shape}")
    print(f"Target vector shape: {y_np.shape}")
    
    return X_np, y_np


# ============================================================================
# 3. TabNet Model Training
# ============================================================================

def create_tabnet_model() -> TabNetRegressor:
    """Create and configure TabNet regressor model optimized for better performance."""
    model = TabNetRegressor(
        n_d=24,                     # Larger dimension for better representation
        n_a=24,                     # Larger attention dimension
        n_steps=4,                  # More decision steps for complex patterns
        gamma=1.5,                  # Feature reuse penalty
        n_independent=2,            # More independent GLU layers
        n_shared=2,                 # More shared GLU layers
        lambda_sparse=1e-4,         # Lower sparsity for more flexibility
        optimizer_params=dict(lr=1e-2, weight_decay=1e-4),
        mask_type="entmax",         # entmax for better feature selection
        seed=RANDOM_STATE,
        verbose=0
    )
    return model


def train_single_split(X_np: np.ndarray, y_np: np.ndarray) -> tuple:
    """
    Train TabNet with single train/validation/test split.
    Returns model, predictions, and metrics.
    """
    print("\n" + "="*60)
    print("Training with Single Train/Validation/Test Split")
    print("="*60)
    
    # Split data: 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_np, y_np, test_size=0.3, random_state=RANDOM_STATE
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_valid.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create and train model
    model = create_tabnet_model()
    
    model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=["rmse"],
        max_epochs=1000,
        patience=100,
        batch_size=min(64, X_train.shape[0]),
        virtual_batch_size=min(32, max(1, X_train.shape[0] // 4)),
        num_workers=0,
        drop_last=False
    )
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nTest Set Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    return model, y_test, y_pred, {'rmse': rmse, 'mae': mae, 'r2': r2}


def train_cross_validation(X_np: np.ndarray, y_np: np.ndarray, n_splits: int = 5) -> tuple:
    """
    Train TabNet with K-Fold cross-validation.
    Returns all predictions, actual values, and per-fold metrics.
    """
    print("\n" + "="*60)
    print(f"Training with {n_splits}-Fold Cross Validation")
    print("="*60)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    rmse_list, mae_list, r2_list = [], [], []
    all_y_test, all_y_pred = [], []
    fold_metrics = []
    feature_importances = []
    
    for fold, (tr_idx, te_idx) in enumerate(kf.split(X_np), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        
        X_tr, X_te = X_np[tr_idx], X_np[te_idx]
        y_tr, y_te = y_np[tr_idx], y_np[te_idx]
        
        # Split training into train/validation for early stopping
        X_tr2, X_va, y_tr2, y_va = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=RANDOM_STATE
        )
        
        # Create and train model
        model = create_tabnet_model()
        
        model.fit(
            X_train=X_tr2, y_train=y_tr2,
            eval_set=[(X_va, y_va)],
            eval_metric=["rmse"],
            max_epochs=1000,
            patience=100,
            batch_size=min(64, X_tr2.shape[0]),
            virtual_batch_size=min(32, max(1, X_tr2.shape[0] // 4)),
            num_workers=0,
            drop_last=False
        )
        
        # Predict
        pred = model.predict(X_te)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_te, pred))
        mae = mean_absolute_error(y_te, pred)
        r2 = r2_score(y_te, pred)
        
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
        
        all_y_test.extend(y_te.flatten())
        all_y_pred.extend(pred.flatten())
        
        fold_metrics.append({'fold': fold, 'rmse': rmse, 'mae': mae, 'r2': r2})
        feature_importances.append(model.feature_importances_)
        
        print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("Cross-Validation Summary")
    print("="*60)
    print(f"RMSE: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")
    print(f"MAE:  {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
    print(f"R²:   {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")
    
    # Average feature importance across folds
    avg_feature_importance = np.mean(feature_importances, axis=0)
    
    cv_summary = {
        'rmse_mean': np.mean(rmse_list),
        'rmse_std': np.std(rmse_list),
        'mae_mean': np.mean(mae_list),
        'mae_std': np.std(mae_list),
        'r2_mean': np.mean(r2_list),
        'r2_std': np.std(r2_list),
    }
    
    return (np.array(all_y_test), np.array(all_y_pred), 
            fold_metrics, avg_feature_importance, cv_summary)


# ============================================================================
# 4. Visualization
# ============================================================================

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                     title: str = "Prediction Results", 
                     save_path: str = None):
    """Plot actual vs predicted values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot: Actual vs Predicted
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual Values', fontsize=12)
    ax1.set_ylabel('Predicted Values', fontsize=12)
    ax1.set_title(f'{title}: Actual vs Predicted', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residual plot
    ax2 = axes[1]
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Values', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.set_title(f'{title}: Residual Plot', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_feature_importance(feature_names: list, importance: np.ndarray, 
                            save_path: str = None):
    """Plot feature importance bar chart."""
    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_importance = importance[sorted_idx]
    
    # Plot top 20 features
    n_features = min(20, len(feature_names))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(n_features)
    ax.barh(y_pos, sorted_importance[:n_features], align='center', color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names[:n_features])
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('TabNet Feature Importance (Top 20)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature importance plot saved to: {save_path}")
    
    plt.show()
    
    return sorted_names, sorted_importance


def plot_cv_fold_metrics(fold_metrics: list, save_path: str = None):
    """Plot metrics across CV folds."""
    folds = [m['fold'] for m in fold_metrics]
    rmse_vals = [m['rmse'] for m in fold_metrics]
    mae_vals = [m['mae'] for m in fold_metrics]
    r2_vals = [m['r2'] for m in fold_metrics]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # RMSE by fold
    axes[0].bar(folds, rmse_vals, color='coral', edgecolor='black')
    axes[0].axhline(y=np.mean(rmse_vals), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(rmse_vals):.4f}')
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('RMSE by Fold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE by fold
    axes[1].bar(folds, mae_vals, color='skyblue', edgecolor='black')
    axes[1].axhline(y=np.mean(mae_vals), color='blue', linestyle='--',
                    label=f'Mean: {np.mean(mae_vals):.4f}')
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('MAE by Fold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # R² by fold
    axes[2].bar(folds, r2_vals, color='lightgreen', edgecolor='black')
    axes[2].axhline(y=np.mean(r2_vals), color='green', linestyle='--',
                    label=f'Mean: {np.mean(r2_vals):.4f}')
    axes[2].set_xlabel('Fold')
    axes[2].set_ylabel('R²')
    axes[2].set_title('R² by Fold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"CV metrics plot saved to: {save_path}")
    
    plt.show()


def plot_prediction_distribution(y_true: np.ndarray, y_pred: np.ndarray,
                                 save_path: str = None):
    """Plot distribution of actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(y_true, bins=30, alpha=0.5, label='Actual', color='blue', edgecolor='black')
    ax.hist(y_pred, bins=30, alpha=0.5, label='Predicted', color='orange', edgecolor='black')
    
    ax.set_xlabel('autoencoder_value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution: Actual vs Predicted Values', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Distribution plot saved to: {save_path}")
    
    plt.show()


# ============================================================================
# 5. Report Generation
# ============================================================================

def generate_report(cv_summary: dict, fold_metrics: list, 
                    feature_names: list, feature_importance: np.ndarray,
                    save_path: str = None) -> str:
    """Generate a comprehensive prediction report."""
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("TabNet Regression Model - Prediction Report")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 70)
    
    report_lines.append("\n## Model Configuration")
    report_lines.append("-" * 40)
    report_lines.append("Model: TabNetRegressor")
    report_lines.append("n_d / n_a: 24 / 24")
    report_lines.append("n_steps: 4")
    report_lines.append("gamma: 1.5")
    report_lines.append("lambda_sparse: 1e-4")
    report_lines.append("mask_type: entmax")
    report_lines.append("Optimizer: Adam (lr=0.01, weight_decay=1e-4)")
    report_lines.append("Max epochs: 1000, Patience: 100")
    
    report_lines.append("\n## Cross-Validation Results (5-Fold)")
    report_lines.append("-" * 40)
    report_lines.append(f"RMSE:  {cv_summary['rmse_mean']:.4f} ± {cv_summary['rmse_std']:.4f}")
    report_lines.append(f"MAE:   {cv_summary['mae_mean']:.4f} ± {cv_summary['mae_std']:.4f}")
    report_lines.append(f"R²:    {cv_summary['r2_mean']:.4f} ± {cv_summary['r2_std']:.4f}")
    
    report_lines.append("\n## Per-Fold Metrics")
    report_lines.append("-" * 40)
    report_lines.append(f"{'Fold':<6} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    for m in fold_metrics:
        report_lines.append(f"{m['fold']:<6} {m['rmse']:<12.4f} {m['mae']:<12.4f} {m['r2']:<12.4f}")
    
    report_lines.append("\n## Top 10 Most Important Features")
    report_lines.append("-" * 40)
    sorted_idx = np.argsort(feature_importance)[::-1]
    for i, idx in enumerate(sorted_idx[:10], 1):
        report_lines.append(f"{i:2}. {feature_names[idx]:<35} {feature_importance[idx]:.4f}")
    
    report_lines.append("\n## Model Interpretation")
    report_lines.append("-" * 40)
    
    # Interpret R² score
    r2_mean = cv_summary['r2_mean']
    if r2_mean >= 0.8:
        interpretation = "Excellent - Model explains most variance in the data"
    elif r2_mean >= 0.6:
        interpretation = "Good - Model captures significant patterns"
    elif r2_mean >= 0.4:
        interpretation = "Moderate - Model has predictive power but room for improvement"
    else:
        interpretation = "Limited - Consider feature engineering or alternative models"
    
    report_lines.append(f"R² Interpretation: {interpretation}")
    report_lines.append(f"Average Prediction Error (MAE): {cv_summary['mae_mean']:.4f}")
    
    report_lines.append("\n" + "=" * 70)
    report_lines.append("End of Report")
    report_lines.append("=" * 70)
    
    report = "\n".join(report_lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {save_path}")
    
    return report


# ============================================================================
# 6. Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("TabNet Regression for Wordle Difficulty Prediction")
    print("Target: autoencoder_value")
    print("="*70)
    
    # Load data
    data_path = "data/mcm_processed_data.csv"
    df = load_data(data_path)
    
    # Prepare features
    X, y, feature_names = prepare_features(df)
    
    # Preprocess
    X_np, y_np = preprocess_data(X, y)
    
    # =========================================
    # Option 1: Single Split Training
    # =========================================
    model, y_test_single, y_pred_single, single_metrics = train_single_split(X_np, y_np)
    
    # Plot single split results
    plot_predictions(
        y_test_single.flatten(), 
        y_pred_single.flatten(),
        title="Single Split",
        save_path="output/tabnet_single_split_predictions.png"
    )
    
    # =========================================
    # Option 2: Cross-Validation Training
    # =========================================
    (y_test_cv, y_pred_cv, fold_metrics, 
     feature_importance, cv_summary) = train_cross_validation(X_np, y_np, n_splits=5)
    
    # Plot CV results
    plot_predictions(
        y_test_cv, 
        y_pred_cv,
        title="5-Fold Cross-Validation",
        save_path="output/tabnet_cv_predictions.png"
    )
    
    plot_cv_fold_metrics(
        fold_metrics,
        save_path="output/tabnet_cv_fold_metrics.png"
    )
    
    plot_feature_importance(
        feature_names, 
        feature_importance,
        save_path="output/tabnet_feature_importance.png"
    )
    
    plot_prediction_distribution(
        y_test_cv,
        y_pred_cv,
        save_path="output/tabnet_prediction_distribution.png"
    )
    
    # Generate and print report
    report = generate_report(
        cv_summary, 
        fold_metrics, 
        feature_names, 
        feature_importance,
        save_path="output/tabnet_prediction_report.txt"
    )
    print("\n" + report)
    
    print("\n" + "="*70)
    print("TabNet Regression Complete!")
    print("="*70)
    
    return model, cv_summary, feature_importance


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    import os
    os.makedirs("output", exist_ok=True)
    
    # Run main
    model, cv_summary, feature_importance = main()
