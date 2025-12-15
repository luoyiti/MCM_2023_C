"""
分布预测模型库
预测Wordle游戏尝试次数的概率分布（7维softmax输出）
支持的模型：Linear-Softmax, MLP-Softmax
"""

import os
import copy
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Heiti TC'  # 替换为你选择的字体

from . import config

warnings.filterwarnings("ignore")

# ============================================================================
# 全局配置
# ============================================================================

# 分布预测特征列
DIST_FEATURE_COLS = [
    'Zipf-value',
    'letter_entropy',
    'feedback_entropy',
    'max_consecutive_vowels',
    'letter_freq_mean',
    'scrabble_score',
    'has_common_suffix',
    'num_rare_letters',
    'position_rarity',
    'positional_freq_min',
    'hamming_neighbors',
    'keyboard_distance',
    'semantic_distance',
    '1_try_simulate_random', '2_try_simulate_random', '3_try_simulate_random',
    '4_try_simulate_random', '5_try_simulate_random', '6_try_simulate_random', '7_try_simulate_random',
    '1_try_simulate_freq', '2_try_simulate_freq', '3_try_simulate_freq',
    '4_try_simulate_freq', '5_try_simulate_freq', '6_try_simulate_freq', '7_try_simulate_freq',
    '1_try_simulate_entropy', '2_try_simulate_entropy', '3_try_simulate_entropy',
    '4_try_simulate_entropy', '5_try_simulate_entropy', '6_try_simulate_entropy', '7_try_simulate_entropy',
    'rl_1_try_low_training', 'rl_2_try_low_training', 'rl_3_try_low_training',
    'rl_4_try_low_training', 'rl_5_try_low_training', 'rl_6_try_low_training', 'rl_7_try_low_training',
    'rl_1_try_high_training', 'rl_2_try_high_training', 'rl_3_try_high_training',
    'rl_4_try_high_training', 'rl_5_try_high_training', 'rl_6_try_high_training', 'rl_7_try_high_training',
    'rl_1_try_little_training', 'rl_2_try_little_training', 'rl_3_try_little_training',
    'rl_4_try_little_training', 'rl_5_try_little_training', 'rl_6_try_little_training', 'rl_7_try_little_training',
]

# 7维分布目标列
DIST_TARGET_COLS = [
    "1_try", "2_tries", "3_tries", "4_tries",
    "5_tries", "6_tries", "7_or_more_tries_x"
]

RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setup_plot_style(font_family: str = 'Heiti TC'):
    """设置绘图样式"""
    plt.rcParams['font.family'] = font_family
    sns.set_style("whitegrid")
    sns.set_palette("husl")


# ============================================================================
# 数据加载与预处理
# ============================================================================

def load_distribution_data(
    data_path: str = None,
    feature_cols: Optional[List[str]] = None,
    target_cols: Optional[List[str]] = None,
    test_size: float = 0.3,
    val_ratio: float = 0.5,
    random_state: int = RANDOM_STATE
) -> Dict[str, np.ndarray]:
    """
    加载并预处理分布预测数据
    
    Parameters:
    -----------
    data_path : str
        数据文件路径
    feature_cols : list, optional
        特征列名
    target_cols : list, optional
        目标列名（7维分布）
    test_size : float
        测试集+验证集占比
    val_ratio : float
        验证集在test_size中的占比
        
    Returns:
    --------
    dict : 包含训练、验证、测试数据的字典
    """
    if feature_cols is None:
        feature_cols = DIST_FEATURE_COLS
    if target_cols is None:
        target_cols = DIST_TARGET_COLS
    
    if data_path is None:
        data_path = config.PROCESSED_DATA
    
    df = pd.read_csv(data_path)
    print(f"加载数据: {len(df)} 样本")
    
    X = df[feature_cols].copy()
    P = df[target_cols].copy()
    
    # 缺失值处理
    X = X.fillna(X.median(numeric_only=True))
    P = P.fillna(0.0)
    
    # 百分比转比例
    if P.to_numpy().max() > 1.5:
        P = P / 100.0
    
    # 归一化确保概率和为1
    P = P.clip(lower=0.0)
    row_sum = P.sum(axis=1).replace(0, np.nan)
    P = P.div(row_sum, axis=0).fillna(1.0 / len(target_cols))
    
    X_np = X.to_numpy().astype(np.float32)
    P_np = P.to_numpy().astype(np.float32)
    
    # 划分数据集
    X_train, X_tmp, P_train, P_tmp = train_test_split(
        X_np, P_np, test_size=test_size, random_state=random_state
    )
    X_val, X_test, P_val, P_test = train_test_split(
        X_tmp, P_tmp, test_size=val_ratio, random_state=random_state
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    
    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    
    return {
        'X_train': X_train, 'P_train': P_train,
        'X_val': X_val, 'P_val': P_val,
        'X_test': X_test, 'P_test': P_test,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'target_cols': target_cols,
    }


# ============================================================================
# 模型定义
# ============================================================================

class LinearSoftmax(nn.Module):
    """线性层 + Softmax（Softmax回归）"""
    def __init__(self, d_in: int, n_out: int = 7):
        super().__init__()
        self.linear = nn.Linear(d_in, n_out)
    
    def forward(self, x):
        logits = self.linear(x)
        return F.softmax(logits, dim=1)


class MLPSoftmax(nn.Module):
    """MLP + Softmax（多层感知机分类器）"""
    def __init__(self, d_in: int, n_out: int = 7, 
                 hidden_dims: List[int] = [128, 64, 32],
                 dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = d_in
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, n_out))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        logits = self.network(x)
        return F.softmax(logits, dim=1)


def soft_cross_entropy(p_hat: torch.Tensor, p_true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """软交叉熵损失函数"""
    p_hat = torch.clamp(p_hat, eps, 1.0)
    return -(p_true * torch.log(p_hat)).sum(dim=1).mean()


# ============================================================================
# 评估指标
# ============================================================================

def calculate_distribution_metrics(P_true: np.ndarray, P_pred: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    """
    计算分布预测评估指标
    
    Parameters:
    -----------
    P_true : np.ndarray
        真实分布 (n_samples, 7)
    P_pred : np.ndarray
        预测分布 (n_samples, 7)
        
    Returns:
    --------
    dict : 包含多种评估指标
    """
    # Mean Absolute Error
    mae = np.mean(np.abs(P_pred - P_true))
    
    # 每个bin的MAE
    mae_per_bin = np.mean(np.abs(P_pred - P_true), axis=0)
    
    # KL散度 KL(P_true || P_pred)
    kl = np.mean(np.sum(P_true * (np.log(P_true + eps) - np.log(P_pred + eps)), axis=1))
    
    # Jensen-Shannon散度
    M = 0.5 * (P_true + P_pred)
    js = 0.5 * np.mean(np.sum(P_true * (np.log(P_true + eps) - np.log(M + eps)), axis=1)) + \
         0.5 * np.mean(np.sum(P_pred * (np.log(P_pred + eps) - np.log(M + eps)), axis=1))
    
    # Total Variation Distance
    tvd = 0.5 * np.mean(np.sum(np.abs(P_pred - P_true), axis=1))
    
    # 余弦相似度
    cos_sim = np.mean(np.sum(P_true * P_pred, axis=1) / 
                      (np.linalg.norm(P_true, axis=1) * np.linalg.norm(P_pred, axis=1) + eps))
    
    # RMSE
    rmse = np.sqrt(np.mean((P_pred - P_true) ** 2))
    
    return {
        'mae': mae,
        'mae_per_bin': mae_per_bin,
        'kl_divergence': kl,
        'js_divergence': js,
        'total_variation': tvd,
        'cosine_similarity': cos_sim,
        'rmse': rmse,
    }


def print_distribution_metrics(metrics: Dict, model_name: str = ""):
    """打印分布预测评估结果"""
    print("\n" + "=" * 60)
    print(f"{model_name} 分布预测评估结果")
    print("=" * 60)
    print(f"  MAE (平均绝对误差):     {metrics['mae']:.6f}")
    print(f"  RMSE (均方根误差):      {metrics['rmse']:.6f}")
    print(f"  KL散度:                 {metrics['kl_divergence']:.6f}")
    print(f"  JS散度:                 {metrics['js_divergence']:.6f}")
    print(f"  Total Variation:        {metrics['total_variation']:.6f}")
    print(f"  余弦相似度:             {metrics['cosine_similarity']:.6f}")


# ============================================================================
# 模型训练
# ============================================================================

def train_distribution_model(
    model: nn.Module,
    X_train: np.ndarray,
    P_train: np.ndarray,
    X_val: np.ndarray,
    P_val: np.ndarray,
    model_name: str = "Model",
    epochs: int = 500,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    patience: int = 30,
    verbose: bool = True
) -> Tuple[nn.Module, Dict]:
    """
    训练分布预测模型
    
    Parameters:
    -----------
    model : nn.Module
        PyTorch模型
    X_train, P_train : np.ndarray
        训练数据
    X_val, P_val : np.ndarray
        验证数据
    model_name : str
        模型名称
    epochs : int
        最大训练轮数
    lr : float
        学习率
    weight_decay : float
        权重衰减
    patience : int
        早停耐心值
        
    Returns:
    --------
    model : 训练好的模型
    history : 训练历史
    """
    device = DEVICE
    model = model.to(device)
    
    Xtr = torch.tensor(X_train, device=device)
    Ptr = torch.tensor(P_train, device=device)
    Xva = torch.tensor(X_val, device=device)
    Pva = torch.tensor(P_val, device=device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_state = None
    best_val_loss = float("inf")
    bad_epochs = 0
    
    train_losses = []
    val_losses = []
    
    if verbose:
        print(f"\n训练 {model_name}...")
        print(f"设备: {device}, 参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(1, epochs + 1):
        # 训练
        model.train()
        p_hat = model(Xtr)
        train_loss = soft_cross_entropy(p_hat, Ptr)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            p_val = model(Xva)
            val_loss = soft_cross_entropy(p_val, Pva).item()
        
        scheduler.step(val_loss)
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss)
        
        # 早停检查
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        
        if verbose and epoch % 50 == 0:
            print(f"  Epoch {epoch:3d}: train_loss={train_loss.item():.4f}, val_loss={val_loss:.4f}")
        
        if bad_epochs >= patience:
            if verbose:
                print(f"  早停于 Epoch {epoch}")
            break
    
    # 加载最优模型
    model.load_state_dict(best_state)
    
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'final_epoch': len(train_losses),
    }
    
    return model, history


def predict_distribution(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """使用模型预测分布"""
    model.eval()
    X_tensor = torch.tensor(X, device=DEVICE)
    with torch.no_grad():
        P_pred = model(X_tensor).cpu().numpy()
    return P_pred


# ============================================================================
# 完整训练与评估流程
# ============================================================================

def train_and_evaluate_distribution_model(
    model_type: str,
    data: Dict,
    output_dir: str = "softmax_results",
    save_plots: bool = True,
    show_plots: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    完整的分布模型训练与评估流程
    
    Parameters:
    -----------
    model_type : str
        模型类型: 'linear' 或 'mlp'
    data : dict
        数据字典（来自load_distribution_data）
    output_dir : str
        输出目录
    save_plots : bool
        是否保存图表
    show_plots : bool
        是否显示图表
        
    Returns:
    --------
    dict : 包含模型、预测、指标等
    """
    os.makedirs(output_dir, exist_ok=True)
    
    X_train = data['X_train']
    P_train = data['P_train']
    X_val = data['X_val']
    P_val = data['P_val']
    X_test = data['X_test']
    P_test = data['P_test']
    target_cols = data['target_cols']
    
    d_in = X_train.shape[1]
    n_out = P_train.shape[1]
    
    # 创建模型
    if model_type.lower() == 'linear':
        model = LinearSoftmax(d_in, n_out)
        model_name = "Linear-Softmax"
    elif model_type.lower() == 'mlp':
        hidden_dims = kwargs.get('hidden_dims', [128, 64, 32])
        dropout = kwargs.get('dropout', 0.2)
        model = MLPSoftmax(d_in, n_out, hidden_dims=hidden_dims, dropout=dropout)
        model_name = "MLP-Softmax"
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 训练参数
    train_kwargs = {
        'epochs': kwargs.get('epochs', 500),
        'lr': kwargs.get('lr', 1e-2),
        'weight_decay': kwargs.get('weight_decay', 1e-4),
        'patience': kwargs.get('patience', 30),
        'verbose': kwargs.get('verbose', True),
    }
    
    # 训练模型
    model, history = train_distribution_model(
        model, X_train, P_train, X_val, P_val,
        model_name=model_name, **train_kwargs
    )
    
    # 预测
    P_train_pred = predict_distribution(model, X_train)
    P_val_pred = predict_distribution(model, X_val)
    P_test_pred = predict_distribution(model, X_test)
    
    # 计算指标
    train_metrics = calculate_distribution_metrics(P_train, P_train_pred)
    val_metrics = calculate_distribution_metrics(P_val, P_val_pred)
    test_metrics = calculate_distribution_metrics(P_test, P_test_pred)
    
    # 打印结果
    print_distribution_metrics(test_metrics, model_name + " (测试集)")
    
    # 可视化
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Heiti TC'  # 替换为你选择的字体
    if save_plots or show_plots:
        plot_distribution_results(
            P_test, P_test_pred, target_cols, test_metrics, history,
            model_name=model_name, output_dir=output_dir,
            save=save_plots, show=show_plots
        )
    
    return {
        'model': model,
        'model_name': model_name,
        'model_type': model_type,
        'history': history,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'P_train_pred': P_train_pred,
        'P_val_pred': P_val_pred,
        'P_test_pred': P_test_pred,
    }


# ============================================================================
# 可视化函数
# ============================================================================

def plot_distribution_results(
    P_true: np.ndarray,
    P_pred: np.ndarray,
    target_cols: List[str],
    metrics: Dict,
    history: Dict,
    model_name: str = "Model",
    output_dir: str = "softmax_results",
    save: bool = True,
    show: bool = True
):
    """绘制分布预测结果可视化"""
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Heiti TC'  # 替换为你选择的字体
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 平均分布对比
    ax = axes[0, 0]
    avg_true = P_true.mean(axis=0)
    avg_pred = P_pred.mean(axis=0)
    x_pos = np.arange(len(target_cols))
    width = 0.35
    ax.bar(x_pos - width/2, avg_true, width, label='True', alpha=0.8, color='steelblue')
    ax.bar(x_pos + width/2, avg_pred, width, label='Predicted', alpha=0.8, color='coral')
    ax.set_xlabel('Try Bin', fontsize=11)
    ax.set_ylabel('Mean Probability', fontsize=11)
    ax.set_title(f'{model_name}: Average Distribution Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(target_cols, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2. 各bin的MAE
    ax = axes[0, 1]
    mae_per_bin = metrics['mae_per_bin']
    bars = ax.bar(x_pos, mae_per_bin, alpha=0.8, color='coral', edgecolor='black')
    ax.set_xlabel('Try Bin', fontsize=11)
    ax.set_ylabel('Mean Absolute Error', fontsize=11)
    ax.set_title(f'{model_name}: MAE by Bin', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(target_cols, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    for bar, mae in zip(bars, mae_per_bin):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{mae:.4f}',
                ha='center', va='bottom', fontsize=8)
    
    # 3. 训练曲线
    ax = axes[0, 2]
    ax.plot(history['train_losses'], label='Train Loss', alpha=0.8, linewidth=1.5)
    ax.plot(history['val_losses'], label='Val Loss', alpha=0.8, linewidth=1.5)
    ax.axhline(history['best_val_loss'], color='r', linestyle='--', 
               label=f"Best Val={history['best_val_loss']:.4f}", linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss (Soft Cross Entropy)', fontsize=11)
    ax.set_title(f'{model_name}: Training Curve', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. 样本级MAE分布
    ax = axes[1, 0]
    sample_mae = np.mean(np.abs(P_pred - P_true), axis=1)
    ax.hist(sample_mae, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(sample_mae.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean={sample_mae.mean():.4f}')
    ax.set_xlabel('Sample-wise MAE', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{model_name}: Distribution of Sample MAE', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 5. 预测vs真实散点图（主要bin）
    ax = axes[1, 1]
    # 选择中间几个bin（2-5次尝试最常见）
    for i, (col, color) in enumerate(zip([1, 2, 3, 4], ['blue', 'green', 'orange', 'red'])):
        ax.scatter(P_true[:, col], P_pred[:, col], alpha=0.4, s=20, 
                   label=target_cols[col], color=color)
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect')
    ax.set_xlabel('True Probability', fontsize=11)
    ax.set_ylabel('Predicted Probability', fontsize=11)
    ax.set_title(f'{model_name}: True vs Predicted (Main Bins)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.05, 0.6)
    ax.set_ylim(-0.05, 0.6)
    
    # 6. 评估指标汇总
    ax = axes[1, 2]
    ax.axis('off')
    metrics_text = f"""
    {model_name} 评估指标汇总
    {'='*40}
    
    MAE (平均绝对误差):     {metrics['mae']:.6f}
    RMSE (均方根误差):      {metrics['rmse']:.6f}
    KL散度:                 {metrics['kl_divergence']:.6f}
    JS散度:                 {metrics['js_divergence']:.6f}
    Total Variation:        {metrics['total_variation']:.6f}
    余弦相似度:             {metrics['cosine_similarity']:.6f}
    
    训练轮数:               {history['final_epoch']}
    最佳验证损失:           {history['best_val_loss']:.6f}
    """
    ax.text(0.1, 0.5, metrics_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save:
        save_path = os.path.join(output_dir, f'{model_name.lower().replace("-", "_")}_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def compare_distribution_models(
    results_list: List[Dict],
    output_dir: str = "softmax_results",
    save: bool = True,
    show: bool = True
):
    """比较多个分布预测模型"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    model_names = [r['model_name'] for r in results_list]
    
    # 1. MAE对比
    ax = axes[0]
    metrics_list = ['mae', 'rmse', 'kl_divergence', 'js_divergence']
    x = np.arange(len(metrics_list))
    width = 0.8 / len(results_list)
    
    for i, result in enumerate(results_list):
        values = [result['test_metrics'][m] for m in metrics_list]
        ax.bar(x + i * width, values, width, label=result['model_name'], alpha=0.8)
    
    ax.set_xlabel('Metric', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Model Comparison: Error Metrics', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * (len(results_list) - 1) / 2)
    ax.set_xticklabels(['MAE', 'RMSE', 'KL', 'JS'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2. 训练曲线对比
    ax = axes[1]
    for result in results_list:
        ax.plot(result['history']['val_losses'], label=result['model_name'], alpha=0.8, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation Loss', fontsize=11)
    ax.set_title('Training Curve Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. 余弦相似度对比
    ax = axes[2]
    cos_sims = [r['test_metrics']['cosine_similarity'] for r in results_list]
    bars = ax.bar(model_names, cos_sims, alpha=0.8, color=['steelblue', 'coral'][:len(results_list)])
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Cosine Similarity', fontsize=11)
    ax.set_title('Model Comparison: Cosine Similarity', fontsize=12, fontweight='bold')
    ax.set_ylim(0.9, 1.0)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, cos_sims):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        save_path = os.path.join(output_dir, 'models_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================================
# 报告生成
# ============================================================================

def generate_distribution_report(
    results_list: List[Dict],
    data: Dict,
    output_path: str = "softmax_results/report.txt"
):
    """生成分布预测模型报告"""
    
    lines = []
    lines.append("=" * 80)
    lines.append("分布预测模型 (Softmax) - 综合评估报告")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    
    lines.append("\n## 一、任务描述")
    lines.append("-" * 40)
    lines.append("预测Wordle游戏玩家尝试次数的概率分布（7维）")
    lines.append(f"目标列: {data['target_cols']}")
    lines.append(f"特征数量: {len(data['feature_cols'])}")
    
    lines.append(f"\n训练集: {len(data['X_train'])} 样本")
    lines.append(f"验证集: {len(data['X_val'])} 样本")
    lines.append(f"测试集: {len(data['X_test'])} 样本")
    
    lines.append("\n## 二、模型对比")
    lines.append("-" * 40)
    
    # 创建对比表
    headers = ["Model", "MAE", "RMSE", "KL散度", "JS散度", "TV距离", "余弦相似度"]
    rows = []
    for r in results_list:
        m = r['test_metrics']
        rows.append([
            r['model_name'],
            f"{m['mae']:.6f}",
            f"{m['rmse']:.6f}",
            f"{m['kl_divergence']:.6f}",
            f"{m['js_divergence']:.6f}",
            f"{m['total_variation']:.6f}",
            f"{m['cosine_similarity']:.6f}",
        ])
    
    # 格式化表格
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    header_line = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    lines.append(header_line)
    lines.append("-" * len(header_line))
    for row in rows:
        lines.append(" | ".join(f"{cell:<{col_widths[i]}}" for i, cell in enumerate(row)))
    
    # 找出最佳模型
    best_result = min(results_list, key=lambda x: x['test_metrics']['mae'])
    lines.append(f"\n最佳模型 (按MAE): {best_result['model_name']}")
    
    lines.append("\n## 三、各模型详细信息")
    lines.append("-" * 40)
    
    for result in results_list:
        lines.append(f"\n### {result['model_name']}")
        lines.append(f"  训练轮数: {result['history']['final_epoch']}")
        lines.append(f"  最佳验证损失: {result['history']['best_val_loss']:.6f}")
        lines.append(f"  模型类型: {result['model_type']}")
        
        lines.append(f"\n  测试集指标:")
        for key in ['mae', 'rmse', 'kl_divergence', 'js_divergence', 'total_variation', 'cosine_similarity']:
            lines.append(f"    {key}: {result['test_metrics'][key]:.6f}")
        
        lines.append(f"\n  各Bin的MAE:")
        for i, (col, mae) in enumerate(zip(data['target_cols'], result['test_metrics']['mae_per_bin'])):
            lines.append(f"    {col}: {mae:.6f}")
    
    lines.append("\n## 四、结论与建议")
    lines.append("-" * 40)
    
    # 比较两个模型
    if len(results_list) >= 2:
        linear_result = next((r for r in results_list if r['model_type'] == 'linear'), None)
        mlp_result = next((r for r in results_list if r['model_type'] == 'mlp'), None)
        
        if linear_result and mlp_result:
            linear_mae = linear_result['test_metrics']['mae']
            mlp_mae = mlp_result['test_metrics']['mae']
            
            if mlp_mae < linear_mae:
                improvement = (linear_mae - mlp_mae) / linear_mae * 100
                lines.append(f"\n1. MLP-Softmax 相比 Linear-Softmax 提升了 {improvement:.2f}% (按MAE)")
                lines.append("   MLP能够捕捉非线性关系，对于复杂分布预测更有效")
            else:
                lines.append("\n1. Linear-Softmax 表现与 MLP-Softmax 相近或更优")
                lines.append("   说明特征与目标之间的关系较为线性")
    
    lines.append("\n2. 分布预测评估指标说明:")
    lines.append("   - MAE: 预测概率与真实概率的平均绝对误差")
    lines.append("   - KL散度: 衡量两个概率分布的差异")
    lines.append("   - JS散度: KL散度的对称版本，更稳定")
    lines.append("   - Total Variation: 分布差异的上界")
    lines.append("   - 余弦相似度: 分布向量的方向相似性")
    
    lines.append("\n3. 建议:")
    lines.append("   - 可尝试更复杂的网络结构或正则化技术")
    lines.append("   - 考虑使用交叉验证评估模型稳定性")
    lines.append("   - 特征工程可能进一步提升性能")
    
    lines.append("\n" + "=" * 80)
    lines.append("报告结束")
    lines.append("=" * 80)
    
    report = "\n".join(lines)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n报告已保存到 {output_path}")
    return report


# ============================================================================
# 便捷函数
# ============================================================================

def run_distribution_comparison(
    data_path: str = "../data/mcm_processed_data.csv",
    output_dir: str = "../softmax_results",
    show_plots: bool = True
) -> Tuple[Dict, List[Dict]]:
    """
    运行完整的分布模型比较实验
    
    Returns:
    --------
    data : 数据字典
    results_list : 所有模型结果列表
    """
    print("=" * 60)
    print("分布预测模型比较实验")
    print("=" * 60)
    
    # 加载数据
    data = load_distribution_data(data_path)
    
    # 训练Linear-Softmax
    linear_results = train_and_evaluate_distribution_model(
        'linear', data, output_dir=output_dir,
        save_plots=True, show_plots=show_plots
    )
    
    # 训练MLP-Softmax
    mlp_results = train_and_evaluate_distribution_model(
        'mlp', data, output_dir=output_dir,
        save_plots=True, show_plots=show_plots,
        hidden_dims=[128, 64, 32], dropout=0.2
    )
    
    results_list = [linear_results, mlp_results]
    
    # 模型对比
    compare_distribution_models(results_list, output_dir=output_dir, show=show_plots)
    
    # 生成报告
    generate_distribution_report(results_list, data, 
                                  output_path=os.path.join(output_dir, "report.txt"))
    
    return data, results_list


if __name__ == "__main__":
    setup_plot_style()
    data, results = run_distribution_comparison(show_plots=True)
