"""
训练模块

包含 MoE 模型的训练逻辑和评估函数。
"""

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon

from .moe import MoE
from .losses import (
    soft_cross_entropy,
    weighted_soft_cross_entropy,
    expert_diversity_penalty,
    expert_output_diversity_loss,
    expert_js_divergence_loss,
)
from .metrics import compute_metrics
from .config import (
    DEVICE,
    LR,
    WEIGHT_DECAY,
    MAX_EPOCHS,
    PATIENCE,
    NUM_EXPERTS,
    HIDDEN_SIZE,
    TOP_K,
    AUX_COEF,
    EXPERT_DIVERSITY_COEF,
    EXPERT_OUTPUT_DIVERSITY_COEF,
    EXPERT_JS_DIVERGENCE_COEF,
)


def set_seed(seed: int = 42) -> None:
    """设置随机种子，确保实验可复现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def expert_output_separation_js(
    model: MoE, 
    X_data: np.ndarray,
    num_experts: int = None,
    device: str = None,
) -> float:
    """
    用 JS 距离衡量专家"输出分布差异"（越大越不同）。

    做法：
    - 用门控 gates 的 argmax 将样本分配到专家
    - 对每个专家计算其负责样本的"预测均值分布"
    - 计算专家均值分布的 pairwise JS 距离并取平均
    
    注意：这是"可解释指标"，不是训练损失；用于调参/监控。
    
    参数:
        model: MoE 模型
        X_data: 输入数据
        num_experts: 专家数量
        device: 计算设备
    
    返回:
        专家分化度（JS 距离的平均值）
    """
    if num_experts is None:
        num_experts = NUM_EXPERTS
    if device is None:
        device = DEVICE
        
    model.eval()
    Xte = torch.tensor(X_data, device=device)
    with torch.no_grad():
        gates, _ = model.noisy_top_k_gating(Xte, train=False)
        assigned = gates.argmax(dim=1).cpu().numpy()
        P_pred, _ = model(Xte)
        P_pred = P_pred.cpu().numpy()

    means = []
    for e in range(num_experts):
        mask = assigned == e
        if mask.sum() == 0:
            continue
        means.append(P_pred[mask].mean(axis=0))

    if len(means) <= 1:
        return 0.0

    js_vals = []
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            js_vals.append(float(jensenshannon(means[i], means[j])))
    return float(np.mean(js_vals))


def train_moe(
    X_train: np.ndarray,
    P_train: np.ndarray,
    X_val: np.ndarray,
    P_val: np.ndarray,
    Wtr: torch.Tensor = None,
    Wva: torch.Tensor = None,
    num_experts: int = None,
    hidden_size: int = None,
    top_k: int = None,
    aux_coef: float = None,
    expert_diversity_coef: float = None,
    expert_output_diversity_coef: float = None,
    expert_js_divergence_coef: float = None,
    max_epochs: int = None,
    patience: int = None,
    lr: float = None,
    weight_decay: float = None,
    device: str = None,
    verbose: bool = True,
) -> tuple:
    """
    训练 MoE 模型。

    参数:
        X_train, P_train: 训练数据
        X_val, P_val: 验证数据
        Wtr, Wva: 样本权重（可选）
        num_experts: 专家数量
        hidden_size: 隐藏层大小
        top_k: 每个样本使用的专家数
        aux_coef: 辅助损失系数
        expert_diversity_coef: 专家参数分化正则项系数
        expert_output_diversity_coef: 专家输出相似度惩罚系数（核心差异化）
        expert_js_divergence_coef: 专家JS散度鼓励系数（核心差异化）
        max_epochs: 最大训练轮次
        patience: 早停耐心值
        lr: 学习率
        weight_decay: 权重衰减
        device: 计算设备
        verbose: 是否打印训练过程
    
    返回:
        (model, info) 其中 info 包含训练历史和统计信息
    """
    # 使用默认值或传入值
    if num_experts is None:
        num_experts = NUM_EXPERTS
    if hidden_size is None:
        hidden_size = HIDDEN_SIZE
    if top_k is None:
        top_k = TOP_K
    if aux_coef is None:
        aux_coef = AUX_COEF
    if expert_diversity_coef is None:
        expert_diversity_coef = EXPERT_DIVERSITY_COEF
    if expert_output_diversity_coef is None:
        expert_output_diversity_coef = EXPERT_OUTPUT_DIVERSITY_COEF
    if expert_js_divergence_coef is None:
        expert_js_divergence_coef = EXPERT_JS_DIVERGENCE_COEF
    if max_epochs is None:
        max_epochs = MAX_EPOCHS
    if patience is None:
        patience = PATIENCE
    if lr is None:
        lr = LR
    if weight_decay is None:
        weight_decay = WEIGHT_DECAY
    if device is None:
        device = DEVICE

    model = MoE(
        input_size=X_train.shape[1],
        output_size=7,
        num_experts=num_experts,
        hidden_size=hidden_size,
        noisy_gating=True,
        k=top_k,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xtr = torch.tensor(X_train, device=device)
    Ptr = torch.tensor(P_train, device=device)
    Xva = torch.tensor(X_val, device=device)
    Pva = torch.tensor(P_val, device=device)

    best_state = None
    best_val_loss = float("inf")
    bad = 0
    train_losses: list = []
    val_losses: list = []
    aux_losses: list = []
    output_div_losses: list = []  # 跟踪输出差异化损失

    for epoch in range(1, max_epochs + 1):
        model.train()
        p_hat, aux_loss = model(Xtr)

        if Wtr is None:
            loss_main = soft_cross_entropy(p_hat, Ptr)
        else:
            loss_main = weighted_soft_cross_entropy(p_hat, Ptr, Wtr)

        # 参数差异化惩罚（原有）
        div_pen = expert_diversity_penalty(model, device)
        
        # 【核心】输出差异化损失：直接惩罚专家输出相似度
        output_div_loss = expert_output_diversity_loss(model, Xtr, device)
        js_div_loss = expert_js_divergence_loss(model, Xtr, device)
        
        # 组合损失：
        # - loss_main: 主任务损失（软交叉熵）
        # - aux_loss: 负载平衡损失
        # - div_pen: 参数差异化惩罚
        # - output_div_loss: 输出余弦相似度惩罚（最小化相似度）
        # - js_div_loss: 负JS散度（最小化负值 = 最大化JS散度 = 最大化输出差异）
        loss = (
            loss_main 
            + aux_coef * aux_loss 
            + expert_diversity_coef * div_pen
            + expert_output_diversity_coef * output_div_loss
            + expert_js_divergence_coef * js_div_loss
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            p_val, aux_val = model(Xva)
            if Wva is None:
                val_main = soft_cross_entropy(p_val, Pva)
            else:
                val_main = weighted_soft_cross_entropy(p_val, Pva, Wva)
            val_loss = val_main + aux_coef * aux_val

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        aux_losses.append(aux_loss.item())
        output_div_losses.append(output_div_loss.item())

        if val_loss.item() < best_val_loss - 1e-6:
            best_val_loss = val_loss.item()
            best_state = {
                k: v.detach().cpu().clone() 
                for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1

        if verbose and epoch % 50 == 0:
            print(
                f"[MoE] epoch={epoch:3d} train_loss={loss.item():.4f} "
                f"val_loss={val_loss.item():.4f} aux_loss={aux_loss.item():.6f} "
                f"div_pen={div_pen.item():.6f} output_div={output_div_loss.item():.4f}"
            )
            # 额外打印专家分化指标
            try:
                js_sep = expert_output_separation_js(model, X_val, num_experts, device)
                print(f"[MoE] epoch={epoch:3d} expert_js_separation={js_sep:.4f}")
            except Exception as e:
                print(f"[MoE] expert_js_separation 计算失败: {e}")

        if bad >= patience:
            if verbose:
                print(f"[MoE] Early stopping at epoch {epoch}.")
            break

    if best_state:
        model.load_state_dict(best_state)

    info = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "aux_losses": aux_losses,
        "output_div_losses": output_div_losses,
        "best_epoch": len(train_losses) - bad,
        "best_val_loss": best_val_loss,
        "bad": bad,
        "config": {
            "num_experts": num_experts,
            "hidden_size": hidden_size,
            "top_k": top_k,
            "aux_coef": aux_coef,
            "expert_diversity_coef": expert_diversity_coef,
            "expert_output_diversity_coef": expert_output_diversity_coef,
            "expert_js_divergence_coef": expert_js_divergence_coef,
        },
    }
    return model, info


def evaluate(
    model: MoE, 
    X_test: np.ndarray, 
    P_test: np.ndarray,
    device: str = None,
    verbose: bool = True,
) -> tuple:
    """
    在测试集上评估模型。
    
    参数:
        model: 训练好的 MoE 模型
        X_test: 测试特征
        P_test: 测试标签
        device: 计算设备
        verbose: 是否打印结果
    
    返回:
        (P_pred, metrics) 预测结果和性能指标
    """
    if device is None:
        device = DEVICE
        
    model.eval()
    Xte = torch.tensor(X_test, device=device)
    with torch.no_grad():
        P_pred, _ = model(Xte)
        P_pred = P_pred.cpu().numpy()

    metrics = compute_metrics(P_pred, P_test)

    if verbose:
        print("\n[MoE] 测试集评估")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")

    return P_pred, metrics
