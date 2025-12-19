"""
超参数搜索模块

包含专家分化搜索和网格搜索功能。
"""

import os
import numpy as np
import pandas as pd
import torch

from .train import train_moe, expert_output_separation_js
from .metrics import compute_metrics
from .data import make_weights_from_N
from .config import (
    DEVICE,
    WEIGHT_MODE,
    NUM_EXPERTS,
    HIDDEN_SIZE,
    TOP_K,
    AUX_COEF,
    EXPERT_DIVERSITY_COEF,
    SEARCH_MAX_TRIALS,
    SEARCH_EPOCH_SCALE,
    SEARCH_AUX_COEF_GRID,
    SEARCH_DIVERSITY_COEF_GRID,
    SEARCH_OBJECTIVE_BETA_JS,
    EXPERT_NUM_GRID,
    TOPK_GRID,
    HIDDEN_SIZE_GRID,
    GRID_SEARCH_EPOCH_SCALE,
    MAX_EPOCHS,
)


def specialization_search(
    X_train: np.ndarray,
    P_train: np.ndarray,
    N_train: np.ndarray,
    X_val: np.ndarray,
    P_val: np.ndarray,
    N_val: np.ndarray,
    output_dir: str,
    max_epochs: int = None,
) -> dict:
    """
    轻量调参：在保持 val_loss 不太差的情况下，尽量增大专家分化(JS)。

    搜索维度（小范围）：AUX_COEF 与 EXPERT_DIVERSITY_COEF。
    
    参数:
        X_train, P_train, N_train: 训练数据
        X_val, P_val, N_val: 验证数据
        output_dir: 输出目录
        max_epochs: 每次搜索的最大轮次
    
    返回:
        包含 CSV 路径和最优配置的字典
    """
    if max_epochs is None:
        max_epochs = max(1, int(MAX_EPOCHS * SEARCH_EPOCH_SCALE))
    
    Wtr = (
        torch.tensor(make_weights_from_N(N_train, WEIGHT_MODE), device=DEVICE)
        if N_train is not None
        else None
    )
    Wva = (
        torch.tensor(make_weights_from_N(N_val, WEIGHT_MODE), device=DEVICE)
        if N_val is not None
        else None
    )

    rows = []
    trial = 0
    for aux in SEARCH_AUX_COEF_GRID:
        for div in SEARCH_DIVERSITY_COEF_GRID:
            trial += 1
            if trial > SEARCH_MAX_TRIALS:
                break
            print(f"[Search] trial={trial} aux_coef={aux} div_coef={div}")
            model, info = train_moe(
                X_train=X_train,
                P_train=P_train,
                X_val=X_val,
                P_val=P_val,
                Wtr=Wtr,
                Wva=Wva,
                num_experts=NUM_EXPERTS,
                hidden_size=HIDDEN_SIZE,
                top_k=TOP_K,
                aux_coef=aux,
                expert_diversity_coef=div,
                max_epochs=max_epochs,
                verbose=False,
            )
            val_best = float(info.get("best_val_loss", np.nan))
            js_sep = float(expert_output_separation_js(model, X_val, NUM_EXPERTS))
            # 目标：val_loss 越小越好，JS 越大越好
            obj = val_best - SEARCH_OBJECTIVE_BETA_JS * js_sep
            rows.append({
                "trial": trial,
                "aux_coef": aux,
                "div_coef": div,
                "best_val_loss": val_best,
                "expert_js_sep": js_sep,
                "objective": obj,
            })
        if trial > SEARCH_MAX_TRIALS:
            break

    df = pd.DataFrame(rows).sort_values("objective", ascending=True)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "specialization_search_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Search] 结果已保存: {csv_path}")
    best = df.iloc[0].to_dict() if len(df) else {}
    return {"csv_path": csv_path, "best": best, "all_results": df}


def expert_topk_grid_search(
    X_train: np.ndarray,
    P_train: np.ndarray,
    N_train: np.ndarray,
    X_val: np.ndarray,
    P_val: np.ndarray,
    N_val: np.ndarray,
    X_test: np.ndarray,
    P_test: np.ndarray,
    output_dir: str,
    expert_num_grid: list = None,
    topk_grid: list = None,
    hidden_size_grid: list = None,
    max_epochs: int = None,
) -> dict:
    """
    执行专家数量与 TOP-K 的网格搜索，找到最优超参数组合。
    
    该函数测试不同的专家数量(NUM_EXPERTS)、路由数量(TOP_K)和隐藏层大小(HIDDEN_SIZE)
    的组合，评估每种配置在验证集和测试集上的表现。
    
    参数:
        X_train, P_train, N_train: 训练集数据
        X_val, P_val, N_val: 验证集数据
        X_test, P_test: 测试集数据
        output_dir: 结果输出目录
        expert_num_grid: 专家数量候选列表
        topk_grid: TOP-K 候选列表
        hidden_size_grid: 隐藏层大小候选列表
        max_epochs: 每次搜索的最大轮次
    
    返回:
        包含最优配置和完整搜索结果的字典
    """
    if expert_num_grid is None:
        expert_num_grid = EXPERT_NUM_GRID
    if topk_grid is None:
        topk_grid = TOPK_GRID
    if hidden_size_grid is None:
        hidden_size_grid = HIDDEN_SIZE_GRID
    if max_epochs is None:
        max_epochs = max(1, int(MAX_EPOCHS * GRID_SEARCH_EPOCH_SCALE))
    
    print(f"\n{'='*70}")
    print(f"[网格搜索] 开始专家数量与 TOP-K 超参数搜索")
    print(f"[网格搜索] 专家数量候选: {expert_num_grid}")
    print(f"[网格搜索] TOP-K 候选: {topk_grid}")
    print(f"[网格搜索] 隐藏层大小候选: {hidden_size_grid}")
    print(f"[网格搜索] 训练轮次: {max_epochs}")
    print(f"{'='*70}\n")
    
    # 计算样本权重
    Wtr = (
        torch.tensor(make_weights_from_N(N_train, WEIGHT_MODE), device=DEVICE)
        if N_train is not None
        else None
    )
    Wva = (
        torch.tensor(make_weights_from_N(N_val, WEIGHT_MODE), device=DEVICE)
        if N_val is not None
        else None
    )
    
    results = []
    trial = 0
    total_trials = len(expert_num_grid) * len(topk_grid) * len(hidden_size_grid)
    
    for num_exp in expert_num_grid:
        for topk in topk_grid:
            # 跳过无效配置：TOP-K 不能超过专家数量
            if topk > num_exp:
                print(f"[网格搜索] 跳过无效配置: num_experts={num_exp}, top_k={topk}")
                continue
                
            for hidden in hidden_size_grid:
                trial += 1
                print(f"\n[网格搜索] Trial {trial}/{total_trials}: "
                      f"num_experts={num_exp}, top_k={topk}, hidden={hidden}")
                
                # 训练模型
                model, info = train_moe(
                    X_train=X_train,
                    P_train=P_train,
                    X_val=X_val,
                    P_val=P_val,
                    Wtr=Wtr,
                    Wva=Wva,
                    num_experts=num_exp,
                    hidden_size=hidden,
                    top_k=topk,
                    aux_coef=AUX_COEF,
                    expert_diversity_coef=EXPERT_DIVERSITY_COEF,
                    max_epochs=max_epochs,
                    verbose=False,
                )
                
                # 验证集最佳损失
                val_best = float(info.get("best_val_loss", np.nan))
                
                # 在测试集上评估
                model.eval()
                with torch.no_grad():
                    Xte = torch.tensor(X_test, device=DEVICE)
                    P_pred, _ = model(Xte)
                    P_pred = P_pred.cpu().numpy()
                
                # 计算测试集指标
                metrics = compute_metrics(P_pred, P_test)
                
                # 计算专家分化度（JS 距离）
                js_sep = float(expert_output_separation_js(model, X_val, num_exp))
                
                # 记录结果
                result = {
                    "trial": trial,
                    "num_experts": num_exp,
                    "top_k": topk,
                    "hidden_size": hidden,
                    "best_val_loss": val_best,
                    "best_epoch": info.get("best_epoch"),
                    "expert_js_sep": js_sep,
                    **metrics,
                }
                results.append(result)
                
                print(f"  -> val_loss={val_best:.4f}, mae={metrics['mae']:.4f}, "
                      f"js={metrics['js_mean']:.4f}, r2={metrics['r2']:.4f}")
    
    # 创建结果 DataFrame
    df = pd.DataFrame(results)
    
    # 综合目标：主要优化 MAE（越小越好），同时考虑模型稳定性
    df["objective"] = df["mae"] + 0.1 * df["best_val_loss"] - 0.05 * df["expert_js_sep"]
    df = df.sort_values("objective", ascending=True)
    
    # 保存完整结果
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "expert_topk_grid_search_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[网格搜索] 完整结果已保存: {csv_path}")
    
    # 获取最优配置
    best = df.iloc[0].to_dict() if len(df) > 0 else {}
    
    # 打印最优结果
    print(f"\n{'='*70}")
    print(f"[网格搜索] 最优配置:")
    print(f"  - num_experts: {best.get('num_experts')}")
    print(f"  - top_k: {best.get('top_k')}")
    print(f"  - hidden_size: {best.get('hidden_size')}")
    print(f"  - best_val_loss: {best.get('best_val_loss'):.4f}")
    print(f"  - mae: {best.get('mae'):.4f}")
    print(f"  - r2: {best.get('r2'):.4f}")
    print(f"{'='*70}\n")
    
    return {
        "csv_path": csv_path,
        "best": best,
        "all_results": df,
    }
