"""
损失函数模块

包含软标签交叉熵、加权交叉熵和专家分化正则项。
"""

import torch
import torch.nn.functional as F
from .moe import MoE
from .config import DEVICE


def soft_cross_entropy(
    p_hat: torch.Tensor, 
    p_true: torch.Tensor, 
    eps: float = 1e-12
) -> torch.Tensor:
    """
    软标签交叉熵：-Σ p_true * log(p_hat)，对 batch 取平均。
    
    参数:
        p_hat: 预测分布 [batch_size, num_classes]
        p_true: 真实分布 [batch_size, num_classes]
        eps: 数值稳定性的小常数
    
    返回:
        标量损失值
    """
    p_hat = torch.clamp(p_hat, eps, 1.0)
    return -(p_true * torch.log(p_hat)).sum(dim=1).mean()


def weighted_soft_cross_entropy(
    p_hat: torch.Tensor,
    p_true: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    加权软标签交叉熵：对每个样本的 CE 乘 w 后求平均。
    
    参数:
        p_hat: 预测分布 [batch_size, num_classes]
        p_true: 真实分布 [batch_size, num_classes]
        w: 样本权重 [batch_size]
        eps: 数值稳定性的小常数
    
    返回:
        标量损失值
    """
    p_hat = torch.clamp(p_hat, eps, 1.0)
    per_sample = -(p_true * torch.log(p_hat)).sum(dim=1)
    return (w * per_sample).mean()


def expert_diversity_penalty(model: MoE, device: str = None) -> torch.Tensor:
    """
    鼓励不同专家参数差异：平均 pairwise cosine^2（越小越"分化"）。
    
    思路：惩罚不同专家参数向量的相似度（用 cosine^2），鼓励正交化/分化。
    
    参数:
        model: MoE 模型
        device: 计算设备
    
    返回:
        分化惩罚项（标量）
    """
    if device is None:
        device = DEVICE
        
    if getattr(model, "experts", None) is None:
        return torch.tensor(0.0, device=device)

    experts = list(model.experts)
    if len(experts) <= 1:
        return torch.tensor(0.0, device=device)

    vecs = []
    for exp in experts:
        # 只用权重（不含 bias）来避免尺度/偏置噪声
        params = []
        for name, p in exp.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith("bias"):
                continue
            params.append(p.reshape(-1))
        if not params:
            continue
        vecs.append(torch.cat(params, dim=0))

    if len(vecs) <= 1:
        return torch.tensor(0.0, device=device)

    pen = torch.tensor(0.0, device=vecs[0].device)
    cnt = 0
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            cos = F.cosine_similarity(vecs[i], vecs[j], dim=0)
            pen = pen + cos * cos
            cnt += 1
    return pen / max(1, cnt)


def expert_output_diversity_loss(
    model: MoE,
    x: torch.Tensor,
    device: str = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    专家输出差异化损失（核心差异化机制）。
    
    原理：直接计算各专家在相同输入上的输出分布，惩罚它们的相似度。
    这是比参数差异化更直接、更有效的方法。
    
    做法：
    1. 对每个样本，让所有专家都计算输出（不经过门控）
    2. 计算专家输出之间的pairwise余弦相似度
    3. 返回平均相似度作为损失（需要最小化）
    
    参数:
        model: MoE 模型
        x: 输入张量 [batch_size, input_size]
        device: 计算设备
        eps: 数值稳定性
    
    返回:
        专家输出相似度损失（越小表示专家越不同）
    """
    if device is None:
        device = DEVICE
    
    if getattr(model, "experts", None) is None:
        return torch.tensor(0.0, device=device)
    
    experts = list(model.experts)
    num_experts = len(experts)
    
    if num_experts <= 1:
        return torch.tensor(0.0, device=device)
    
    # 计算每个专家在所有样本上的输出
    # expert_outputs: [num_experts, batch_size, output_size]
    expert_outputs = []
    for expert in experts:
        out = expert(x)  # [batch_size, output_size]
        expert_outputs.append(out)
    expert_outputs = torch.stack(expert_outputs, dim=0)  # [num_experts, batch_size, output_size]
    
    # 计算每个专家的平均输出分布
    # expert_means: [num_experts, output_size]
    expert_means = expert_outputs.mean(dim=1)
    
    # 计算 pairwise 余弦相似度并取平均
    similarity_sum = torch.tensor(0.0, device=device)
    count = 0
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            # 余弦相似度
            cos_sim = F.cosine_similarity(expert_means[i:i+1], expert_means[j:j+1], dim=1)
            # 惩罚相似度（相似度越高，损失越大）
            similarity_sum = similarity_sum + cos_sim.abs()
            count += 1
    
    if count == 0:
        return torch.tensor(0.0, device=device)
    
    return similarity_sum / count


def expert_js_divergence_loss(
    model: MoE,
    x: torch.Tensor,
    device: str = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    专家 JS 散度差异化损失（另一种差异化机制）。
    
    原理：使用 JS 散度衡量专家输出分布的差异，并鼓励差异最大化。
    
    做法：
    1. 计算每个专家在所有样本上的平均输出分布
    2. 计算专家之间的 pairwise JS 散度
    3. 返回负的平均JS散度（需要最大化JS散度，即最小化负值）
    
    参数:
        model: MoE 模型
        x: 输入张量 [batch_size, input_size]
        device: 计算设备
        eps: 数值稳定性
    
    返回:
        负的专家JS散度（最小化此损失等于最大化专家差异）
    """
    if device is None:
        device = DEVICE
    
    if getattr(model, "experts", None) is None:
        return torch.tensor(0.0, device=device)
    
    experts = list(model.experts)
    num_experts = len(experts)
    
    if num_experts <= 1:
        return torch.tensor(0.0, device=device)
    
    # 计算每个专家在所有样本上的输出
    expert_outputs = []
    for expert in experts:
        out = expert(x)  # [batch_size, output_size]
        expert_outputs.append(out)
    expert_outputs = torch.stack(expert_outputs, dim=0)  # [num_experts, batch_size, output_size]
    
    # 计算每个专家的平均输出分布
    expert_means = expert_outputs.mean(dim=1)  # [num_experts, output_size]
    expert_means = torch.clamp(expert_means, eps, 1.0)
    
    # 计算 pairwise JS 散度
    js_sum = torch.tensor(0.0, device=device)
    count = 0
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            p = expert_means[i]
            q = expert_means[j]
            m = 0.5 * (p + q)
            
            # KL(p || m) + KL(q || m)
            kl_pm = (p * (torch.log(p + eps) - torch.log(m + eps))).sum()
            kl_qm = (q * (torch.log(q + eps) - torch.log(m + eps))).sum()
            js = 0.5 * (kl_pm + kl_qm)
            
            js_sum = js_sum + js
            count += 1
    
    if count == 0:
        return torch.tensor(0.0, device=device)
    
    # 返回负的 JS 散度（最小化此损失等于最大化差异）
    return -js_sum / count
