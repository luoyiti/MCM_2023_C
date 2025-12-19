# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py

"""
MoE (Mixture of Experts) 模型定义

包含 SparseDispatcher、MLP 专家网络和 MoE 主模型。
"""

import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class SparseDispatcher(object):
    """
    用于实现 Mixture of Experts 的辅助类。
    
    该类的目的是为专家创建输入 minibatch，并将专家的结果组合成统一的输出张量。
    
    主要功能:
    - dispatch: 接收输入张量，为每个专家创建对应的输入张量
    - combine: 将各专家的输出张量加权组合成最终输出
    
    该类用 "gates" 张量初始化，指定哪些 batch 元素去哪个专家，
    以及组合输出时使用的权重。
    """

    def __init__(self, num_experts, gates):
        """创建 SparseDispatcher"""
        self._gates = gates
        self._num_experts = num_experts
        # 排序专家
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # 删除索引
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # 获取每个专家对应的 batch 索引
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # 计算每个专家获得的样本数
        self._part_sizes = (gates > 0).sum(0).tolist()
        # 扩展 gates 以匹配 self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """
        为每个专家创建输入张量。
        
        参数:
            inp: 形状为 [batch_size, <extra_input_dims>] 的张量
        
        返回:
            num_experts 个张量的列表，形状为 [expert_batch_size_i, <extra_input_dims>]
        """
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """
        将专家输出加权求和。
        
        参数:
            expert_out: num_experts 个张量的列表
            multiply_by_gates: 是否乘以门控权重
        
        返回:
            形状为 [batch_size, <extra_output_dims>] 的张量
        """
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(
            self._gates.size(0), 
            expert_out[-1].size(1), 
            requires_grad=True, 
            device=stitched.device
        )
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        """
        返回每个专家对应的门控值。
        
        返回:
            num_experts 个一维张量的列表
        """
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MLP(nn.Module):
    """
    简单的两层 MLP 专家网络。
    
    结构: Linear -> ReLU -> Linear -> Softmax
    """
    
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.soft(out)
        return out


class MoE(nn.Module):
    """
    稀疏门控的 Mixture of Experts 层。
    
    使用单层前馈网络作为专家。
    
    参数:
        input_size: 输入维度
        output_size: 输出维度
        num_experts: 专家数量
        hidden_size: 专家隐藏层大小
        noisy_gating: 是否使用噪声门控
        k: 每个样本使用的专家数量
    """

    def __init__(
        self, 
        input_size, 
        output_size, 
        num_experts, 
        hidden_size, 
        noisy_gating=True, 
        k=4
    ):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        
        # 实例化专家
        self.experts = nn.ModuleList([
            MLP(self.input_size, self.output_size, self.hidden_size) 
            for i in range(self.num_experts)
        ])
        self.w_gate = nn.Parameter(
            torch.zeros(input_size, num_experts), 
            requires_grad=True
        )
        self.w_noise = nn.Parameter(
            torch.zeros(input_size, num_experts), 
            requires_grad=True
        )

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.num_experts

    def cv_squared(self, x):
        """
        计算样本的变异系数平方。
        
        用于鼓励正分布更均匀的损失函数。
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """计算每个专家的真实负载"""
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """NoisyTopKGating 的辅助函数"""
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.k
        )
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """
        噪声 top-k 门控。
        
        参数:
            x: 输入张量 [batch_size, input_size]
            train: 是否在训练模式（训练时添加噪声）
            noise_epsilon: 噪声的小值
        
        返回:
            gates: [batch_size, num_experts]
            load: [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(
            min(self.k + 1, self.num_experts), dim=1
        )
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (
                self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)
            ).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        """
        前向传播。
        
        参数:
            x: 输入张量 [batch_size, input_size]
            loss_coef: 负载均衡损失的系数
        
        返回:
            y: 输出张量 [batch_size, output_size]
            extra_training_loss: 负载均衡损失
        """
        gates, load = self.noisy_top_k_gating(x, self.training)
        importance = gates.sum(0)
        
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [
            self.experts[i](expert_inputs[i]) for i in range(self.num_experts)
        ]
        y = dispatcher.combine(expert_outputs)
        return y, loss
