import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Heiti TC'  # 替换为你选择的字体

# 0) 读数据
df = pd.read_csv("data/mcm_processed_data.csv")  # 改成你的文件名

# 1) 特征列（按你现有那 9 个）
feature_cols = [
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
    'rl_4_try_little_training', 'rl_5_try_little_training', 'rl_6_try_little_training', 'rl_7_try_little_training'
]

# 2) 7维分布列（你给的列名）
dist_cols = [
    "1_try", "2_tries", "3_tries", "4_tries",
    "5_tries", "6_tries", "7_or_more_tries_x"
]

# 3) 取出 X 和 P
X = df[feature_cols].copy()
P = df[dist_cols].copy()

# 3.1) 缺失值处理
X = X.fillna(X.median(numeric_only=True))
P = P.fillna(0.0)

# 3.2) 百分比转比例（若你的值是 0-100）
if P.to_numpy().max() > 1.5:
    P = P / 100.0

# 3.3) 防止舍入误差导致和不为1：重新归一化
P = P.clip(lower=0.0)
row_sum = P.sum(axis=1).replace(0, np.nan)
P = P.div(row_sum, axis=0).fillna(1.0 / len(dist_cols))

X_np = X.to_numpy().astype(np.float32)
P_np = P.to_numpy().astype(np.float32)

# 4) 划分 train/val/test（70/15/15）
X_train, X_tmp, P_train, P_tmp = train_test_split(
    X_np, P_np, test_size=0.3, random_state=42
)
X_val, X_test, P_val, P_test = train_test_split(
    X_tmp, P_tmp, test_size=0.5, random_state=42
)

# 5) 标准化 X（强烈推荐）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val   = scaler.transform(X_val).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

# 6) Softmax 回归模型（线性层 + softmax）
class SoftmaxRegression(nn.Module):
    def __init__(self, d_in, n_out=7):
        super().__init__()
        self.linear = nn.Linear(d_in, n_out)

    def forward(self, x):
        logits = self.linear(x)
        return F.softmax(logits, dim=1)

def soft_cross_entropy(p_hat, p_true, eps=1e-12):
    p_hat = torch.clamp(p_hat, eps, 1.0)
    return -(p_true * torch.log(p_hat)).sum(dim=1).mean()

device = "cuda" if torch.cuda.is_available() else "cpu"

Xtr = torch.tensor(X_train, device=device)
Ptr = torch.tensor(P_train, device=device)
Xva = torch.tensor(X_val, device=device)
Pva = torch.tensor(P_val, device=device)

model = SoftmaxRegression(d_in=X_train.shape[1], n_out=7).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

best_state = None
best_val = float("inf")
patience, bad = 30, 0

for epoch in range(1, 501):
    model.train()
    p_hat = model(Xtr)
    loss = soft_cross_entropy(p_hat, Ptr)

    opt.zero_grad()
    loss.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
        p_val = model(Xva)
        val_loss = soft_cross_entropy(p_val, Pva).item()

    if val_loss < best_val - 1e-6:
        best_val = val_loss
        best_state = copy.deepcopy(model.state_dict())
        bad = 0
    else:
        bad += 1

    if epoch % 50 == 0:
        print(f"epoch={epoch:3d} train_loss={loss.item():.4f} val_loss={val_loss:.4f}")

    if bad >= patience:
        print("Early stopping.")
        break

model.load_state_dict(best_state)

# 7) 测试集预测 + 简单评估
Xte = torch.tensor(X_test, device=device)
Pte = torch.tensor(P_test, device=device)

model.eval()
with torch.no_grad():
    P_pred = model(Xte).cpu().numpy()

mae = np.mean(np.abs(P_pred - P_test))
print("Test mean absolute error over 7 bins:", mae)

# （可选）也可以算 KL(p||q) 作为分布误差
eps = 1e-12
kl = np.mean(np.sum(P_test * (np.log(P_test + eps) - np.log(P_pred + eps)), axis=1))
print("Test mean KL(p||q):", kl)

# 8) 输出预测分布
pred_df = pd.DataFrame(P_pred, columns=[f"pred_{c}" for c in dist_cols])
pred_df.to_csv("softmax_pred_output.csv", index=False)
print("Saved: softmax_pred_output.csv")

# 9) 绘制比较图
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 9.1) 平均分布对比（柱状图）
ax = axes[0, 0]
avg_true = P_test.mean(axis=0)
avg_pred = P_pred.mean(axis=0)
x_pos = np.arange(len(dist_cols))
width = 0.35
ax.bar(x_pos - width/2, avg_true, width, label='True', alpha=0.8)
ax.bar(x_pos + width/2, avg_pred, width, label='Predicted', alpha=0.8)
ax.set_xlabel('Try Bin')
ax.set_ylabel('Mean Probability')
ax.set_title('Average Distribution: True vs Predicted')
ax.set_xticks(x_pos)
ax.set_xticklabels(dist_cols, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 9.2) 各bin的绝对误差（柱状图）
ax = axes[0, 1]
abs_errors = np.abs(P_pred - P_test)
mean_abs_errors = abs_errors.mean(axis=0)
ax.bar(x_pos, mean_abs_errors, alpha=0.8, color='coral')
ax.set_xlabel('Try Bin')
ax.set_ylabel('Mean Absolute Error')
ax.set_title('MAE by Bin')
ax.set_xticks(x_pos)
ax.set_xticklabels(dist_cols, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

# 9.3) 样本预测 vs 真实对比（选前20个样本）
ax = axes[1, 0]
n_samples_show = 20
x_samples = np.arange(n_samples_show)
for i, col in enumerate(dist_cols):
    ax.plot(x_samples, P_test[:n_samples_show, i], 'o-', label=f'True_{col}', alpha=0.6)
    ax.plot(x_samples, P_pred[:n_samples_show, i], 's--', label=f'Pred_{col}', alpha=0.6)
ax.set_xlabel('Sample Index')
ax.set_ylabel('Probability')
ax.set_title(f'First {n_samples_show} Samples: True vs Predicted')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(alpha=0.3)

# 9.4) 整体MAE分布（直方图）
ax = axes[1, 1]
sample_mae = np.mean(np.abs(P_pred - P_test), axis=1)
ax.hist(sample_mae, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax.axvline(sample_mae.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={sample_mae.mean():.4f}')
ax.set_xlabel('Sample-wise MAE')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Sample-wise MAE')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('softmax_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: softmax_comparison.png")
plt.show()

