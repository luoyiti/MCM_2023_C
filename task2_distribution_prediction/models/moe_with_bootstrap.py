"""\
MoE (Mixture of Experts) + MLP + Softmax + Bootstrap 不确定性估计脚本

本脚本严格复用 Moe_Softmax.py 的核心结构（数据、MoE、loss、训练逻辑），
只做“最小侵入式”的 Bootstrap 改造：

核心改造点：
	把“train + predict”这一整段重复 B 次（每次对训练集做 bootstrap 重采样），
	在同一个测试集上收集 B 份预测分布，进而构造经验分布的：
		- 均值分布 P_mean
		- 标准差 P_std
		- 置信区间 [P_low, P_high]（逐桶/逐维分位数）

目标：
	- 不改 MoE 结构
	- 不改 loss（仍是软标签交叉熵 + aux loss）
	- 不推翻 train/val/test 与 early stopping
	- 只在外面包一层 bootstrap 外循环

输出：
	moe_bootstrap_output/
		- moe_softmax_pred_output.csv              (单模型点预测)
		- moe_bootstrap_pred_summary.csv           (bootstrap: mean/std/CI)
		- moe_bootstrap_report.json                (配置、指标、置信度统计)
"""

import os
import json
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon

# 允许从任意工作目录运行/导入：确保当前脚本目录在 sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
	sys.path.insert(0, _THIS_DIR)

from moe import MoE


# 设置中文字体（仅影响 matplotlib 输出；本脚本默认不画图，但保留一致性）
plt.rcParams["font.family"] = "Heiti TC"
plt.rcParams["axes.unicode_minus"] = False


# ---------------- 全局配置 ----------------
# 获取脚本所在目录的父目录（task2_distribution_prediction），然后定位到 data 目录
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))  # 向上两级到项目根目录
DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "mcm_processed_data.csv")
N_COL = "number_of_reported_results"
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "moe_bootstrap_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- Holdout 配置（新增：eerie 不参与训练/测试） ----------------
HOLDOUT_WORD = "eerie"
# 数据中“单词”列名在不同版本里可能不同，这里做候选匹配
WORD_COL_CANDIDATES = ["word", "Word", "target_word", "answer", "Answer"]


# 与 Moe_Softmax.py 保持一致的特征列
FEATURE_COLS = [
	"Zipf-value",
	"letter_entropy",
	"feedback_entropy",
	"max_consecutive_vowels",
	"letter_freq_mean",
	"scrabble_score",
	"has_common_suffix",
	"num_rare_letters",
	"position_rarity",
	"positional_freq_min",
	"hamming_neighbors",
	"keyboard_distance",
	"semantic_distance",
	"1_try_simulate_random",
	"2_try_simulate_random",
	"3_try_simulate_random",
	"4_try_simulate_random",
	"5_try_simulate_random",
	"6_try_simulate_random",
	"7_try_simulate_random",
	"1_try_simulate_freq",
	"2_try_simulate_freq",
	"3_try_simulate_freq",
	"4_try_simulate_freq",
	"5_try_simulate_freq",
	"6_try_simulate_freq",
	"7_try_simulate_freq",
	"1_try_simulate_entropy",
	"2_try_simulate_entropy",
	"3_try_simulate_entropy",
	"4_try_simulate_entropy",
	"5_try_simulate_entropy",
	"6_try_simulate_entropy",
	"7_try_simulate_entropy",
	"rl_1_try_low_training",
	"rl_2_try_low_training",
	"rl_3_try_low_training",
	"rl_4_try_low_training",
	"rl_5_try_low_training",
	"rl_6_try_low_training",
	"rl_7_try_low_training",
	"rl_1_try_high_training",
	"rl_2_try_high_training",
	"rl_3_try_high_training",
	"rl_4_try_high_training",
	"rl_5_try_high_training",
	"rl_6_try_high_training",
	"rl_7_try_high_training",
	"rl_1_try_little_training",
	"rl_2_try_little_training",
	"rl_3_try_little_training",
	"rl_4_try_little_training",
	"rl_5_try_little_training",
	"rl_6_try_little_training",
	"rl_7_try_little_training",
]


DIST_COLS = [
	"1_try",
	"2_tries",
	"3_tries",
	"4_tries",
	"5_tries",
	"6_tries",
	"7_or_more_tries_x",
]


# ==============================================================================
# 训练超参数配置（保持与 Moe_Softmax.py 一致）
# ==============================================================================
LR = 5e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 500
PATIENCE = 50
WEIGHT_MODE = "sqrt"


# ==============================================================================
# MoE 超参数配置（保持与 Moe_Softmax.py 一致）
# ==============================================================================
NUM_EXPERTS = 3
HIDDEN_SIZE = 64
TOP_K = 2
AUX_COEF = 1e-3

# 让专家“差异更大”的正则项（默认很小；设为 0 可关闭）
# 思路：惩罚不同专家参数向量的相似度（用 cosine^2），鼓励正交化/分化。
EXPERT_DIVERSITY_COEF = 1e-4

# 是否启用“专家分化 vs 性能”轻量调参（默认关闭，避免跑很久）
ENABLE_SPECIALIZATION_SEARCH = True
SEARCH_MAX_TRIALS = 10
SEARCH_EPOCH_SCALE = 0.4
SEARCH_AUX_COEF_GRID = [1e-3, 5e-3, 1e-2]
SEARCH_DIVERSITY_COEF_GRID = [0.0, 1e-5, 1e-4, 5e-4, 1e-3]
SEARCH_OBJECTIVE_BETA_JS = 0.5  # JS 越大越好；目标里减去 beta * JS


# ==============================================================================
# 超参数网格搜索配置（新增：测试专家数量与 TOP-K 对模型性能的影响）
# ==============================================================================
# 是否启用专家数量和 TOP-K 的网格搜索
ENABLE_EXPERT_TOPK_SEARCH = True

# 搜索空间：专家数量候选值
# 可通过环境变量 MOE_EXPERT_GRID 覆盖，格式如 "2,3,4"
EXPERT_NUM_GRID = [2, 3, 4, 6]

# 搜索空间：TOP-K 候选值（注意：TOP-K 不能超过专家数量）
# 可通过环境变量 MOE_TOPK_GRID 覆盖
TOPK_GRID = [1, 2, 3]

# 隐藏层大小候选值
# 可通过环境变量 MOE_HIDDEN_GRID 覆盖
HIDDEN_SIZE_GRID = [32, 64, 128]

# 网格搜索训练轮次缩放（节省时间）
GRID_SEARCH_EPOCH_SCALE = 0.5

# 是否在网格搜索中也进行 Bootstrap 评估（耗时但更可靠）
GRID_SEARCH_WITH_BOOTSTRAP = False

# 网格搜索时的 Bootstrap 次数（若启用）
GRID_SEARCH_BOOTSTRAP_B = 20

# 环境变量覆盖网格搜索参数
try:
	_expert_grid_env = os.getenv("MOE_EXPERT_GRID")
	if _expert_grid_env:
		EXPERT_NUM_GRID = [int(x) for x in _expert_grid_env.split(",")]
except Exception:
	pass

try:
	_topk_grid_env = os.getenv("MOE_TOPK_GRID")
	if _topk_grid_env:
		TOPK_GRID = [int(x) for x in _topk_grid_env.split(",")]
except Exception:
	pass

try:
	_hidden_grid_env = os.getenv("MOE_HIDDEN_GRID")
	if _hidden_grid_env:
		HIDDEN_SIZE_GRID = [int(x) for x in _hidden_grid_env.split(",")]
except Exception:
	pass


# ==============================================================================
# Bootstrap 配置（新增：最小侵入式）
# ==============================================================================
BOOTSTRAP_B = 100
BOOTSTRAP_EPOCH_SCALE = 0.6
BOOTSTRAP_CI_LEVEL = 0.95

# 环境变量快速开关/降本（便于快速验证作图）：
# MOE_ENABLE_SEARCH=0/1 关闭或开启轻量调参；MOE_BOOTSTRAP_B=10 减少bootstrap次数；
# MOE_MAX_EPOCHS=100 覆盖训练轮次
try:
	ENABLE_SPECIALIZATION_SEARCH = bool(int(os.getenv("MOE_ENABLE_SEARCH", str(int(ENABLE_SPECIALIZATION_SEARCH)))))
except Exception:
	pass
try:
	BOOTSTRAP_B = int(os.getenv("MOE_BOOTSTRAP_B", str(BOOTSTRAP_B)))
except Exception:
	pass
try:
	_override_epochs = os.getenv("MOE_MAX_EPOCHS")
	if _override_epochs is not None:
		MAX_EPOCHS = int(_override_epochs)
except Exception:
	pass


def set_seed(seed: int = RANDOM_SEED) -> None:
	"""设置随机种子，确保实验可复现"""
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def make_weights_from_N(N_array: np.ndarray, mode: str = "sqrt") -> np.ndarray:
	"""根据每个样本的参与人数 N 计算样本权重，并做均值归一化（均值=1）。"""
	if mode == "sqrt":
		w = np.sqrt(N_array)
	elif mode == "log1p":
		w = np.log1p(N_array)
	else:
		raise ValueError("mode must be 'sqrt' or 'log1p'")
	w = w / (w.mean() + 1e-12)
	return w.astype(np.float32)


def soft_cross_entropy(p_hat: torch.Tensor, p_true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
	"""软标签交叉熵：-Σ p_true * log(p_hat)，对 batch 取平均。"""
	p_hat = torch.clamp(p_hat, eps, 1.0)
	return -(p_true * torch.log(p_hat)).sum(dim=1).mean()


def weighted_soft_cross_entropy(
	p_hat: torch.Tensor,
	p_true: torch.Tensor,
	w: torch.Tensor,
	eps: float = 1e-12,
) -> torch.Tensor:
	"""加权软标签交叉熵：对每个样本的 CE 乘 w 后求平均。"""
	p_hat = torch.clamp(p_hat, eps, 1.0)
	per_sample = -(p_true * torch.log(p_hat)).sum(dim=1)
	return (w * per_sample).mean()


def expert_diversity_penalty(model: MoE) -> torch.Tensor:
	"""鼓励不同专家参数差异：平均 pairwise cosine^2（越小越“分化”）。"""
	if getattr(model, "experts", None) is None:
		return torch.tensor(0.0, device=DEVICE)

	experts = list(model.experts)
	if len(experts) <= 1:
		return torch.tensor(0.0, device=DEVICE)

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
		return torch.tensor(0.0, device=DEVICE)

	pen = torch.tensor(0.0, device=vecs[0].device)
	cnt = 0
	for i in range(len(vecs)):
		for j in range(i + 1, len(vecs)):
			cos = F.cosine_similarity(vecs[i], vecs[j], dim=0)
			pen = pen + cos * cos
			cnt += 1
	return pen / max(1, cnt)


def expert_output_separation_js(model: MoE, X_data: np.ndarray) -> float:
	"""用 JS 距离衡量专家“输出分布差异”（越大越不同）。

	做法：
	- 用门控 gates 的 argmax 将样本分配到专家
	- 对每个专家计算其负责样本的“预测均值分布”
	- 计算专家均值分布的 pairwise JS 距离并取平均
	
	注意：这是“可解释指标”，不是训练损失；用于调参/监控。
	"""
	model.eval()
	Xte = torch.tensor(X_data, device=DEVICE)
	with torch.no_grad():
		gates, _ = model.noisy_top_k_gating(Xte, train=False)
		assigned = gates.argmax(dim=1).cpu().numpy()
		P_pred, _ = model(Xte)
		P_pred = P_pred.cpu().numpy()

	means = []
	for e in range(NUM_EXPERTS):
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


def load_and_split_data():
	"""加载数据、预处理并划分 train/val/test（70/15/15），并对特征做标准化。

	新增：将 df 中单词列等于 HOLDOUT_WORD（默认 eerie）的样本抽出来，
	不参与训练/验证/测试；最后用于“已训练模型”的预测与不确定性估计。
	"""
	df_raw = pd.read_csv(DATA_PATH)

	# --- 1) 找到 word 列并切分 holdout ---
	word_col = None
	for c in WORD_COL_CANDIDATES:
		if c in df_raw.columns:
			word_col = c
			break

	holdout_pack = None
	if word_col is not None:
		word_series = df_raw[word_col].astype(str)
		mask_holdout = word_series.str.lower().eq(HOLDOUT_WORD.lower())
		if mask_holdout.any():
			df_holdout = df_raw.loc[mask_holdout].copy()
			df = df_raw.loc[~mask_holdout].copy()
			holdout_pack = {
				"word_col": word_col,
				"word": df_holdout[word_col].astype(str).tolist(),
				"df": df_holdout,
			}
			print(f"[Holdout] 已抽取 {mask_holdout.sum()} 条 '{HOLDOUT_WORD}' 样本，不参与 train/val/test")
		else:
			df = df_raw
	else:
		df = df_raw
		print("[Holdout] 未找到 word 列，跳过 holdout")

	# --- 2) 特征与标签预处理（用 df 的统计量，避免泄露 holdout） ---
	X = df[FEATURE_COLS].copy()
	feat_median = X.median(numeric_only=True)
	X = X.fillna(feat_median)

	P = df[DIST_COLS].copy().fillna(0.0)
	if P.to_numpy().max() > 1.5:
		P = P / 100.0
	P = P.clip(lower=0.0)
	row_sum = P.sum(axis=1).replace(0, np.nan)
	P = P.div(row_sum, axis=0).fillna(1.0 / len(DIST_COLS))

	if N_COL is not None and N_COL in df.columns:
		N = df[N_COL].fillna(df[N_COL].median()).clip(lower=1)
		N_np = N.to_numpy().astype(np.float32)
	else:
		N_np = None

	X_np = X.to_numpy().astype(np.float32)
	P_np = P.to_numpy().astype(np.float32)

	if N_np is None:
		X_train, X_tmp, P_train, P_tmp = train_test_split(
			X_np, P_np, test_size=0.3, random_state=RANDOM_SEED
		)
		X_val, X_test, P_val, P_test = train_test_split(
			X_tmp, P_tmp, test_size=0.5, random_state=RANDOM_SEED
		)
		N_train = N_val = N_test = None
	else:
		X_train, X_tmp, P_train, P_tmp, N_train, N_tmp = train_test_split(
			X_np, P_np, N_np, test_size=0.3, random_state=RANDOM_SEED
		)
		X_val, X_test, P_val, P_test, N_val, N_test = train_test_split(
			X_tmp, P_tmp, N_tmp, test_size=0.5, random_state=RANDOM_SEED
		)

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train).astype(np.float32)
	X_val = scaler.transform(X_val).astype(np.float32)
	X_test = scaler.transform(X_test).astype(np.float32)

	# --- 3) Holdout 数据也按同样的预处理与 scaler 变换 ---
	if holdout_pack is not None:
		df_holdout = holdout_pack["df"]
		Xh = df_holdout[FEATURE_COLS].copy().fillna(feat_median)
		Xh_np = scaler.transform(Xh.to_numpy().astype(np.float32)).astype(np.float32)

		Ph = df_holdout[DIST_COLS].copy().fillna(0.0)
		if Ph.to_numpy().max() > 1.5:
			Ph = Ph / 100.0
		Ph = Ph.clip(lower=0.0)
		row_sum_h = Ph.sum(axis=1).replace(0, np.nan)
		Ph = Ph.div(row_sum_h, axis=0).fillna(1.0 / len(DIST_COLS))
		Ph_np = Ph.to_numpy().astype(np.float32)

		holdout_pack = {
			"word_col": holdout_pack["word_col"],
			"word": holdout_pack["word"],
			"X": Xh_np,
			"P_true": Ph_np,
		}

	return X_train, X_val, X_test, P_train, P_val, P_test, N_train, N_val, N_test, holdout_pack


def train_moe(
	X_train: np.ndarray,
	P_train: np.ndarray,
	X_val: np.ndarray,
	P_val: np.ndarray,
	Wtr: torch.Tensor | None,
	Wva: torch.Tensor | None,
) -> tuple:
	return train_moe_with_params(
		X_train=X_train,
		P_train=P_train,
		X_val=X_val,
		P_val=P_val,
		Wtr=Wtr,
		Wva=Wva,
		num_experts=NUM_EXPERTS,
		hidden_size=HIDDEN_SIZE,
		top_k=TOP_K,
		aux_coef=AUX_COEF,
		expert_diversity_coef=EXPERT_DIVERSITY_COEF,
	)


def train_moe_with_params(
	X_train: np.ndarray,
	P_train: np.ndarray,
	X_val: np.ndarray,
	P_val: np.ndarray,
	Wtr: torch.Tensor | None,
	Wva: torch.Tensor | None,
	*,
	num_experts: int,
	hidden_size: int,
	top_k: int,
	aux_coef: float,
	expert_diversity_coef: float,
) -> tuple:
	"""训练 MoE（完全复用 Moe_Softmax.py 的训练逻辑）。"""

	model = MoE(
		input_size=X_train.shape[1],
		output_size=7,
		num_experts=num_experts,
		hidden_size=hidden_size,
		noisy_gating=True,
		k=top_k,
	).to(DEVICE)

	opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

	Xtr = torch.tensor(X_train, device=DEVICE)
	Ptr = torch.tensor(P_train, device=DEVICE)
	Xva = torch.tensor(X_val, device=DEVICE)
	Pva = torch.tensor(P_val, device=DEVICE)

	best_state = None
	best_val_loss = float("inf")
	bad = 0
	train_losses: list[float] = []
	val_losses: list[float] = []
	aux_losses: list[float] = []

	for epoch in range(1, MAX_EPOCHS + 1):
		model.train()
		p_hat, aux_loss = model(Xtr)

		if Wtr is None:
			loss_main = soft_cross_entropy(p_hat, Ptr)
		else:
			loss_main = weighted_soft_cross_entropy(p_hat, Ptr, Wtr)

		div_pen = expert_diversity_penalty(model)
		loss = loss_main + aux_coef * aux_loss + expert_diversity_coef * div_pen

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
			val_loss = val_main + AUX_COEF * aux_val

		train_losses.append(loss.item())
		val_losses.append(val_loss.item())
		aux_losses.append(aux_loss.item())

		if val_loss.item() < best_val_loss - 1e-6:
			best_val_loss = val_loss.item()
			best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
			bad = 0
		else:
			bad += 1

		if epoch % 50 == 0:
			print(
				f"[MoE] epoch={epoch:3d} train_loss={loss.item():.4f} "
				f"val_loss={val_loss.item():.4f} aux_loss={aux_loss.item():.6f} "
				f"div_pen={div_pen.item():.6f}"
			)
			# 额外打印一个“专家分化指标”（JS 越大越分化）
			try:
				# 用“当前模型的 num_experts”更新临时全局，保证指标函数的循环上限合理
				global NUM_EXPERTS
				old_ne = NUM_EXPERTS
				NUM_EXPERTS = num_experts
				js_sep = expert_output_separation_js(model, X_val)
				NUM_EXPERTS = old_ne
				print(f"[MoE] epoch={epoch:3d} expert_js_separation={js_sep:.4f}")
			except Exception as e:
				print(f"[MoE] expert_js_separation 计算失败: {e}")

		if bad >= PATIENCE:
			print(f"[MoE] Early stopping at epoch {epoch}.")
			break

	if best_state:
		model.load_state_dict(best_state)

	info = {
		"train_losses": train_losses,
		"val_losses": val_losses,
		"aux_losses": aux_losses,
		"best_epoch": len(train_losses) - bad,
		"best_val_loss": best_val_loss,
		"bad": bad,
	}
	return model, info


def specialization_search(
	X_train: np.ndarray,
	P_train: np.ndarray,
	N_train: np.ndarray | None,
	X_val: np.ndarray,
	P_val: np.ndarray,
	N_val: np.ndarray | None,
	output_dir: str,
) -> dict:
	"""轻量调参：在保持 val_loss 不太差的情况下，尽量增大专家分化(JS)。

	搜索维度（小范围）：AUX_COEF 与 EXPERT_DIVERSITY_COEF。
	输出：CSV 排序表 + 最优配置 dict。
	"""
	global MAX_EPOCHS
	old_epochs = MAX_EPOCHS
	MAX_EPOCHS = max(1, int(old_epochs * SEARCH_EPOCH_SCALE))
	try:
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
				model, info = train_moe_with_params(
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
				)
				val_best = float(info.get("best_val_loss", np.nan))
				js_sep = float(expert_output_separation_js(model, X_val))
				# 目标：val_loss 越小越好，JS 越大越好
				obj = val_best - SEARCH_OBJECTIVE_BETA_JS * js_sep
				rows.append(
					{
						"trial": trial,
						"aux_coef": aux,
						"div_coef": div,
						"best_val_loss": val_best,
						"expert_js_sep": js_sep,
						"objective": obj,
					}
				)
			if trial > SEARCH_MAX_TRIALS:
				break

		df = pd.DataFrame(rows).sort_values("objective", ascending=True)
		os.makedirs(output_dir, exist_ok=True)
		csv_path = os.path.join(output_dir, "specialization_search_results.csv")
		df.to_csv(csv_path, index=False)
		print(f"[Search] 结果已保存: {csv_path}")
		best = df.iloc[0].to_dict() if len(df) else {}
		return {"csv_path": csv_path, "best": best}
	finally:
		MAX_EPOCHS = old_epochs


def compute_metrics(P_pred: np.ndarray, P_test: np.ndarray) -> dict:
	"""
	计算预测分布与真实分布之间的多种性能指标。
	
	参数:
		P_pred: 预测的概率分布，形状 (n_samples, 7)
		P_test: 真实的概率分布，形状 (n_samples, 7)
	
	返回:
		包含以下指标的字典：
		- mae: 平均绝对误差（越小越好）
		- rmse: 均方根误差（越小越好）
		- kl: KL散度，衡量分布差异（越小越好）
		- js_mean: Jensen-Shannon散度，对称的分布差异度量（越小越好）
		- cos_sim: 余弦相似度，衡量分布向量的方向相似性（越接近1越好）
		- r2: 决定系数（越接近1越好）
		- max_error: 单个概率值的最大偏差
		- mse: 均方误差（越小越好）
		- total_variation: 总变差距离（越小越好）
	"""
	# 基础误差指标
	mae = np.mean(np.abs(P_pred - P_test))
	mse = np.mean((P_pred - P_test) ** 2)
	rmse = np.sqrt(mse)
	max_error = np.max(np.abs(P_pred - P_test))
	
	# 分布差异指标
	eps = 1e-12
	# KL散度: KL(p_true || p_pred)
	kl = np.mean(np.sum(P_test * (np.log(P_test + eps) - np.log(P_pred + eps)), axis=1))
	# Jensen-Shannon散度: 对称版本的KL
	js_mean = np.mean([jensenshannon(P_test[i], P_pred[i]) for i in range(len(P_test))])
	# 总变差距离 (Total Variation Distance)
	tv_distance = np.mean(np.sum(np.abs(P_pred - P_test), axis=1) / 2.0)
	
	# 相似性指标
	cos_sim = np.mean([cosine_similarity([P_test[i]], [P_pred[i]])[0, 0] for i in range(len(P_test))])
	r2 = r2_score(P_test, P_pred)
	
	return {
		"mae": float(mae),
		"rmse": float(rmse),
		"mse": float(mse),
		"kl": float(kl),
		"js_mean": float(js_mean),
		"tv_distance": float(tv_distance),
		"cos_sim": float(cos_sim),
		"r2": float(r2),
		"max_error": float(max_error),
	}


# ==============================================================================
# 专家数量与 TOP-K 网格搜索（新增功能）
# ==============================================================================
def expert_topk_grid_search(
	X_train: np.ndarray,
	P_train: np.ndarray,
	N_train: np.ndarray | None,
	X_val: np.ndarray,
	P_val: np.ndarray,
	N_val: np.ndarray | None,
	X_test: np.ndarray,
	P_test: np.ndarray,
	output_dir: str,
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
	
	返回:
		包含最优配置和完整搜索结果的字典
	
	搜索过程:
		1. 遍历所有超参数组合
		2. 为每个组合训练模型并评估
		3. 记录验证集损失、测试集指标、专家分化度(JS)
		4. 按综合目标排序选出最优配置
	"""
	global MAX_EPOCHS, NUM_EXPERTS, HIDDEN_SIZE, TOP_K
	
	# 保存原始配置
	old_epochs = MAX_EPOCHS
	old_num_experts = NUM_EXPERTS
	old_hidden = HIDDEN_SIZE
	old_topk = TOP_K
	
	# 缩短训练轮次以节省搜索时间
	MAX_EPOCHS = max(1, int(old_epochs * GRID_SEARCH_EPOCH_SCALE))
	
	print(f"\n{'='*70}")
	print(f"[网格搜索] 开始专家数量与 TOP-K 超参数搜索")
	print(f"[网格搜索] 专家数量候选: {EXPERT_NUM_GRID}")
	print(f"[网格搜索] TOP-K 候选: {TOPK_GRID}")
	print(f"[网格搜索] 隐藏层大小候选: {HIDDEN_SIZE_GRID}")
	print(f"[网格搜索] 训练轮次: {MAX_EPOCHS} (缩放比例: {GRID_SEARCH_EPOCH_SCALE})")
	print(f"{'='*70}\n")
	
	try:
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
		
		results: list[dict] = []
		trial = 0
		total_trials = len(EXPERT_NUM_GRID) * len(TOPK_GRID) * len(HIDDEN_SIZE_GRID)
		
		for num_exp in EXPERT_NUM_GRID:
			for topk in TOPK_GRID:
				# 跳过无效配置：TOP-K 不能超过专家数量
				if topk > num_exp:
					print(f"[网格搜索] 跳过无效配置: num_experts={num_exp}, top_k={topk} (top_k > num_experts)")
					continue
					
				for hidden in HIDDEN_SIZE_GRID:
					trial += 1
					print(f"\n[网格搜索] Trial {trial}/{total_trials}: "
						  f"num_experts={num_exp}, top_k={topk}, hidden={hidden}")
					
					# 更新全局配置供 expert_output_separation_js 使用
					NUM_EXPERTS = num_exp
					HIDDEN_SIZE = hidden
					TOP_K = topk
					
					# 训练模型
					model, info = train_moe_with_params(
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
					js_sep = float(expert_output_separation_js(model, X_val))
					
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
						  f"js={metrics['js_mean']:.4f}, r2={metrics['r2']:.4f}, "
						  f"expert_sep={js_sep:.4f}")
		
		# 创建结果 DataFrame
		df = pd.DataFrame(results)
		
		# 综合目标：主要优化 MAE（越小越好），同时考虑模型稳定性
		# objective = mae + 0.1 * val_loss - 0.05 * expert_js_sep
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
		print(f"  - js_mean: {best.get('js_mean'):.4f}")
		print(f"  - r2: {best.get('r2'):.4f}")
		print(f"  - expert_js_sep: {best.get('expert_js_sep'):.4f}")
		print(f"{'='*70}\n")
		
		# 可视化搜索结果
		plot_grid_search_results(df, output_dir)
		
		return {
			"csv_path": csv_path,
			"best": best,
			"all_results": df,
		}
		
	finally:
		# 恢复原始配置
		MAX_EPOCHS = old_epochs
		NUM_EXPERTS = old_num_experts
		HIDDEN_SIZE = old_hidden
		TOP_K = old_topk


def plot_grid_search_results(df: pd.DataFrame, output_dir: str) -> None:
	"""
	可视化网格搜索结果，生成多个图表帮助分析超参数的影响。
	
	图表包括:
	1. 不同专家数量下的 MAE 分布（箱线图）
	2. 不同 TOP-K 下的 MAE 分布（箱线图）
	3. 专家数量 vs TOP-K 的热力图（颜色=MAE）
	4. 所有配置的性能指标对比（条形图）
	"""
	os.makedirs(output_dir, exist_ok=True)
	
	# 创建综合图表
	fig, axes = plt.subplots(2, 2, figsize=(14, 12))
	
	# --- 图1: 不同专家数量下的 MAE 分布 ---
	ax1 = axes[0, 0]
	expert_groups = df.groupby("num_experts")["mae"].apply(list)
	ax1.boxplot([expert_groups.get(e, []) for e in EXPERT_NUM_GRID], 
				tick_labels=[str(e) for e in EXPERT_NUM_GRID])
	ax1.set_xlabel("专家数量 (num_experts)")
	ax1.set_ylabel("MAE")
	ax1.set_title("不同专家数量下的 MAE 分布")
	ax1.grid(axis="y", alpha=0.3)
	
	# --- 图2: 不同 TOP-K 下的 MAE 分布 ---
	ax2 = axes[0, 1]
	topk_groups = df.groupby("top_k")["mae"].apply(list)
	valid_topks = [k for k in TOPK_GRID if k in topk_groups.index]
	ax2.boxplot([topk_groups.get(k, []) for k in valid_topks],
				tick_labels=[str(k) for k in valid_topks])
	ax2.set_xlabel("TOP-K")
	ax2.set_ylabel("MAE")
	ax2.set_title("不同 TOP-K 下的 MAE 分布")
	ax2.grid(axis="y", alpha=0.3)
	
	# --- 图3: 专家数量 vs TOP-K 热力图（按最小 MAE 聚合） ---
	ax3 = axes[1, 0]
	pivot_data = df.groupby(["num_experts", "top_k"])["mae"].min().unstack(fill_value=np.nan)
	im = ax3.imshow(pivot_data.values, aspect="auto", cmap="RdYlGn_r")
	ax3.set_xticks(range(len(pivot_data.columns)))
	ax3.set_xticklabels(pivot_data.columns)
	ax3.set_yticks(range(len(pivot_data.index)))
	ax3.set_yticklabels(pivot_data.index)
	ax3.set_xlabel("TOP-K")
	ax3.set_ylabel("专家数量")
	ax3.set_title("专家数量 vs TOP-K 的最小 MAE（热力图）")
	cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
	cbar.set_label("MAE")
	
	# 在热力图上添加数值标注
	for i in range(len(pivot_data.index)):
		for j in range(len(pivot_data.columns)):
			val = pivot_data.values[i, j]
			if not np.isnan(val):
				ax3.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9)
	
	# --- 图4: Top 10 配置的性能对比 ---
	ax4 = axes[1, 1]
	top_10 = df.head(10).copy()
	top_10["config"] = top_10.apply(
		lambda r: f"E{int(r['num_experts'])}-K{int(r['top_k'])}-H{int(r['hidden_size'])}", axis=1
	)
	x = np.arange(len(top_10))
	width = 0.35
	ax4.bar(x - width/2, top_10["mae"], width, label="MAE", color="steelblue")
	ax4.bar(x + width/2, top_10["js_mean"], width, label="JS", color="coral")
	ax4.set_xticks(x)
	ax4.set_xticklabels(top_10["config"], rotation=45, ha="right")
	ax4.set_ylabel("指标值")
	ax4.set_title("Top 10 配置的性能对比")
	ax4.legend()
	ax4.grid(axis="y", alpha=0.3)
	
	fig.suptitle("网格搜索结果分析", fontsize=14, y=1.02)
	fig.tight_layout()
	
	save_path = os.path.join(output_dir, "grid_search_analysis.png")
	fig.savefig(save_path, dpi=250, bbox_inches="tight")
	plt.close(fig)
	print(f"[网格搜索] 分析图表已保存: {save_path}")


def evaluate(model, X_test: np.ndarray, P_test: np.ndarray) -> tuple[np.ndarray, dict]:
	"""在测试集上评估（与 Moe_Softmax.py 一致）。"""
	model.eval()
	Xte = torch.tensor(X_test, device=DEVICE)
	with torch.no_grad():
		P_pred, _ = model(Xte)
		P_pred = P_pred.cpu().numpy()

	metrics = compute_metrics(P_pred, P_test)

	print("\n[MoE] 测试集评估")
	for k, v in metrics.items():
		print(f"  {k}: {v:.6f}")

	return P_pred, metrics


def save_predictions(P_pred: np.ndarray, path: str) -> None:
	"""保存单模型点预测（与 Moe_Softmax.py 输出兼容的列名）。"""
	df_pred = pd.DataFrame(P_pred, columns=[f"moe_pred_{c}" for c in DIST_COLS])
	df_pred.to_csv(path, index=False)
	print(f"[MoE] 预测结果已保存: {path}")


# ==============================================================================
# Bootstrap：训练 + 预测（新增模块 2）
# ==============================================================================
def bootstrap_predict(
	X_train: np.ndarray,
	P_train: np.ndarray,
	N_train: np.ndarray | None,
	X_val: np.ndarray,
	P_val: np.ndarray,
	N_val: np.ndarray | None,
	X_test: np.ndarray,
	X_holdout: np.ndarray | None = None,
	B: int = BOOTSTRAP_B,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
	"""\
	使用 Bootstrap 对 MoE 模型进行不确定性估计。

	做法：
		重复 B 次：
			1) 对训练集进行 bootstrap 重采样（有放回，size=n_train）
			2) 训练一个 MoE 模型（不改结构与 loss）
			3) 在同一个测试集上预测，得到一份分布 P_pred^(b)

	返回:
		- 若 X_holdout is None: P_test_all, shape (B, n_test, 7)
		- 否则: (P_test_all, P_holdout_all)
			P_holdout_all shape (B, n_holdout, 7)
	"""

	P_test_all: list[np.ndarray] = []
	P_holdout_all: list[np.ndarray] = []
	n_train = X_train.shape[0]

	# 预先计算验证集权重（每次 bootstrap 训练都复用）
	Wva = (
		torch.tensor(make_weights_from_N(N_val, WEIGHT_MODE), device=DEVICE)
		if N_val is not None
		else None
	)

	global MAX_EPOCHS
	old_epochs = MAX_EPOCHS
	bootstrap_epochs = max(1, int(old_epochs * BOOTSTRAP_EPOCH_SCALE))

	for b in range(B):
		print(f"[Bootstrap] Run {b + 1}/{B}")

		# -------- 1) Bootstrap 重采样训练集 --------
		idx = np.random.choice(n_train, size=n_train, replace=True)
		Xb = X_train[idx]
		Pb = P_train[idx]
		Nb = N_train[idx] if N_train is not None else None

		# -------- 2) 重新计算训练集权重（保持与原逻辑一致） --------
		Wb = (
			torch.tensor(make_weights_from_N(Nb, WEIGHT_MODE), device=DEVICE)
			if Nb is not None
			else None
		)

		# -------- 3) 临时降低训练轮次（节省总计算） --------
		MAX_EPOCHS = bootstrap_epochs
		try:
			model_b, _ = train_moe(Xb, Pb, X_val, P_val, Wb, Wva)
		finally:
			MAX_EPOCHS = old_epochs

		# -------- 4) 在 test 集预测 --------
		model_b.eval()
		with torch.no_grad():
			Xte = torch.tensor(X_test, device=DEVICE)
			P_pred_b, _ = model_b(Xte)
			P_pred_b = P_pred_b.cpu().numpy()

			if X_holdout is not None:
				Xho = torch.tensor(X_holdout, device=DEVICE)
				P_hold_b, _ = model_b(Xho)
				P_hold_b = P_hold_b.cpu().numpy()

		P_test_all.append(P_pred_b)
		if X_holdout is not None:
			P_holdout_all.append(P_hold_b)

	P_test_boot = np.stack(P_test_all, axis=0)
	if X_holdout is None:
		return P_test_boot
	return P_test_boot, np.stack(P_holdout_all, axis=0)


def save_holdout_predictions_with_ci(
	words: list[str],
	P_mean: np.ndarray,
	P_std: np.ndarray,
	P_low: np.ndarray,
	P_high: np.ndarray,
	P_true: np.ndarray | None,
	path: str,
) -> None:
	"""保存 holdout（如 eerie）的预测与 bootstrap CI。"""
	rows: list[dict] = []
	for i, w in enumerate(words):
		row = {"word": w}
		for j, col in enumerate(DIST_COLS):
			row[f"mean_{col}"] = float(P_mean[i, j])
			row[f"std_{col}"] = float(P_std[i, j])
			row[f"ci_low_{col}"] = float(P_low[i, j])
			row[f"ci_high_{col}"] = float(P_high[i, j])
			if P_true is not None:
				row[f"true_{col}"] = float(P_true[i, j])
		rows.append(row)

	df = pd.DataFrame(rows)
	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
	df.to_csv(path, index=False)
	print(f"[Holdout] 预测+CI 已保存: {path}")


def plot_holdout_bar_with_ci(
	P_mean: np.ndarray,
	P_low: np.ndarray,
	P_high: np.ndarray,
	P_true: np.ndarray | None,
	word: str,
	save_path: str,
) -> None:
	"""holdout 的均值分布柱状/误差条图（逐桶）。

	如果 holdout 有多条样本，会先对样本取平均后作图。
	"""
	mean = P_mean.mean(axis=0)
	low = P_low.mean(axis=0)
	high = P_high.mean(axis=0)
	yerr = np.vstack([np.maximum(0, mean - low), np.maximum(0, high - mean)])

	fig, ax = plt.subplots(figsize=(10, 4))
	x = np.arange(len(DIST_COLS))
	ax.bar(x, mean, color="steelblue", alpha=0.8, label="bootstrap mean")
	ax.errorbar(x, mean, yerr=yerr, fmt="none", ecolor="black", capsize=4, linewidth=1)
	if P_true is not None:
		true_mean = P_true.mean(axis=0)
		ax.plot(x, true_mean, linestyle="--", marker="x", color="coral", label="true")
	ax.set_xticks(x)
	ax.set_xticklabels(DIST_COLS, rotation=30, ha="right")
	ax.set_ylabel("Probability")
	ax.set_ylim(0.0, max(0.35, float(high.max()) * 1.25))
	ax.set_title(f"Holdout '{word}' predicted distribution (mean + CI)")
	ax.grid(axis="y", alpha=0.3)
	ax.legend(loc="best")
	fig.tight_layout()
	os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
	fig.savefig(save_path, dpi=250, bbox_inches="tight")
	plt.close(fig)
	print(f"[Holdout] 柱状+CI 图已保存: {save_path}")


def plot_holdout_violin(
	P_holdout_boot: np.ndarray,
	word: str,
	save_path: str,
) -> None:
	"""holdout 的 bootstrap 分布小提琴图。

	P_holdout_boot: shape (B, n_holdout, 7)
	将每个桶的 (B * n_holdout) 个预测值作为一个小提琴。
	"""
	B, n_h, d = P_holdout_boot.shape
	vals = []
	for j in range(d):
		vals.append(P_holdout_boot[:, :, j].reshape(B * n_h))

	fig, ax = plt.subplots(figsize=(12, 4.5))
	parts = ax.violinplot(vals, positions=np.arange(d), showmeans=True, showextrema=True)
	for pc in parts["bodies"]:
		pc.set_facecolor("slateblue")
		pc.set_edgecolor("black")
		pc.set_alpha(0.55)
	ax.set_xticks(np.arange(d))
	ax.set_xticklabels(DIST_COLS, rotation=30, ha="right")
	ax.set_ylabel("Probability")
	ax.set_title(f"Holdout '{word}' bootstrap predictive distribution (violin)")
	ax.grid(axis="y", alpha=0.3)
	fig.tight_layout()
	os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
	fig.savefig(save_path, dpi=250, bbox_inches="tight")
	plt.close(fig)
	print(f"[Holdout] 小提琴图已保存: {save_path}")


def bootstrap_summary(P_boot: np.ndarray, ci_level: float = BOOTSTRAP_CI_LEVEL) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""对 bootstrap 预测做聚合，返回 (mean, std, low, high)。"""
	alpha_low = (1 - ci_level) / 2
	alpha_high = 1 - alpha_low

	P_mean = P_boot.mean(axis=0)
	P_std = P_boot.std(axis=0)
	P_low = np.percentile(P_boot, alpha_low * 100, axis=0)
	P_high = np.percentile(P_boot, alpha_high * 100, axis=0)

	return P_mean, P_std, P_low, P_high


def save_bootstrap_predictions(
	P_mean: np.ndarray,
	P_std: np.ndarray,
	P_low: np.ndarray,
	P_high: np.ndarray,
	path: str,
) -> None:
	"""保存 bootstrap 预测汇总到 CSV（每个桶输出 mean/std/CI）。"""
	data: dict[str, np.ndarray] = {}
	for i, col in enumerate(DIST_COLS):
		data[f"mean_{col}"] = P_mean[:, i]
		data[f"std_{col}"] = P_std[:, i]
		data[f"ci_low_{col}"] = P_low[:, i]
		data[f"ci_high_{col}"] = P_high[:, i]

	df = pd.DataFrame(data)
	df.to_csv(path, index=False)
	print(f"[Bootstrap] 汇总预测已保存: {path}")


# ==============================================================================
# 作图工具（仿造 Moe_Softmax.py 的作图风格，内置到本脚本中）
# ==============================================================================

def plot_training_history(
	train_losses: list[float],
	val_losses: list[float],
	bad: int,
	best_val_loss: float,
	save_path: str,
) -> dict:
	"""绘制训练/验证损失曲线（单模型）。"""
	fig, ax = plt.subplots(figsize=(10, 4))
	ax.plot(train_losses, label="Train", alpha=0.85)
	ax.plot(val_losses, label="Val", alpha=0.85)
	best_epoch = max(0, len(train_losses) - bad)
	ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.8, label=f"Best@{best_epoch}")
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Loss")
	ax.set_title(f"Training History (best_val={best_val_loss:.4f})")
	ax.grid(alpha=0.3)
	ax.legend(loc="best")
	fig.tight_layout()
	os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
	fig.savefig(save_path, dpi=250, bbox_inches="tight")
	plt.close(fig)
	print(f"[MoE] 训练曲线已保存: {save_path}")
	return {"best_epoch": int(best_epoch), "best_val_loss": float(best_val_loss)}


def plot_random_sample_distributions(
	P_test: np.ndarray,
	P_pred: np.ndarray,
	sample_size: int,
	save_path: str,
) -> list[int]:
	"""随机抽样对比真实分布 vs 预测分布（单模型或 P_mean 都可）。"""
	n = len(P_test)
	if n == 0:
		return []
	sample_size = int(min(sample_size, n))
	idx = np.random.choice(n, size=sample_size, replace=False).tolist()

	n_cols = min(5, sample_size)
	n_rows = (sample_size + n_cols - 1) // n_cols
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.3 * n_rows))
	axes = np.atleast_2d(axes).flatten()

	x = np.arange(len(DIST_COLS))
	width = 0.35
	for k, i in enumerate(idx):
		ax = axes[k]
		ax.bar(x - width / 2, P_test[i], width, label="True", color="steelblue")
		ax.bar(x + width / 2, P_pred[i], width, label="Pred", color="coral")
		ax.set_xticks(x)
		ax.set_xticklabels([c.replace("_", "\n") for c in DIST_COLS], fontsize=7)
		ax.set_ylim(0, max(0.35, float(max(P_test[i].max(), P_pred[i].max())) * 1.2))
		ax.set_title(f"Sample {i}")
		ax.grid(axis="y", alpha=0.3)
		ax.legend(fontsize=8)

	for k in range(len(idx), len(axes)):
		axes[k].set_visible(False)

	fig.tight_layout()
	os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
	fig.savefig(save_path, dpi=250, bbox_inches="tight")
	plt.close(fig)
	print(f"[MoE] 随机样本分布对比图已保存: {save_path}")
	return idx


def plot_error_analysis(P_test: np.ndarray, P_pred: np.ndarray, save_path: str) -> dict:
	"""
	误差分析：每桶 MAE + 样本级 MAE 分布。
	
	参数:
		P_test: 真实分布，形状 (n_samples, 7)
		P_pred: 预测分布，形状 (n_samples, 7)
		save_path: 图表保存路径
	
	返回:
		包含每桶MAE和每样本MAE的字典
	"""
	errors = P_pred - P_test
	mae_per_dim = np.mean(np.abs(errors), axis=0)
	mae_per_sample = np.mean(np.abs(errors), axis=1)

	fig, axes = plt.subplots(1, 2, figsize=(12, 4))
	x = np.arange(len(DIST_COLS))

	axes[0].bar(x, mae_per_dim, color="teal", edgecolor="black")
	axes[0].set_xticks(x)
	axes[0].set_xticklabels([c.replace("_", "\n") for c in DIST_COLS], fontsize=7)
	axes[0].set_ylabel("MAE")
	axes[0].set_title("各桶的平均绝对误差 (MAE per Bin)")
	axes[0].grid(axis="y", alpha=0.3)

	axes[1].hist(mae_per_sample, bins=30, color="slateblue", alpha=0.75, edgecolor="black")
	axes[1].set_xlabel("每样本 MAE (Per-sample MAE)")
	axes[1].set_ylabel("样本数量")
	axes[1].set_title("样本级 MAE 分布")
	axes[1].axvline(np.mean(mae_per_sample), color="red", linestyle="--", 
					label=f"均值={np.mean(mae_per_sample):.4f}")
	axes[1].legend()
	axes[1].grid(axis="y", alpha=0.3)

	fig.tight_layout()
	os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
	fig.savefig(save_path, dpi=250, bbox_inches="tight")
	plt.close(fig)
	print(f"[MoE] 误差分析图已保存: {save_path}")

	return {"mae_per_dim": mae_per_dim.tolist(), "mae_per_sample": mae_per_sample, "errors": errors}


def plot_performance_metrics(
	P_test: np.ndarray,
	P_pred: np.ndarray,
	save_path: str,
) -> dict:
	"""
	绘制各项性能指标的可视化图表。
	
	图表包括:
	1. 各性能指标的雷达图
	2. 预测值与真实值的散点图（各桶）
	3. 残差分析图
	
	参数:
		P_test: 真实分布
		P_pred: 预测分布
		save_path: 保存路径
	
	返回:
		计算得到的性能指标字典
	"""
	# 计算各项指标
	metrics = compute_metrics(P_pred, P_test)
	
	fig, axes = plt.subplots(2, 2, figsize=(14, 12))
	
	# --- 图1: 性能指标条形图 ---
	ax1 = axes[0, 0]
	# 选择主要指标进行展示（数值范围相近的）
	main_metrics = ["mae", "rmse", "js_mean", "tv_distance", "max_error"]
	values = [metrics[m] for m in main_metrics]
	colors = ["steelblue", "coral", "seagreen", "purple", "orange"]
	bars = ax1.bar(range(len(main_metrics)), values, color=colors, edgecolor="black")
	ax1.set_xticks(range(len(main_metrics)))
	ax1.set_xticklabels(["MAE", "RMSE", "JS散度", "TV距离", "最大误差"], rotation=15)
	ax1.set_ylabel("指标值")
	ax1.set_title("误差类指标对比 (越小越好)")
	ax1.grid(axis="y", alpha=0.3)
	# 添加数值标注
	for bar, val in zip(bars, values):
		ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
				f"{val:.4f}", ha="center", va="bottom", fontsize=9)
	
	# --- 图2: 相似性指标 ---
	ax2 = axes[0, 1]
	sim_metrics = ["cos_sim", "r2"]
	sim_values = [metrics[m] for m in sim_metrics]
	colors2 = ["steelblue", "coral"]
	bars2 = ax2.bar(range(len(sim_metrics)), sim_values, color=colors2, edgecolor="black")
	ax2.set_xticks(range(len(sim_metrics)))
	ax2.set_xticklabels(["余弦相似度", "R² 决定系数"])
	ax2.set_ylabel("指标值")
	ax2.set_title("相似性指标对比 (越接近1越好)")
	ax2.set_ylim(0, 1.1)
	ax2.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="理想值=1")
	ax2.grid(axis="y", alpha=0.3)
	ax2.legend()
	for bar, val in zip(bars2, sim_values):
		ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
				f"{val:.4f}", ha="center", va="bottom", fontsize=10)
	
	# --- 图3: 各桶的预测值 vs 真实值散点图 ---
	ax3 = axes[1, 0]
	for j, col in enumerate(DIST_COLS):
		ax3.scatter(P_test[:, j], P_pred[:, j], alpha=0.4, s=15, label=col)
	# 添加对角线（完美预测线）
	ax3.plot([0, 1], [0, 1], "k--", alpha=0.8, label="完美预测线")
	ax3.set_xlabel("真实概率")
	ax3.set_ylabel("预测概率")
	ax3.set_title("各桶预测值 vs 真实值散点图")
	ax3.set_xlim(-0.02, max(0.5, P_test.max() * 1.1))
	ax3.set_ylim(-0.02, max(0.5, P_pred.max() * 1.1))
	ax3.legend(fontsize=8, ncol=2, loc="best")
	ax3.grid(alpha=0.3)
	
	# --- 图4: 残差分析（预测 - 真实） ---
	ax4 = axes[1, 1]
	residuals = P_pred - P_test
	residual_means = residuals.mean(axis=0)
	residual_stds = residuals.std(axis=0)
	x = np.arange(len(DIST_COLS))
	ax4.bar(x, residual_means, yerr=residual_stds, color="steelblue", 
			edgecolor="black", alpha=0.7, capsize=4)
	ax4.axhline(y=0, color="red", linestyle="--", alpha=0.8)
	ax4.set_xticks(x)
	ax4.set_xticklabels([c.replace("_", "\n") for c in DIST_COLS], fontsize=8)
	ax4.set_ylabel("残差 (预测 - 真实)")
	ax4.set_title("各桶残差分析 (均值 ± 标准差)")
	ax4.grid(axis="y", alpha=0.3)
	
	fig.suptitle("MoE 模型性能指标综合分析", fontsize=14, y=1.02)
	fig.tight_layout()
	
	os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
	fig.savefig(save_path, dpi=250, bbox_inches="tight")
	plt.close(fig)
	print(f"[MoE] 性能指标图已保存: {save_path}")
	
	return metrics


def plot_comprehensive_summary(
	train_losses: list[float],
	val_losses: list[float],
	bad: int,
	best_val_loss: float,
	metrics: dict,
	mae_per_dim: list[float],
	P_test: np.ndarray,
	P_pred: np.ndarray,
	expert_usage: list[float],
	save_path: str,
) -> None:
	"""
	生成综合汇总图，将所有关键信息整合到一张大图中。
	
	图表布局 (3x3):
	- 训练损失曲线
	- 性能指标条形图
	- 各桶MAE
	- 专家使用率
	- 预测vs真实散点图
	- 随机样本对比
	- 残差分布
	- 每样本MAE分布
	- 指标总结文本
	
	参数:
		train_losses, val_losses: 训练和验证损失历史
		bad: 早停时无改善的轮次数
		best_val_loss: 最佳验证损失
		metrics: 性能指标字典
		mae_per_dim: 各桶的MAE
		P_test, P_pred: 真实和预测分布
		expert_usage: 专家使用率
		save_path: 保存路径
	"""
	fig = plt.figure(figsize=(18, 15))
	gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
	
	# --- 图1: 训练损失曲线 ---
	ax1 = fig.add_subplot(gs[0, 0])
	ax1.plot(train_losses, label="训练损失", alpha=0.85)
	ax1.plot(val_losses, label="验证损失", alpha=0.85)
	best_epoch = max(0, len(train_losses) - bad)
	ax1.axvline(best_epoch, color="red", linestyle="--", alpha=0.8, label=f"最佳@{best_epoch}")
	ax1.set_xlabel("Epoch")
	ax1.set_ylabel("损失")
	ax1.set_title(f"训练曲线 (最佳验证损失={best_val_loss:.4f})")
	ax1.legend(fontsize=9)
	ax1.grid(alpha=0.3)
	
	# --- 图2: 主要性能指标 ---
	ax2 = fig.add_subplot(gs[0, 1])
	main_metrics = ["mae", "rmse", "js_mean", "cos_sim", "r2"]
	values = [metrics.get(m, 0) for m in main_metrics]
	labels = ["MAE", "RMSE", "JS散度", "余弦相似度", "R²"]
	colors = ["steelblue", "coral", "seagreen", "purple", "orange"]
	bars = ax2.bar(range(len(main_metrics)), values, color=colors, edgecolor="black")
	ax2.set_xticks(range(len(main_metrics)))
	ax2.set_xticklabels(labels, rotation=20, fontsize=9)
	ax2.set_title("性能指标")
	ax2.grid(axis="y", alpha=0.3)
	for bar, val in zip(bars, values):
		ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
				f"{val:.3f}", ha="center", va="bottom", fontsize=8)
	
	# --- 图3: 各桶MAE ---
	ax3 = fig.add_subplot(gs[0, 2])
	x = np.arange(len(DIST_COLS))
	ax3.bar(x, mae_per_dim, color="teal", edgecolor="black")
	ax3.set_xticks(x)
	ax3.set_xticklabels([c.replace("_", "\n") for c in DIST_COLS], fontsize=7)
	ax3.set_ylabel("MAE")
	ax3.set_title("各桶 MAE")
	ax3.grid(axis="y", alpha=0.3)
	
	# --- 图4: 专家使用率 ---
	ax4 = fig.add_subplot(gs[1, 0])
	exp_x = np.arange(len(expert_usage))
	ax4.bar(exp_x, expert_usage, color="coral", edgecolor="black")
	ax4.set_xticks(exp_x)
	ax4.set_xticklabels([f"专家{i}" for i in range(len(expert_usage))])
	ax4.set_ylabel("使用率")
	ax4.set_title("专家使用率")
	ax4.grid(axis="y", alpha=0.3)
	
	# --- 图5: 预测vs真实散点图 ---
	ax5 = fig.add_subplot(gs[1, 1])
	for j, col in enumerate(DIST_COLS[:4]):  # 只画前4个桶避免拥挤
		ax5.scatter(P_test[:, j], P_pred[:, j], alpha=0.3, s=10, label=col)
	ax5.plot([0, 0.5], [0, 0.5], "k--", alpha=0.8)
	ax5.set_xlabel("真实概率")
	ax5.set_ylabel("预测概率")
	ax5.set_title("预测 vs 真实 (部分桶)")
	ax5.legend(fontsize=8)
	ax5.grid(alpha=0.3)
	
	# --- 图6: 随机样本对比 ---
	ax6 = fig.add_subplot(gs[1, 2])
	n = min(3, len(P_test))
	idx = np.random.choice(len(P_test), size=n, replace=False)
	width = 0.25
	for k, i in enumerate(idx):
		offset = (k - 1) * width
		ax6.bar(x + offset, P_test[i], width, alpha=0.7, label=f"真实{k+1}")
	ax6.set_xticks(x)
	ax6.set_xticklabels([c.split("_")[0] for c in DIST_COLS], fontsize=8)
	ax6.set_title("随机样本真实分布")
	ax6.legend(fontsize=8)
	ax6.grid(axis="y", alpha=0.3)
	
	# --- 图7: 残差分布直方图 ---
	ax7 = fig.add_subplot(gs[2, 0])
	residuals = (P_pred - P_test).flatten()
	ax7.hist(residuals, bins=50, color="slateblue", alpha=0.7, edgecolor="black")
	ax7.axvline(0, color="red", linestyle="--")
	ax7.set_xlabel("残差")
	ax7.set_ylabel("频数")
	ax7.set_title(f"残差分布 (均值={residuals.mean():.4f})")
	ax7.grid(axis="y", alpha=0.3)
	
	# --- 图8: 每样本MAE分布 ---
	ax8 = fig.add_subplot(gs[2, 1])
	mae_per_sample = np.mean(np.abs(P_pred - P_test), axis=1)
	ax8.hist(mae_per_sample, bins=30, color="coral", alpha=0.7, edgecolor="black")
	ax8.axvline(mae_per_sample.mean(), color="red", linestyle="--",
				label=f"均值={mae_per_sample.mean():.4f}")
	ax8.set_xlabel("每样本 MAE")
	ax8.set_ylabel("样本数")
	ax8.set_title("样本级MAE分布")
	ax8.legend()
	ax8.grid(axis="y", alpha=0.3)
	
	# --- 图9: 指标总结文本 ---
	ax9 = fig.add_subplot(gs[2, 2])
	ax9.axis("off")
	summary_text = f"""
	模型配置
	─────────────────
	专家数量: {NUM_EXPERTS}
	隐藏层大小: {HIDDEN_SIZE}
	Top-K: {TOP_K}
	辅助损失系数: {AUX_COEF}
	
	性能指标
	─────────────────
	MAE: {metrics.get('mae', 0):.4f}
	RMSE: {metrics.get('rmse', 0):.4f}
	JS散度: {metrics.get('js_mean', 0):.4f}
	KL散度: {metrics.get('kl', 0):.4f}
	余弦相似度: {metrics.get('cos_sim', 0):.4f}
	R²: {metrics.get('r2', 0):.4f}
	最大误差: {metrics.get('max_error', 0):.4f}
	TV距离: {metrics.get('tv_distance', 0):.4f}
	
	训练信息
	─────────────────
	最佳轮次: {best_epoch}
	最佳验证损失: {best_val_loss:.4f}
	"""
	ax9.text(0.1, 0.95, summary_text, transform=ax9.transAxes, 
			fontsize=10, verticalalignment="top", fontfamily="monospace",
			bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
	
	fig.suptitle("MoE 模型综合性能汇总", fontsize=16, y=0.98)
	
	os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
	fig.savefig(save_path, dpi=250, bbox_inches="tight")
	plt.close(fig)
	print(f"[MoE] 综合汇总图已保存: {save_path}")


def plot_aux_loss_curve(aux_losses: list[float], save_path: str) -> None:
	"""
	绘制辅助损失（负载平衡损失）曲线。
	
	辅助损失用于平衡各专家的使用率，避免"专家塌陷"问题。
	理想情况下，辅助损失应该保持较低且稳定。
	
	参数:
		aux_losses: 每个epoch的辅助损失列表
		save_path: 图表保存路径
	"""
	fig, ax = plt.subplots(figsize=(10, 4))
	ax.plot(aux_losses, label="辅助损失 (负载平衡)", color="green", alpha=0.8)
	ax.set_xlabel("Epoch")
	ax.set_ylabel("辅助损失")
	ax.set_title("MoE 辅助损失曲线")
	ax.legend()
	ax.grid(alpha=0.3)
	fig.tight_layout()
	os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
	fig.savefig(save_path, dpi=250, bbox_inches="tight")
	plt.close(fig)
	print(f"[MoE] 辅助损失曲线已保存: {save_path}")


def analyze_expert_usage(model: MoE, X_data: np.ndarray, save_path: str) -> dict:
	"""分析并可视化专家使用率（需要读取门控 gates）。"""
	model.eval()
	Xte = torch.tensor(X_data, device=DEVICE)
	with torch.no_grad():
		gates, _load = model.noisy_top_k_gating(Xte, train=False)
		gates_np = gates.cpu().numpy()

	expert_usage = (gates_np > 0).mean(axis=0)
	expert_avg_weight = gates_np.mean(axis=0)

	fig, axes = plt.subplots(1, 2, figsize=(12, 4))
	x = np.arange(NUM_EXPERTS)

	axes[0].bar(x, expert_usage, color="steelblue", edgecolor="black")
	axes[0].set_xlabel("Expert")
	axes[0].set_ylabel("Usage Rate")
	axes[0].set_title("Expert Usage Rate")
	axes[0].set_xticks(x)
	axes[0].set_xticklabels([f"E{i}" for i in range(NUM_EXPERTS)])
	axes[0].grid(axis="y", alpha=0.3)

	axes[1].bar(x, expert_avg_weight, color="coral", edgecolor="black")
	axes[1].set_xlabel("Expert")
	axes[1].set_ylabel("Avg Gate Weight")
	axes[1].set_title("Average Gate Weight")
	axes[1].set_xticks(x)
	axes[1].set_xticklabels([f"E{i}" for i in range(NUM_EXPERTS)])
	axes[1].grid(axis="y", alpha=0.3)

	fig.tight_layout()
	os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
	fig.savefig(save_path, dpi=250, bbox_inches="tight")
	plt.close(fig)
	print(f"[MoE] 专家使用率图已保存: {save_path}")

	return {
		"expert_usage": expert_usage.tolist(),
		"expert_avg_weight": expert_avg_weight.tolist(),
	}


def explain_expert_distributions(
	model: MoE,
	X_data: np.ndarray,
	P_true: np.ndarray,
	P_pred: np.ndarray,
	output_dir: str,
	prefix: str = "test",
) -> dict:
	"""按门控路由把样本分配给专家，并输出每个专家负责样本的分布画像。

	输出：
	- CSV：每个专家的样本数、占比、真实/预测均值分布、每桶 MAE
	- PNG：样本占比柱状图 + 每专家真实vs预测均值分布图 + 合并曲线图
	"""
	os.makedirs(output_dir, exist_ok=True)
	model.eval()
	Xte = torch.tensor(X_data, device=DEVICE)
	with torch.no_grad():
		gates, _load = model.noisy_top_k_gating(Xte, train=False)
		gates_np = gates.cpu().numpy()

	assigned_expert = gates_np.argmax(axis=1)
	rows = []
	for e in range(NUM_EXPERTS):
		mask = assigned_expert == e
		cnt = int(mask.sum())
		ratio = float(cnt / len(assigned_expert)) if len(assigned_expert) else 0.0
		if cnt == 0:
			true_mean = np.zeros(len(DIST_COLS), dtype=float)
			pred_mean = np.zeros(len(DIST_COLS), dtype=float)
			mae_bins = np.zeros(len(DIST_COLS), dtype=float)
		else:
			true_mean = P_true[mask].mean(axis=0)
			pred_mean = P_pred[mask].mean(axis=0)
			mae_bins = np.mean(np.abs(P_pred[mask] - P_true[mask]), axis=0)

		row = {"expert": e, "count": cnt, "ratio": ratio}
		for i, c in enumerate(DIST_COLS):
			row[f"true_mean_{c}"] = float(true_mean[i])
			row[f"pred_mean_{c}"] = float(pred_mean[i])
			row[f"mae_{c}"] = float(mae_bins[i])
		rows.append(row)

	df_exp = pd.DataFrame(rows)
	csv_path = os.path.join(output_dir, f"moe_expert_distribution_summary_{prefix}.csv")
	df_exp.to_csv(csv_path, index=False)
	print(f"[MoE] 专家分布解释CSV已保存: {csv_path}")

	# 图 1：样本占比
	fig, ax = plt.subplots(figsize=(8, 4))
	ax.bar(df_exp["expert"].astype(str), df_exp["ratio"], color="slateblue", edgecolor="black")
	ax.set_xlabel("Expert")
	ax.set_ylabel("Sample Ratio")
	ax.set_title(f"Expert Sample Ratio ({prefix})")
	ax.grid(axis="y", alpha=0.3)
	ratio_path = os.path.join(output_dir, f"moe_expert_sample_ratio_{prefix}.png")
	fig.tight_layout()
	fig.savefig(ratio_path, dpi=250, bbox_inches="tight")
	plt.close(fig)
	print(f"[MoE] 专家样本占比图已保存: {ratio_path}")

	# 图 2：每个专家的真实 vs 预测均值分布（逐专家分面）
	n_bins = len(DIST_COLS)
	x = np.arange(n_bins)
	fig, axes = plt.subplots(NUM_EXPERTS, 1, figsize=(12, 3.2 * NUM_EXPERTS), sharex=True)
	if NUM_EXPERTS == 1:
		axes = [axes]
	for e in range(NUM_EXPERTS):
		ax = axes[e]
		row = df_exp[df_exp["expert"] == e].iloc[0]
		true_mean = np.array([row[f"true_mean_{c}"] for c in DIST_COLS])
		pred_mean = np.array([row[f"pred_mean_{c}"] for c in DIST_COLS])
		width = 0.35
		ax.bar(x - width / 2, true_mean, width, label="True mean", color="steelblue")
		ax.bar(x + width / 2, pred_mean, width, label="Pred mean", color="coral")
		ax.set_title(f"Expert {e} (n={int(row['count'])}, ratio={float(row['ratio']):.1%})")
		ax.grid(axis="y", alpha=0.3)
		ax.legend(loc="best", fontsize=9)
	axes[-1].set_xticks(x)
	axes[-1].set_xticklabels(DIST_COLS)
	axes[-1].set_xlabel("Bins (tries)")
	dist_path = os.path.join(output_dir, f"moe_expert_mean_distribution_{prefix}.png")
	fig.tight_layout()
	fig.savefig(dist_path, dpi=250, bbox_inches="tight")
	plt.close(fig)
	print(f"[MoE] 专家均值分布对比图已保存: {dist_path}")

	# 图 3：合并曲线（不同专家的预测均值分布叠加）
	fig, ax = plt.subplots(figsize=(12, 5))
	for e in range(NUM_EXPERTS):
		row = df_exp[df_exp["expert"] == e].iloc[0]
		pred_mean = np.array([row[f"pred_mean_{c}"] for c in DIST_COLS])
		true_mean = np.array([row[f"true_mean_{c}"] for c in DIST_COLS])
		ax.plot(x, pred_mean, marker="o", linewidth=2, label=f"E{e} pred")
		ax.plot(x, true_mean, linestyle="--", alpha=0.8, label=f"E{e} true")
	ax.set_xticks(x)
	ax.set_xticklabels(DIST_COLS)
	ax.set_xlabel("Bins (tries)")
	ax.set_ylabel("Probability")
	ax.set_title(f"Expert Mean Distribution Curves ({prefix})")
	ax.grid(alpha=0.3)
	ax.legend(ncol=2, fontsize=9)
	curve_path = os.path.join(output_dir, f"moe_expert_mean_distribution_curve_{prefix}.png")
	fig.tight_layout()
	fig.savefig(curve_path, dpi=250, bbox_inches="tight")
	plt.close(fig)
	print(f"[MoE] 专家均值分布曲线图已保存: {curve_path}")

	return {
		"csv_path": csv_path,
		"ratio_path": ratio_path,
		"dist_path": dist_path,
		"curve_path": curve_path,
	}


# ---------------------- 新增：更细的专家可视化 ----------------------

def compute_expert_outputs(
	model: MoE,
	X_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""返回 (y_experts, gates, y_mixture)

	- y_experts: shape (num_experts, n, 7)
	- gates:     shape (n, num_experts)
	- y_mixture: shape (n, 7) = sum_e gates[:, e] * y_experts[e]
	"""
	model.eval()
	Xte = torch.tensor(X_data, device=DEVICE)
	with torch.no_grad():
		# 逐专家前向
		y_exps = []
		for exp in model.experts:
			y_exp = exp(Xte)  # (n, 7)
			y_exps.append(y_exp)
		y_experts = torch.stack(y_exps, dim=0)  # (E, n, 7)

		# 门控权重（top-k 后归一）
		gates, _ = model.noisy_top_k_gating(Xte, train=False)  # (n, E)

		# 混合输出
		y_mix = (gates.unsqueeze(-1) * y_experts.permute(1, 0, 2)).sum(dim=1)  # (n, 7)

	return (
		y_experts.cpu().numpy(),
		gates.cpu().numpy(),
		y_mix.cpu().numpy(),
	)


def plot_expert_gate_heatmap(
	gates: np.ndarray,
	save_path: str,
	max_samples: int = 500,
) -> None:
	"""展示部分样本的门控权重热力图，直观反映路由差异。"""
	n = gates.shape[0]
	idx = np.arange(n)
	if n > max_samples:
		idx = np.random.choice(n, size=max_samples, replace=False)
	G = gates[idx]
	# 按分配专家排序，便于观察
	assigned = G.argmax(axis=1)
	order = np.argsort(assigned)
	G = G[order]
	fig, ax = plt.subplots(figsize=(1.2 * gates.shape[1] + 2, 0.12 * G.shape[0] + 2))
	im = ax.imshow(G, aspect="auto", cmap="viridis")
	ax.set_xlabel("Experts")
	ax.set_ylabel("Samples (sorted by assigned expert)")
	ax.set_xticks(np.arange(gates.shape[1]))
	ax.set_xticklabels([f"E{i}" for i in range(gates.shape[1])])
	ax.set_title("Gate Weights Heatmap")
	cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
	cbar.set_label("Gate Weight")
	fig.tight_layout()
	os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
	fig.savefig(save_path, dpi=250, bbox_inches="tight")
	plt.close(fig)
	print(f"[MoE] 门控权重热力图已保存: {save_path}")


def plot_expert_mae_heatmap(
	P_true: np.ndarray,
	y_experts: np.ndarray,
	gates: np.ndarray,
	save_path: str,
) -> None:
	"""按分配专家计算每个专家在各个桶的 MAE，并作热力图。

	- y_experts: (E, n, 7)
	- gates: (n, E)
	"""
	E = y_experts.shape[0]
	n = gates.shape[0]
	assigned = gates.argmax(axis=1)
	mae_mat = np.zeros((E, len(DIST_COLS)), dtype=float)
	for e in range(E):
		mask = assigned == e
		if not np.any(mask):
			continue
		pred_mean = y_experts[e, mask]  # (m, 7)
		true_sel = P_true[mask]
		mae_bins = np.mean(np.abs(pred_mean - true_sel), axis=0)
		mae_mat[e] = mae_bins

	fig, ax = plt.subplots(figsize=(1.2 * len(DIST_COLS) + 3, 0.8 * E + 2))
	im = ax.imshow(mae_mat, aspect="auto", cmap="magma")
	ax.set_yticks(np.arange(E))
	ax.set_yticklabels([f"E{e}" for e in range(E)])
	ax.set_xticks(np.arange(len(DIST_COLS)))
	ax.set_xticklabels([c.replace("_", "\n") for c in DIST_COLS])
	ax.set_xlabel("Bins (tries)")
	ax.set_title("Expert-wise MAE per Bin (assigned samples)")
	cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
	cbar.set_label("MAE")
	fig.tight_layout()
	os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
	fig.savefig(save_path, dpi=250, bbox_inches="tight")
	plt.close(fig)
	print(f"[MoE] 专家 MAE 热力图已保存: {save_path}")


def plot_sample_expert_decomposition(
	P_true: np.ndarray,
	y_experts: np.ndarray,
	gates: np.ndarray,
	y_mix: np.ndarray,
	save_path: str,
	sample_size: int = 6,
) -> None:
	"""随机抽样若干样本，展示：True vs Mixture 以及 Top-2 专家的输出与权重。"""
	n = P_true.shape[0]
	if n == 0:
		return
	idx = np.random.choice(n, size=min(sample_size, n), replace=False)
	x = np.arange(len(DIST_COLS))
	n_cols = 3
	n_rows = (len(idx) + n_cols - 1) // n_cols
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.6 * n_rows))
	axes = np.atleast_2d(axes).flatten()
	for k, i in enumerate(idx):
		ax = axes[k]
		# mixture vs true
		ax.plot(x, y_mix[i], marker="o", linewidth=2, label="Mixture")
		ax.plot(x, P_true[i], linestyle="--", marker="x", label="True")
		# top-2 experts
		g = gates[i]
		top2 = np.argsort(-g)[:2]
		for t in top2:
			ax.plot(x, y_experts[t, i], linewidth=1.5, alpha=0.8, label=f"E{t} (w={g[t]:.2f})")
		ax.set_xticks(x)
		ax.set_xticklabels([c.replace("_", "\n") for c in DIST_COLS], fontsize=8)
		ax.set_ylim(0, max(0.35, float(max(y_mix[i].max(), P_true[i].max())) * 1.2))
		ax.set_title(f"Sample {i}")
		ax.grid(axis="y", alpha=0.3)
		ax.legend(fontsize=8)
	for k in range(len(idx), len(axes)):
		axes[k].set_visible(False)
	fig.suptitle("Mixture vs True with Top-2 Expert Outputs", fontsize=12, y=0.99)
	fig.tight_layout(rect=[0, 0, 1, 0.97])
	os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
	fig.savefig(save_path, dpi=250, bbox_inches="tight")
	plt.close(fig)
	print(f"[MoE] 样本分解图已保存: {save_path}")


# ---------------------- 可视化与不确定性保存辅助函数 ----------------------

def plot_uncertainty(
	P_mean: np.ndarray,
	P_low: np.ndarray,
	P_high: np.ndarray,
	P_test: np.ndarray | None,
	out_dir: str,
	top_k: int = 5,
	prefix: str = "moe_bootstrap",
) -> None:
	"""\
	将 Bootstrap 的分布预测不确定性可视化并保存到 out_dir。

	输出两类图：
	1) 测试集整体平均分布：mean 及 95% CI（逐桶误差条）
	2) 测试集中不确定性最大的 top_k 个样本：每个样本一张图

	说明：
	- P_mean/P_low/P_high 均为 shape (n_test, 7)
	- 若提供 P_test，则在图中叠加真实分布（虚线）便于对照
	"""
	os.makedirs(out_dir, exist_ok=True)

	# ---------------- 1) 整体平均分布（跨样本平均） ----------------
	mean_all = P_mean.mean(axis=0)
	low_all = P_low.mean(axis=0)
	high_all = P_high.mean(axis=0)

	# 使用不对称误差条：yerr = [mean-low, high-mean]
	yerr_all = np.vstack([np.maximum(0, mean_all - low_all), np.maximum(0, high_all - mean_all)])

	x = np.arange(len(DIST_COLS))
	fig = plt.figure(figsize=(10, 4))
	ax = fig.add_subplot(111)
	ax.errorbar(x, mean_all, yerr=yerr_all, fmt="o", capsize=4)
	ax.set_xticks(x)
	ax.set_xticklabels(DIST_COLS, rotation=30, ha="right")
	ax.set_ylim(0.0, max(0.35, float(high_all.max()) * 1.2))
	ax.set_title("Bootstrap mean distribution with CI (overall average)")
	ax.set_ylabel("Probability")
	ax.grid(True, axis="y", alpha=0.3)
	fig.tight_layout()
	fig.savefig(os.path.join(out_dir, f"{prefix}_overall_uncertainty.png"), dpi=200)
	plt.close(fig)

	# ---------------- 2) Top-k 不确定性样本（按 std 的均值排序） ----------------
	# 近似不确定性：CI 宽度均值（逐桶）
	ci_width = (P_high - P_low)
	unc_score = ci_width.mean(axis=1)  # (n_test,)
	idx_sorted = np.argsort(-unc_score)

	n_show = int(min(top_k, P_mean.shape[0]))
	for rank in range(n_show):
		i = int(idx_sorted[rank])
		m = P_mean[i]
		lo = P_low[i]
		hi = P_high[i]
		yerr = np.vstack([np.maximum(0, m - lo), np.maximum(0, hi - m)])

		fig = plt.figure(figsize=(10, 4))
		ax = fig.add_subplot(111)
		ax.errorbar(x, m, yerr=yerr, fmt="o", capsize=4, label="pred mean (CI)")
		if P_test is not None:
			ax.plot(x, P_test[i], linestyle="--", marker="x", label="true")
		ax.set_xticks(x)
		ax.set_xticklabels(DIST_COLS, rotation=30, ha="right")
		ax.set_ylim(0.0, max(0.35, float(hi.max()) * 1.2))
		ax.set_title(f"Top-{rank+1} uncertainty sample (index={i}, score={unc_score[i]:.4f})")
		ax.set_ylabel("Probability")
		ax.grid(True, axis="y", alpha=0.3)
		ax.legend(loc="best")
		fig.tight_layout()
		fig.savefig(os.path.join(out_dir, f"{prefix}_sample_{rank+1}_idx_{i}.png"), dpi=200)
		plt.close(fig)


def save_uncertainty_arrays(
	P_std: np.ndarray,
	out_path: str,
) -> None:
	"""将不确定性矩阵（std）保存为 npy，便于后续分析/画图。"""
	np.save(out_path, P_std.astype(np.float32))
	print(f"[Bootstrap] 不确定性数组已保存: {out_path}")


def compute_confidence_scores(P_std: np.ndarray) -> dict:
	"""给报告用的“全局置信度/稳定性”统计（不改变模型，只做后处理）。"""
	per_sample_std_mean = P_std.mean(axis=1)
	confidence_score = 1.0 / (per_sample_std_mean + 1e-6)
	return {
		"per_sample_std_mean": per_sample_std_mean,
		"confidence_score": confidence_score,
		"confidence_score_mean": float(confidence_score.mean()),
		"confidence_score_std": float(confidence_score.std()),
		"confidence_score_min": float(confidence_score.min()),
		"confidence_score_max": float(confidence_score.max()),
	}


def write_bootstrap_report(path: str, base_metrics: dict, bootstrap_metrics: dict, extra: dict) -> None:
	report = {
		"model": "MoE + MLP + Softmax",
		"bootstrap": {
			"B": BOOTSTRAP_B,
			"epoch_scale": BOOTSTRAP_EPOCH_SCALE,
			"ci_level": BOOTSTRAP_CI_LEVEL,
		},
		"config": {
			"num_experts": NUM_EXPERTS,
			"hidden_size": HIDDEN_SIZE,
			"top_k": TOP_K,
			"aux_coef": AUX_COEF,
			"lr": LR,
			"weight_decay": WEIGHT_DECAY,
			"max_epochs": MAX_EPOCHS,
			"patience": PATIENCE,
			"weight_mode": WEIGHT_MODE,
		},
		"metrics": {
			"single_model": base_metrics,
			"bootstrap_mean": bootstrap_metrics,
		},
		"uncertainty": extra,
	}
	with open(path, "w", encoding="utf-8") as f:
		json.dump(report, f, ensure_ascii=False, indent=2)
	print(f"[Bootstrap] 报告已保存: {path}")


def main():
	"""
	主函数：执行完整的 MoE 模型训练、评估和 Bootstrap 不确定性估计流程。
	
	执行步骤:
	1. 加载并划分数据集
	2. （可选）执行专家数量与 TOP-K 网格搜索
	3. 使用最优配置训练单模型
	4. 生成各种可视化图表
	5. 执行 Bootstrap 不确定性估计
	6. 保存所有结果和报告
	"""
	global NUM_EXPERTS, HIDDEN_SIZE, TOP_K, AUX_COEF, EXPERT_DIVERSITY_COEF
	
	set_seed()
	print(f"\n{'='*70}")
	print(f"MoE + Bootstrap 分布预测脚本")
	print(f"{'='*70}")
	print(f"设备: {DEVICE}")
	print(f"数据路径: {DATA_PATH}")
	print(f"初始 MoE 配置: num_experts={NUM_EXPERTS}, hidden_size={HIDDEN_SIZE}, k={TOP_K}")
	print(f"Bootstrap 配置: B={BOOTSTRAP_B}, epoch_scale={BOOTSTRAP_EPOCH_SCALE}, ci={BOOTSTRAP_CI_LEVEL:.0%}")
	print(f"{'='*70}\n")

	# 加载数据
	(
		X_train,
		X_val,
		X_test,
		P_train,
		P_val,
		P_test,
		N_train,
		N_val,
		N_test,
		holdout_pack,
	) = load_and_split_data()

	print(f"训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
	print(f"验证集: {X_val.shape[0]} 样本")
	print(f"测试集: {X_test.shape[0]} 样本")

	# ======================================================================
	# 0) （可选）先做一次轻量调参，找到更“分化”的正则系数
	# ======================================================================
	if ENABLE_SPECIALIZATION_SEARCH:
		print("\n[Search] 启用专家分化调参（轻量）...")
		search = specialization_search(
			X_train=X_train,
			P_train=P_train,
			N_train=N_train,
			X_val=X_val,
			P_val=P_val,
			N_val=N_val,
			output_dir=OUTPUT_DIR,
		)
		best = search.get("best", {})
		if best:
			AUX_COEF = float(best.get("aux_coef", AUX_COEF))
			EXPERT_DIVERSITY_COEF = float(best.get("div_coef", EXPERT_DIVERSITY_COEF))
			print(
				f"[Search] 采用 best: aux_coef={AUX_COEF} div_coef={EXPERT_DIVERSITY_COEF} "
				f"(val_loss={best.get('best_val_loss')}, js={best.get('expert_js_sep')})"
			)

	# ======================================================================
	# 0.5) （可选）专家数量/TOP-K 网格搜索
	# ======================================================================
	# 注意：NUM_EXPERTS, TOP_K, HIDDEN_SIZE 已在函数开头声明为 global
	if ENABLE_EXPERT_TOPK_SEARCH:
		print("\n[Grid Search] 启用专家数量/TOP-K/隐藏层大小网格搜索...")
		grid_results = expert_topk_grid_search(
			X_train=X_train,
			P_train=P_train,
			N_train=N_train,
			X_val=X_val,
			P_val=P_val,
			N_val=N_val,
			X_test=X_test,
			P_test=P_test,
			output_dir=OUTPUT_DIR,
		)
		
		# 更新全局超参数为网格搜索找到的最优值
		if grid_results.get("best"):
			best_config = grid_results["best"]
			NUM_EXPERTS = int(best_config.get("num_experts", NUM_EXPERTS))
			TOP_K = int(best_config.get("top_k", TOP_K))
			HIDDEN_SIZE = int(best_config.get("hidden_size", HIDDEN_SIZE))
			print(f"[Grid Search] 采用最优配置: num_experts={NUM_EXPERTS}, top_k={TOP_K}, hidden={HIDDEN_SIZE}")
			print(f"  -> 最优 MAE: {best_config.get('mae', 'N/A'):.4f}")
			print(f"  -> 最优 JS: {best_config.get('js_mean', 'N/A'):.4f}")
		# 注意：plot_grid_search_results 已在 expert_topk_grid_search 内部调用

	# ======================================================================
	# 1) 使用最优配置训练单模型 + 评估
	# ======================================================================
	print(f"\n[训练] 使用最终配置训练模型: num_experts={NUM_EXPERTS}, top_k={TOP_K}, hidden={HIDDEN_SIZE}")
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

	model, info = train_moe(X_train, P_train, X_val, P_val, Wtr, Wva)
	P_pred, base_metrics = evaluate(model, X_test, P_test)
	save_predictions(P_pred, os.path.join(OUTPUT_DIR, "moe_softmax_pred_output.csv"))

	# =========================
	# Holdout（eerie）单模型预测
	# =========================
	if holdout_pack is not None and holdout_pack.get("X") is not None:
		Xh = holdout_pack["X"]
		words = holdout_pack.get("word", [HOLDOUT_WORD])
		P_true_h = holdout_pack.get("P_true")
		model.eval()
		with torch.no_grad():
			Xho = torch.tensor(Xh, device=DEVICE)
			P_single_h, _ = model(Xho)
			P_single_h = P_single_h.cpu().numpy()

		# 单模型预测保存（便于查表）
		df_single = pd.DataFrame(P_single_h, columns=[f"single_{c}" for c in DIST_COLS])
		df_single.insert(0, "word", words)
		single_path = os.path.join(OUTPUT_DIR, f"holdout_{HOLDOUT_WORD}_single_prediction.csv")
		df_single.to_csv(single_path, index=False)
		print(f"[Holdout] 单模型预测已保存: {single_path}")

	# ===== 单模型作图（仿造 Moe_Softmax.py） =====
	plot_training_history(
		train_losses=info["train_losses"],
		val_losses=info["val_losses"],
		bad=info["bad"],
		best_val_loss=info["best_val_loss"],
		save_path=os.path.join(OUTPUT_DIR, "moe_training_history.png"),
	)

	plot_random_sample_distributions(
		P_test=P_test,
		P_pred=P_pred,
		sample_size=10,
		save_path=os.path.join(OUTPUT_DIR, "moe_distribution_comparison.png"),
	)

	error_stats = plot_error_analysis(
		P_test=P_test,
		P_pred=P_pred,
		save_path=os.path.join(OUTPUT_DIR, "moe_error_analysis.png"),
	)

	# 新增：性能指标可视化
	plot_performance_metrics(
		P_test=P_test,
		P_pred=P_pred,
		save_path=os.path.join(OUTPUT_DIR, "moe_performance_metrics.png"),
	)

	# 专家使用率（门控统计）
	expert_stats = analyze_expert_usage(
		model=model,
		X_data=X_test,
		save_path=os.path.join(OUTPUT_DIR, "moe_expert_usage.png"),
	)

	# 新增：辅助损失曲线
	plot_aux_loss_curve(
		aux_losses=info["aux_losses"],
		save_path=os.path.join(OUTPUT_DIR, "moe_aux_loss.png"),
	)

	# 专家分布解释：按门控分组统计每个专家的真实/预测均值分布
	explain_expert_distributions(
		model=model,
		X_data=X_test,
		P_true=P_test,
		P_pred=P_pred,
		output_dir=OUTPUT_DIR,
		prefix="test",
	)

	# 新增：更细粒度的专家差异可视化
	y_experts, gates, y_mix = compute_expert_outputs(model, X_test)
	plot_expert_gate_heatmap(
		gates=gates,
		save_path=os.path.join(OUTPUT_DIR, "moe_expert_gate_heatmap.png"),
		max_samples=500,
	)
	plot_expert_mae_heatmap(
		P_true=P_test,
		y_experts=y_experts,
		gates=gates,
		save_path=os.path.join(OUTPUT_DIR, "moe_expert_mae_heatmap.png"),
	)
	plot_sample_expert_decomposition(
		P_true=P_test,
		y_experts=y_experts,
		gates=gates,
		y_mix=y_mix,
		save_path=os.path.join(OUTPUT_DIR, "moe_expert_decomposition_samples.png"),
		sample_size=6,
	)

	# ======================================================================
	# 2) Bootstrap 不确定性估计（新增：包一层外循环）
	# ======================================================================
	print("\n[Bootstrap] 开始不确定性估计...")
	X_holdout = None if holdout_pack is None else holdout_pack.get("X")
	boot_out = bootstrap_predict(
		X_train,
		P_train,
		N_train,
		X_val,
		P_val,
		N_val,
		X_test,
		X_holdout=X_holdout,
		B=BOOTSTRAP_B,
	)
	if isinstance(boot_out, tuple):
		P_boot, P_holdout_boot = boot_out
	else:
		P_boot = boot_out
		P_holdout_boot = None
	P_mean, P_std, P_low, P_high = bootstrap_summary(P_boot, ci_level=BOOTSTRAP_CI_LEVEL)
	print("[Bootstrap] 完成")

	# =========================
	# Holdout（eerie）bootstrap 预测 + 作图
	# =========================
	if P_holdout_boot is not None and holdout_pack is not None:
		words = holdout_pack.get("word", [HOLDOUT_WORD])
		P_true_h = holdout_pack.get("P_true")
		P_mean_h, P_std_h, P_low_h, P_high_h = bootstrap_summary(P_holdout_boot, ci_level=BOOTSTRAP_CI_LEVEL)

		holdout_csv = os.path.join(OUTPUT_DIR, f"holdout_{HOLDOUT_WORD}_predictions_with_ci.csv")
		save_holdout_predictions_with_ci(
			words=words,
			P_mean=P_mean_h,
			P_std=P_std_h,
			P_low=P_low_h,
			P_high=P_high_h,
			P_true=P_true_h,
			path=holdout_csv,
		)

		plot_holdout_bar_with_ci(
			P_mean=P_mean_h,
			P_low=P_low_h,
			P_high=P_high_h,
			P_true=P_true_h,
			word=HOLDOUT_WORD,
			save_path=os.path.join(OUTPUT_DIR, f"holdout_{HOLDOUT_WORD}_bar_ci.png"),
		)
		plot_holdout_violin(
			P_holdout_boot=P_holdout_boot,
			word=HOLDOUT_WORD,
			save_path=os.path.join(OUTPUT_DIR, f"holdout_{HOLDOUT_WORD}_violin.png"),
		)

	# 以 bootstrap 均值分布作为“点预测”，计算一套对照指标
	bootstrap_metrics = compute_metrics(P_mean, P_test)
	print("\n[Bootstrap] 使用 P_mean 的测试集评估")
	for k, v in bootstrap_metrics.items():
		print(f"  {k}: {v:.6f}")

	# 保存 bootstrap 汇总
	save_bootstrap_predictions(
		P_mean,
		P_std,
		P_low,
		P_high,
		os.path.join(OUTPUT_DIR, "moe_bootstrap_pred_summary.csv"),
	)

	# 可视化不确定性（图会保存到输出目录）
	plot_uncertainty(
		P_mean=P_mean,
		P_low=P_low,
		P_high=P_high,
		P_test=P_test,
		out_dir=OUTPUT_DIR,
		top_k=5,
		prefix="moe_bootstrap",
	)

	# 保存 std 数组，便于后续做更细粒度的不确定性分析
	save_uncertainty_arrays(
		P_std,
		os.path.join(OUTPUT_DIR, "moe_bootstrap_pred_std.npy"),
	)

	# 置信度/稳定性统计（后处理）
	conf = compute_confidence_scores(P_std)
	# numpy 数组太大，不直接塞进 json；只存统计量
	conf_report = {
		"confidence_score_mean": conf["confidence_score_mean"],
		"confidence_score_std": conf["confidence_score_std"],
		"confidence_score_min": conf["confidence_score_min"],
		"confidence_score_max": conf["confidence_score_max"],
		"per_sample_std_mean_mean": float(conf["per_sample_std_mean"].mean()),
		"per_sample_std_mean_std": float(conf["per_sample_std_mean"].std()),
	}

	write_bootstrap_report(
		os.path.join(OUTPUT_DIR, "moe_bootstrap_report.json"),
		base_metrics=base_metrics,
		bootstrap_metrics=bootstrap_metrics,
		extra=conf_report,
	)

	# ======================================================================
	# 3) 生成综合汇总图和报告
	# ======================================================================
	
	# 综合汇总图（将所有关键信息整合到一张大图中）
	plot_comprehensive_summary(
		train_losses=info["train_losses"],
		val_losses=info["val_losses"],
		bad=info["bad"],
		best_val_loss=info["best_val_loss"],
		metrics=base_metrics,
		mae_per_dim=error_stats["mae_per_dim"],
		P_test=P_test,
		P_pred=P_pred,
		expert_usage=expert_stats["expert_usage"],
		save_path=os.path.join(OUTPUT_DIR, "moe_comprehensive_summary.png"),
	)
	
	# 生成文本总结报告
	generate_summary_report(
		metrics=base_metrics,
		bootstrap_metrics=bootstrap_metrics,
		info=info,
		expert_stats=expert_stats,
		conf_report=conf_report,
		output_dir=OUTPUT_DIR,
	)

	print(f"\n{'='*70}")
	print(f"完成：Bootstrap 版 MoE 结果已输出到 {OUTPUT_DIR}/")
	print(f"{'='*70}")


def generate_summary_report(
	metrics: dict,
	bootstrap_metrics: dict,
	info: dict,
	expert_stats: dict,
	conf_report: dict,
	output_dir: str,
) -> None:
	"""
	生成综合文本总结报告。
	
	报告内容包括:
	- 模型配置
	- 训练结果
	- 测试集评估指标
	- Bootstrap 不确定性估计结果
	- 专家使用率分析
	- 输出文件列表
	"""
	report = f"""
{'='*80}
MoE (Mixture of Experts) + Bootstrap 分布预测综合报告
{'='*80}

一、模型配置
─────────────────────────────────────────────────
专家数量 (num_experts): {NUM_EXPERTS}
每个专家的隐藏层大小 (hidden_size): {HIDDEN_SIZE}
Top-K 路由 (k): {TOP_K}
辅助损失系数 (aux_coef): {AUX_COEF}
专家分化正则项 (diversity_coef): {EXPERT_DIVERSITY_COEF}
学习率 (lr): {LR}
权重衰减 (weight_decay): {WEIGHT_DECAY}
早停耐心 (patience): {PATIENCE}
样本权重模式: {WEIGHT_MODE}

二、训练结果
─────────────────────────────────────────────────
最佳验证轮次: {info.get('best_epoch')}
最佳验证损失: {info.get('best_val_loss'):.6f}
总训练轮次: {len(info.get('train_losses', []))}

三、单模型测试集评估指标
─────────────────────────────────────────────────
MAE (平均绝对误差): {metrics.get('mae', 0):.6f}
  → 7个桶的概率平均误差约 {metrics.get('mae', 0)*100:.2f}%

RMSE (均方根误差): {metrics.get('rmse', 0):.6f}

KL散度: {metrics.get('kl', 0):.6f}
  → 预测分布与真实分布的信息差异 (越小越好)

JS散度: {metrics.get('js_mean', 0):.6f}
  → 对称的分布差异度量 (0~1, 越小越好)

TV距离: {metrics.get('tv_distance', 0):.6f}
  → 总变差距离 (越小越好)

余弦相似度: {metrics.get('cos_sim', 0):.6f}
  → 分布向量的相似程度 (越接近1越好)

R²: {metrics.get('r2', 0):.6f}
  → 模型解释方差的比例 (越接近1越好)

最大误差: {metrics.get('max_error', 0):.6f}
  → 单个桶上的最大预测偏差

四、Bootstrap 不确定性估计结果
─────────────────────────────────────────────────
Bootstrap 次数: {BOOTSTRAP_B}
置信区间水平: {BOOTSTRAP_CI_LEVEL:.0%}

Bootstrap均值预测的MAE: {bootstrap_metrics.get('mae', 0):.6f}
Bootstrap均值预测的RMSE: {bootstrap_metrics.get('rmse', 0):.6f}
Bootstrap均值预测的R²: {bootstrap_metrics.get('r2', 0):.6f}

置信度统计:
  - 平均置信度得分: {conf_report.get('confidence_score_mean', 0):.4f}
  - 置信度标准差: {conf_report.get('confidence_score_std', 0):.4f}
  - 最低置信度: {conf_report.get('confidence_score_min', 0):.4f}
  - 最高置信度: {conf_report.get('confidence_score_max', 0):.4f}

五、专家使用率分析
─────────────────────────────────────────────────
"""
	for i, (usage, weight) in enumerate(zip(expert_stats['expert_usage'], expert_stats['expert_avg_weight'])):
		report += f"专家 {i}: 使用率={usage*100:.1f}%, 平均门控权重={weight:.4f}\n"

	report += f"""
六、输出文件
─────────────────────────────────────────────────
预测结果: {output_dir}/moe_softmax_pred_output.csv
训练曲线: {output_dir}/moe_training_history.png
分布对比: {output_dir}/moe_distribution_comparison.png
误差分析: {output_dir}/moe_error_analysis.png
性能指标: {output_dir}/moe_performance_metrics.png
专家使用率: {output_dir}/moe_expert_usage.png
辅助损失曲线: {output_dir}/moe_aux_loss.png
门控热力图: {output_dir}/moe_expert_gate_heatmap.png
专家MAE热力图: {output_dir}/moe_expert_mae_heatmap.png
综合汇总图: {output_dir}/moe_comprehensive_summary.png
Bootstrap汇总: {output_dir}/moe_bootstrap_pred_summary.csv
JSON报告: {output_dir}/moe_bootstrap_report.json

{'='*80}
报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
	
	report_path = os.path.join(output_dir, "moe_summary_report.txt")
	with open(report_path, "w", encoding="utf-8") as f:
		f.write(report)
	
	print(report)
	print(f"\n[MoE] 综合报告已保存: {report_path}")


if __name__ == "__main__":
	main()
