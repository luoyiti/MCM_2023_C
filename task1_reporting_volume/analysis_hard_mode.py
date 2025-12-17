import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


# =============================
# Config (EXPLAIN ONLY)
# =============================
CSV_PATH = None  # None -> auto find ./data/mcm_processed_data.csv (or parent)

# ✅ 最终解释模型：用 ARDL 选出来的 3 阶滞后
BASE_LAGS = 3

# 时间特征
USE_SPLINE_TIME = True
SPLINE_N_KNOTS = 6
SPLINE_DEGREE = 3
ADD_DOW = True
ADD_MONTH = True
STANDARDIZE_TIME = True  # v3: 标准化时间，降低数值病态

# 分组降维（解释用）
GROUP_REDUCTION_METHOD = "pca"  # "pca" or "representative"
PCA_VAR_THRESHOLD = 0.90
PCA_MAX_COMPONENTS = 5

# 共线性诊断
CORR_THRESHOLD = 0.90
VIF_MAX_FEATURES = 30
DROP_NEAR_CONSTANT = True
NEAR_CONSTANT_VAR = 1e-12

# 稳健标准误
HAC_LAGS = 1

# ARDL lag 选择（仅用于报告，不影响最终 lag=3）
ARDL_MAX_LAG = 7

# 输出
TOP_PRINT = 12

# 画图开关
MAKE_PLOTS = True
BINS = 10  # 分位数分箱数
PLOT_TOP_COEF = 12
BINNED_FEATURES = ["autoencoder_value", "Zipf-value", "has_double_letter", "rl_PC3"]


# =============================
# IO + preprocessing
# =============================
def find_csv_path() -> Path:
    if CSV_PATH:
        p = Path(CSV_PATH)
        if p.exists():
            return p
        raise FileNotFoundError(f"CSV_PATH not found: {CSV_PATH}")

    candidates = [
        Path(__file__).parent / "data" / "mcm_processed_data.csv",
        Path(__file__).parent.parent / "data" / "mcm_processed_data.csv",
        Path.cwd() / "data" / "mcm_processed_data.csv",
        Path.cwd() / "mcm_processed_data.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Cannot locate mcm_processed_data.csv. Put it under ./data/ or set CSV_PATH."
    )


def parse_date_series(df: pd.DataFrame) -> pd.Series | None:
    if "date" not in df.columns:
        return None
    d = pd.to_datetime(df["date"], errors="coerce", infer_datetime_format=True)
    if d.notna().sum() == 0:
        return None
    return d


def ensure_time_order(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    d = parse_date_series(df2)
    if d is not None:
        df2["__date"] = d
        df2 = df2.sort_values("__date").drop(columns="__date").reset_index(drop=True)
        return df2
    if "contest_number" in df2.columns:
        c = pd.to_numeric(df2["contest_number"], errors="coerce")
        if c.notna().any():
            df2["__c"] = c
            df2 = df2.sort_values("__c").drop(columns="__c").reset_index(drop=True)
            return df2
    return df2.reset_index(drop=True)


def safe_clean(*arrays):
    """
    对齐并删除含 NaN/Inf 的行。返回与第一个数组同 index 的清洗结果。
    """
    cleaned = []
    for a in arrays:
        if isinstance(a, pd.DataFrame):
            a = a.replace([np.inf, -np.inf], np.nan).copy()
            for c in a.columns:
                a[c] = pd.to_numeric(a[c], errors="coerce")
            a = a.dropna(axis=1, how="all")
        elif isinstance(a, pd.Series):
            a = a.replace([np.inf, -np.inf], np.nan).copy()
            a = pd.to_numeric(a, errors="coerce")
        cleaned.append(a)

    idx = cleaned[0].index
    valid = pd.Series(True, index=idx)
    for a in cleaned:
        if isinstance(a, pd.DataFrame):
            valid &= ~a.isna().any(axis=1)
        else:
            valid &= ~a.isna()

    return [a.loc[valid].copy() for a in cleaned]


def create_lags(df: pd.DataFrame, col: str, n_lags: int):
    df2 = df.copy()
    lag_cols = []
    for k in range(1, n_lags + 1):
        name = f"{col}_lag{k}"
        df2[name] = df2[col].shift(k)
        lag_cols.append(name)
    df2 = df2.dropna(subset=lag_cols + [col])
    return df2, lag_cols


def drop_near_constant_cols(X: pd.DataFrame) -> pd.DataFrame:
    if not DROP_NEAR_CONSTANT:
        return X
    var = X.var(numeric_only=True)
    keep = var[var > NEAR_CONSTANT_VAR].index.tolist()
    return X[keep].copy()


# =============================
# Time feature engineering
# =============================
def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    d = parse_date_series(df)

    if d is None:
        if "contest_number" in df.columns:
            t_raw = pd.to_numeric(df["contest_number"], errors="coerce").astype(float)
        else:
            t_raw = pd.Series(np.arange(len(df)), index=df.index).astype(float)
    else:
        t_raw = d.map(lambda x: x.toordinal()).astype(float)

    if STANDARDIZE_TIME:
        mu = float(np.nanmean(t_raw.values))
        sd = float(np.nanstd(t_raw.values))
        sd = sd if sd > 0 else 1.0
        t = (t_raw - mu) / sd
    else:
        t = t_raw

    out["t_linear"] = t.astype(float)

    if USE_SPLINE_TIME:
        st = SplineTransformer(
            n_knots=SPLINE_N_KNOTS,
            degree=SPLINE_DEGREE,
            include_bias=False
        )
        B = st.fit_transform(t.values.reshape(-1, 1))
        for j in range(B.shape[1]):
            out[f"t_spline{j+1}"] = B[:, j]

    if d is not None:
        if ADD_DOW:
            dow = d.dt.dayofweek
            out = pd.concat([out, pd.get_dummies(dow, prefix="dow", drop_first=True)], axis=1)
        if ADD_MONTH:
            m = d.dt.month
            out = pd.concat([out, pd.get_dummies(m, prefix="month", drop_first=True)], axis=1)

    return out


# =============================
# Group feature reduction
# =============================
def define_groups(columns: list[str]) -> dict[str, list[str]]:
    groups = {"rl": [], "simulate": [], "entropy": [], "other": []}
    for c in columns:
        lc = c.lower()
        if lc.startswith("rl_"):
            groups["rl"].append(c)
        elif "simulate" in lc:
            groups["simulate"].append(c)
        elif "entropy" in lc:
            groups["entropy"].append(c)
        else:
            groups["other"].append(c)
    return groups


class GroupReducer:
    """
    Fit on training attributes, then transform any new attributes.
    """
    def __init__(self):
        self.groups_ = None
        self.other_cols_ = None
        self.models_ = {}
        self.infos_ = []

    def fit(self, X_attr: pd.DataFrame, y: pd.Series):
        X_attr = X_attr.copy()
        self.groups_ = define_groups(list(X_attr.columns))
        self.other_cols_ = self.groups_.get("other", [])

        for g in ["rl", "simulate", "entropy"]:
            cols = self.groups_.get(g, [])
            if len(cols) == 0:
                continue
            Xg = X_attr[cols].astype(float)

            if GROUP_REDUCTION_METHOD == "pca" and len(cols) >= 2:
                scaler = StandardScaler()
                Z = scaler.fit_transform(Xg.values)

                pca_full = PCA(random_state=42).fit(Z)
                cum = np.cumsum(pca_full.explained_variance_ratio_)
                k = int(np.searchsorted(cum, PCA_VAR_THRESHOLD) + 1)
                k = max(1, min(k, PCA_MAX_COMPONENTS, Xg.shape[1]))

                pca = PCA(n_components=k, random_state=42).fit(Z)
                self.models_[g] = {"type": "pca", "cols": cols, "scaler": scaler, "pca": pca, "k": k}
                self.infos_.append({
                    "group": g, "method": "pca",
                    "orig_dim": int(Xg.shape[1]),
                    "k": int(k),
                    "cum_var": float(np.sum(pca.explained_variance_ratio_))
                })
            else:
                cors = {}
                yy = y.values
                for c in cols:
                    v = pd.to_numeric(Xg[c], errors="coerce").values.astype(float)
                    if np.nanstd(v) == 0:
                        cors[c] = 0.0
                    else:
                        cors[c] = float(np.corrcoef(v, yy)[0, 1])
                best = max(cors.keys(), key=lambda k: abs(cors[k]))
                self.models_[g] = {"type": "repr", "cols": cols, "best": best, "corr": cors[best]}
                self.infos_.append({
                    "group": g, "method": "repr",
                    "orig_dim": int(len(cols)),
                    "picked": best,
                    "corr": float(cors[best])
                })
        return self

    def transform(self, X_attr: pd.DataFrame) -> pd.DataFrame:
        X_attr = X_attr.copy()
        pieces = []

        for g, meta in self.models_.items():
            if meta["type"] == "pca":
                cols = meta["cols"]
                Xg = X_attr[cols].astype(float)
                Z = meta["scaler"].transform(Xg.values)
                comps = meta["pca"].transform(Z)
                k = meta["k"]
                out_cols = [f"{g}_PC{i+1}" for i in range(k)]
                pieces.append(pd.DataFrame(comps, columns=out_cols, index=X_attr.index))
            else:
                best = meta["best"]
                pieces.append(X_attr[[best]].rename(columns={best: f"{g}_repr__{best}"}))

        if self.other_cols_:
            keep_other = [c for c in self.other_cols_ if c in X_attr.columns]
            pieces.append(X_attr[keep_other].copy())

        return pd.concat(pieces, axis=1)

    def fit_transform(self, X_attr: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X_attr, y)
        return self.transform(X_attr)


# =============================
# Diagnostics
# =============================
def high_corr_pairs(X: pd.DataFrame, threshold=0.9, top_k=50):
    corr = X.corr(numeric_only=True)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feat1", "level_1": "feat2", 0: "corr"})
    )
    pairs["abs_corr"] = pairs["corr"].abs()
    pairs = pairs[pairs["abs_corr"] >= threshold].sort_values("abs_corr", ascending=False)
    return pairs.head(top_k), corr


def condition_number(X: pd.DataFrame) -> float:
    Xn = X.replace([np.inf, -np.inf], np.nan).dropna()
    Xc = sm.add_constant(Xn)
    return float(np.linalg.cond(Xc.values))


def compute_vif(X: pd.DataFrame, max_features=30) -> pd.Series:
    Xn = X.replace([np.inf, -np.inf], np.nan).dropna()
    if Xn.shape[1] > max_features:
        Xn = Xn.iloc[:, :max_features]
    Xc = sm.add_constant(Xn)
    vifs = []
    for i in range(1, Xc.shape[1]):
        vifs.append(variance_inflation_factor(Xc.values, i))
    return pd.Series(vifs, index=Xn.columns).sort_values(ascending=False)


# =============================
# Models
# =============================
def fit_dynamic_ols_hac(y, X):
    Xc = sm.add_constant(X)
    return sm.OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})


def fit_binomial_glm_counts(hard, total, X):
    fail = total - hard
    endog = np.column_stack([hard.values, fail.values])
    Xc = sm.add_constant(X)
    return sm.GLM(endog, Xc, family=sm.families.Binomial()).fit()


def fit_fractional_glm_weights(ratio, total, X):
    Xc = sm.add_constant(X)
    return sm.GLM(
        ratio.values, Xc,
        family=sm.families.Binomial(),
        freq_weights=total.values
    ).fit()


def fit_quasi_binomial_like(ratio, total, X):
    Xc = sm.add_constant(X)
    return sm.GLM(
        ratio.values, Xc,
        family=sm.families.Binomial(),
        freq_weights=total.values
    ).fit(cov_type="HC0")


def pseudo_r2_glm(glm_fit) -> float:
    try:
        return 1.0 - (glm_fit.deviance / glm_fit.null_deviance)
    except Exception:
        return np.nan


def two_stage_residual_explain(y: pd.Series, X_base: pd.DataFrame, X_attr: pd.DataFrame):
    """
    Stage 1: y ~ (lags + time controls) -> residual
    Stage 2: residual ~ (attributes) with HAC
    返回：m1, m2, resid_stage1(Series), resid_after_attr(Series), resid_r2, joint_p
    """
    X1 = X_base.copy()
    X2 = X_attr.copy()
    X1, y1, X2 = safe_clean(X1, y, X2)

    m1 = sm.OLS(y1, sm.add_constant(X1)).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    resid1 = (y1 - m1.fittedvalues)
    resid1 = pd.Series(resid1.values, index=y1.index, name="resid_stage1")

    # Stage2
    X2 = X2.select_dtypes(include=[np.number]).astype(float)
    X2 = drop_near_constant_cols(X2)

    # align & clean again (attrs might have extra NaNs)
    X2, resid1_aligned = safe_clean(X2, resid1)

    m2 = sm.OLS(resid1_aligned, sm.add_constant(X2)).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    resid_after = resid1_aligned - m2.fittedvalues
    resid_after = pd.Series(resid_after.values, index=resid1_aligned.index, name="resid_after_attr")

    # joint test (no HAC refit)
    try:
        m2_nr = sm.OLS(resid1_aligned, sm.add_constant(X2)).fit()
        R = np.eye(len(m2_nr.params))[1:]
        joint = m2_nr.f_test(R)
        joint_p = float(joint.pvalue)
    except Exception:
        joint_p = np.nan

    resid_r2 = float(m2.rsquared)
    return m1, m2, resid1, resid_after, resid_r2, joint_p


# =============================
# ARDL lag selection (report only)
# =============================
def select_ardl_lag(df, ratio_col, time_cols, X_attr_red_all: pd.DataFrame, max_lag=7):
    """
    ARDL-like: ratio_t ~ ratio_{t-1..t-k} + time + X_attr(t)
    choose lag by AIC (non-robust), report HAC for chosen.
    """
    best_aic = None
    best_lag = None
    best_fit_hac = None

    X_attr_red_all = X_attr_red_all.loc[df.index].copy()

    for k in range(1, max_lag + 1):
        dfk, lag_cols = create_lags(df, ratio_col, k)
        X = pd.concat([
            dfk[lag_cols + time_cols].copy(),
            X_attr_red_all.loc[dfk.index].copy()
        ], axis=1)
        y = dfk[ratio_col].astype(float)

        X = X.select_dtypes(include=[np.number]).astype(float)
        X = drop_near_constant_cols(X)
        X, y = safe_clean(X, y)

        if len(y) < 50 or X.shape[1] == 0:
            continue

        fit_nr = sm.OLS(y, sm.add_constant(X)).fit()
        aic = fit_nr.aic

        if (best_aic is None) or (aic < best_aic):
            best_aic = aic
            best_lag = k
            best_fit_hac = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})

    return best_lag, best_fit_hac


def print_pca_loadings(reducer, group="simulate", topn=12):
    meta = reducer.models_.get(group)
    if (meta is None) or (meta.get("type") != "pca"):
        print(f"[{group}] no PCA model found.")
        return

    cols = meta["cols"]
    pca = meta["pca"]

    loadings = pd.DataFrame(
        pca.components_.T,
        index=cols,
        columns=[f"{group}_PC{i+1}" for i in range(pca.n_components_)]
    )

    print(f"\n[{group}] PCA explained variance ratio:")
    evr = pd.Series(pca.explained_variance_ratio_, index=loadings.columns)
    print(evr.round(4).to_string())

    for pc in loadings.columns:
        s = loadings[pc].sort_values(key=lambda x: x.abs(), ascending=False).head(topn)
        print(f"\nTop loadings for {pc} (by |loading|):")
        print(s.to_string())


# =============================
# Plotting helpers
# =============================
def _outdir():
    d = Path(__file__).resolve().parent.parent / "pictures" / "task1"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _time_x(df1: pd.DataFrame):
    # 优先 date；其次 contest_number；否则 index
    if "date" in df1.columns:
        d = pd.to_datetime(df1["date"], errors="coerce")
        if d.notna().any():
            return d
    if "contest_number" in df1.columns:
        c = pd.to_numeric(df1["contest_number"], errors="coerce")
        if c.notna().any():
            return c
    return pd.Series(np.arange(len(df1)), index=df1.index)


def plot_actual_vs_fitted(df1, y, ols_base, ols_full, fname="fig1_actual_vs_fitted.png"):
    x = _time_x(df1.loc[y.index])
    plt.figure(figsize=(10, 4), dpi=200)
    plt.plot(x, y.values, label="Actual")
    plt.plot(x, ols_base.fittedvalues.values, label="Baseline fitted")
    plt.plot(x, ols_full.fittedvalues.values, label="Full fitted")
    plt.title("hard_mode_ratio: actual vs fitted")
    plt.xlabel("time")
    plt.ylabel("hard_mode_ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_outdir() / fname, bbox_inches="tight")
    plt.close()


def plot_residual_before_after(df1, resid_base, resid_after_attr, fname="fig2_residual_before_after.png"):
    # 两个残差可能 index 不完全一致 -> 用交集
    idx = resid_base.index.intersection(resid_after_attr.index)
    x = _time_x(df1.loc[idx])
    plt.figure(figsize=(10, 4), dpi=200)
    plt.plot(x, resid_base.loc[idx].values, label="Baseline residual")
    plt.plot(x, resid_after_attr.loc[idx].values, label="After attributes (remaining residual)")
    plt.axhline(0, linewidth=1)
    plt.title("Residuals: before vs after attributes")
    plt.xlabel("time")
    plt.ylabel("residual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_outdir() / fname, bbox_inches="tight")
    plt.close()


def plot_coef_forest(m2, top=12, fname="fig3_stage2_coef_forest.png"):
    params = m2.params.drop(labels=["const"], errors="ignore")
    bse = m2.bse.drop(labels=["const"], errors="ignore")
    pvals = m2.pvalues.drop(labels=["const"], errors="ignore")

    if len(pvals) == 0:
        return

    idx = pvals.sort_values().head(top).index
    coef = params.loc[idx]
    se = bse.loc[idx]
    ci_low = coef - 1.96 * se
    ci_high = coef + 1.96 * se

    plt.figure(figsize=(8, 5), dpi=200)
    y_pos = np.arange(len(idx))
    plt.errorbar(
        coef.values,
        y_pos,
        xerr=[(coef - ci_low).values, (ci_high - coef).values],
        fmt="o"
    )
    plt.yticks(y_pos, idx)
    plt.axvline(0, linewidth=1)
    plt.title("Stage2 (attrs on residual): coefficients with 95% CI")
    plt.xlabel("coefficient")
    plt.tight_layout()
    plt.savefig(_outdir() / fname, bbox_inches="tight")
    plt.close()


def plot_binned_residual(X_attr, resid_base, feature, bins=10, fname=None):
    if feature not in X_attr.columns:
        return
    s = pd.Series(X_attr[feature]).astype(float)
    r = pd.Series(resid_base).astype(float)

    tmp = pd.DataFrame({"x": s, "r": r}).dropna()
    if len(tmp) < 30:
        return

    tmp["bin"] = pd.qcut(tmp["x"], q=bins, duplicates="drop")
    g = tmp.groupby("bin")["r"]
    mean = g.mean()
    se = g.sem()

    plt.figure(figsize=(8, 4), dpi=200)
    x = np.arange(len(mean))
    plt.errorbar(x, mean.values, yerr=1.96 * se.values, fmt="o")
    plt.axhline(0, linewidth=1)
    plt.title(f"Binned baseline residual vs {feature}")
    plt.xlabel("quantile bin")
    plt.ylabel("mean residual (±95% CI)")
    plt.tight_layout()

    if fname is None:
        safe = feature.replace("/", "_").replace(" ", "_")
        fname = f"figA_binned_resid_{safe}.png"
    plt.savefig(_outdir() / fname, bbox_inches="tight")
    plt.close()


# =============================
# Main (EXPLANATION-FOCUSED)
# =============================
def main():
    print("\n" + "=" * 100)
    print("EXPLANATION MODEL v3 (FINAL): lag1~lag3 + time trend removed, then attributes explain residuals")
    print("=" * 100)

    # Load
    csv_path = find_csv_path()
    print(f"📂 Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    df = ensure_time_order(df)

    # Target
    df["hard_mode_ratio"] = df["number_in_hard_mode"] / df["number_of_reported_results"]

    # Time controls
    time_df = build_time_features(df)
    for c in time_df.columns:
        df[c] = time_df[c]
    time_cols = list(time_df.columns)

    # Attribute columns
    exclude = {
        "date", "contest_number", "word",
        "number_of_reported_results", "number_in_hard_mode",
        "1_try", "2_tries", "3_tries", "4_tries", "5_tries", "6_tries",
        "7_or_more_tries_x", "7_or_more_tries", "sum",
        "hard_mode_ratio",
        *time_cols
    }
    feature_cols = [c for c in df.columns if c not in exclude]

    print(f"✓ Rows: {len(df)}")
    print(f"✓ Time controls: {len(time_cols)} -> {time_cols[:6]}{'...' if len(time_cols)>6 else ''}")
    print(f"✓ Raw attribute features: {len(feature_cols)}")

    # ✅ lag=3 (final explain model)
    df1, lag_cols = create_lags(df, "hard_mode_ratio", BASE_LAGS)
    df1 = df1.dropna(subset=["hard_mode_ratio"] + lag_cols + time_cols).reset_index(drop=True)

    y = df1["hard_mode_ratio"].astype(float)
    total = df1["number_of_reported_results"].astype(float)
    hard = df1["number_in_hard_mode"].astype(float)

    # Baseline X: AR(3) + time controls
    X_base = df1[lag_cols + time_cols].astype(float)

    # Raw attributes
    X_attr_raw = df1[feature_cols].copy()
    X_base, y, X_attr_raw, total, hard = safe_clean(X_base, y, X_attr_raw, total, hard)

    # Group reduction (fit on df1)
    reducer = GroupReducer()
    X_attr_red = reducer.fit_transform(X_attr_raw, y)

    print("\n[Group reduction]")
    if reducer.infos_:
        for info in reducer.infos_:
            if info["method"] == "pca":
                print(f" - {info['group']}: {info['orig_dim']} -> {info['k']} comps, cum_var={info['cum_var']:.3f}")
            else:
                print(f" - {info['group']}: {info['orig_dim']} -> 1 repr ({info['picked']}), corr={info['corr']:.3f}")
    else:
        print(" - (no grouped features found; kept all attributes)")

    # PCA loadings
    print("\n" + "-" * 100)
    print("[PCA loadings] interpret PCs by top contributing raw features")
    print("-" * 100)
    print_pca_loadings(reducer, group="simulate", topn=12)
    print_pca_loadings(reducer, group="rl", topn=12)
    print_pca_loadings(reducer, group="entropy", topn=12)

    # Full model matrix
    X_full = pd.concat([X_base, X_attr_red], axis=1)
    X_full = drop_near_constant_cols(X_full)

    # Drop highly correlated attributes (only in attributes block)
    attr_only = X_full.drop(columns=lag_cols + time_cols, errors="ignore")
    pairs, _ = high_corr_pairs(attr_only, threshold=CORR_THRESHOLD)
    to_drop = set(pairs["feat2"].tolist()) if len(pairs) else set()
    X_full_red = X_full.drop(columns=[c for c in to_drop if c in X_full.columns], errors="ignore")

    print(f"\n✓ After lag+clean: {len(y)} rows")
    print(f"✓ Full dims: {X_full.shape[1]} -> {X_full_red.shape[1]} (dropped {len(to_drop)} high-corr cols)")

    # Diagnostics
    print("\n" + "-" * 100)
    print("[0] Collinearity diagnostics (after time standardization)")
    print("-" * 100)
    cn = condition_number(X_full_red)
    print(f"Condition number: {cn:.2e}")
    try:
        vif = compute_vif(
            X_full_red.drop(columns=lag_cols + time_cols, errors="ignore"),
            max_features=VIF_MAX_FEATURES
        )
        print("Top VIF (subset):")
        print(vif.head(15))
    except Exception as e:
        print(f"VIF skipped: {e}")

    # Prepare reduced attrs for ALL df rows (for ARDL report)
    X_attr_all = df[feature_cols].copy()
    for c in X_attr_all.columns:
        X_attr_all[c] = pd.to_numeric(X_attr_all[c], errors="coerce")
    X_attr_all = X_attr_all.replace([np.inf, -np.inf], np.nan)
    X_attr_red_all = reducer.transform(X_attr_all).copy()

    # [1] ARDL lag selection (report only)
    print("\n" + "-" * 100)
    print("[1] ARDL (multi-lag) using REDUCED attributes (report only)")
    print("-" * 100)
    best_lag, best_fit = select_ardl_lag(df, "hard_mode_ratio", time_cols, X_attr_red_all, max_lag=ARDL_MAX_LAG)
    if best_fit is None:
        print("ARDL selection failed (not enough data after lags).")
    else:
        print(f"Best ARDL lag (by AIC): {best_lag}  (final model uses BASE_LAGS={BASE_LAGS})")
        tbl = pd.DataFrame({"coef": best_fit.params, "p(HAC)": best_fit.pvalues}).sort_values("p(HAC)")
        print(tbl.head(TOP_PRINT))

    # [2] Dynamic OLS(HAC): Baseline vs Full
    print("\n" + "-" * 100)
    print("[2] Dynamic OLS(HAC): Baseline (lags+time) vs Full (+attributes_reduced)")
    print("-" * 100)
    ols_base = fit_dynamic_ols_hac(y, X_base)
    ols_full = fit_dynamic_ols_hac(y, X_full_red)

    print(f"Baseline: Adj.R²={ols_base.rsquared_adj:.4f}  AIC={ols_base.aic:.2f}  BIC={ols_base.bic:.2f}")
    print(f"Full    : Adj.R²={ols_full.rsquared_adj:.4f}  AIC={ols_full.aic:.2f}  BIC={ols_full.bic:.2f}")
    print(f"ΔAdj.R² : {ols_full.rsquared_adj - ols_base.rsquared_adj:+.4f}")

    try:
        nr_base = sm.OLS(y, sm.add_constant(X_base)).fit()
        nr_full = sm.OLS(y, sm.add_constant(X_full_red)).fit()
        an = anova_lm(nr_base, nr_full)
        p = float(an["Pr(>F)"].iloc[1])
        print(f"Nested F-test (no-HAC refit): p={p:.4g}")
    except Exception as e:
        print(f"Nested F-test skipped: {e}")

    top_ols = pd.DataFrame({"coef": ols_full.params, "p(HAC)": ols_full.pvalues}).sort_values("p(HAC)")
    print("\nTop OLS(full) coefficients:")
    print(top_ols.head(TOP_PRINT))

    # [2b] FINAL explanation model
    print("\n" + "-" * 100)
    print("[2b] FINAL: Residual explanation (remove lags+time, test attributes on residual)")
    print("-" * 100)
    X_attr_only = X_full_red.drop(columns=lag_cols + time_cols, errors="ignore")
    m1, m2, resid_base, resid_after_attr, resid_r2, joint_p = two_stage_residual_explain(y, X_base, X_attr_only)

    print(f"Stage1 (baseline) Adj.R²={m1.rsquared_adj:.4f}")
    print(f"Stage2 (attrs on residual) R²(resid explained)={resid_r2:.4f}  joint F p={joint_p:.4g}")

    top_m2 = pd.DataFrame({"coef": m2.params, "p(HAC)": m2.pvalues}).sort_values("p(HAC)")
    print("\nTop residual-stage coefficients (attributes):")
    print(top_m2.head(TOP_PRINT))

    # [3] Binomial GLM counts (robustness)
    print("\n" + "-" * 100)
    print("[3] Binomial GLM(counts): Baseline vs Full")
    print("-" * 100)
    glm_base = fit_binomial_glm_counts(hard, total, X_base)
    glm_full = fit_binomial_glm_counts(hard, total, X_full_red)

    print(f"Baseline: PseudoR²={pseudo_r2_glm(glm_base):.4f}  AIC={glm_base.aic:.2f}")
    print(f"Full    : PseudoR²={pseudo_r2_glm(glm_full):.4f}  AIC={glm_full.aic:.2f}")
    top_glm = pd.DataFrame({"coef": glm_full.params, "p": glm_full.pvalues}).sort_values("p")
    print("\nTop GLM(full) coefficients:")
    print(top_glm.head(TOP_PRINT))

    # [4] Fractional / Quasi-like (robustness)
    print("\n" + "-" * 100)
    print("[4] Fractional(Binomial weights) + Quasi-like robust SE")
    print("-" * 100)
    frac_base = fit_fractional_glm_weights(y, total, X_base)
    frac_full = fit_fractional_glm_weights(y, total, X_full_red)
    print(f"Fractional Baseline AIC={frac_base.aic:.2f}  PseudoR²={pseudo_r2_glm(frac_base):.4f}")
    print(f"Fractional Full     AIC={frac_full.aic:.2f}  PseudoR²={pseudo_r2_glm(frac_full):.4f}")
    q_full = fit_quasi_binomial_like(y, total, X_full_red)
    print(f"Quasi-like (robust) Full AIC={q_full.aic:.2f}  PseudoR²={pseudo_r2_glm(q_full):.4f}")

    # =============================
    # PLOTS (recommended for explanation)
    # =============================
    if MAKE_PLOTS and MAKE_PLOTS is True:
        print("\n" + "-" * 100)
        print("[PLOTS] saving figures to pictures/task1/")
        print("-" * 100)

        # 图1：实际 vs 拟合
        plot_actual_vs_fitted(df1, y, ols_base, ols_full)

        # 图2：残差 before/after
        plot_residual_before_after(df1, resid_base, resid_after_attr)

        # 图3：Stage2 系数森林图
        plot_coef_forest(m2, top=PLOT_TOP_COEF)

        # 图A：关键属性的分箱残差均值图（用 baseline 残差）
        for feat in BINNED_FEATURES:
            plot_binned_residual(X_attr_only, resid_base, feat, bins=BINS)

        print(f"Saved figures under: {str(_outdir())}")

    print("\nDONE.")


def analyze_hard_mode():
    """兼容 run_task1.py 的接口（如果你项目里用到这个函数名）"""
    main()


if __name__ == "__main__":
    main()
