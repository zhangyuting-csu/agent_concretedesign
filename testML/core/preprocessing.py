import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings('ignore')

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from core.plotting_utils import NATURE_COLORS, apply_nature_style, fig_to_base64


def _sanitize_numeric_frame(df):
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        return numeric_df
    numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    for col in numeric_df.columns:
        fill_value = numeric_df[col].median()
        if pd.isna(fill_value):
            fill_value = 0.0
        numeric_df[col] = numeric_df[col].fillna(fill_value)
    return numeric_df


def _sanitize_target(y, ptype="Classification"):
    y = pd.Series(y).replace([np.inf, -np.inf], np.nan)
    if ptype == "Classification":
        mode_vals = y.dropna().mode()
        fill_value = mode_vals.iloc[0] if not mode_vals.empty else "Missing"
        return y.fillna(fill_value)
    y = pd.to_numeric(y, errors='coerce')
    return y.fillna(y.median() if not y.dropna().empty else 0.0)

def apply_outlier_removal(df, method="Isolation Forest", contamination=0.05):
    numeric_df = _sanitize_numeric_frame(df)
    if numeric_df.empty: return df, 0
    contam = float(contamination)
    
    if method == "Isolation Forest":
        detector = IsolationForest(contamination=contam, random_state=42)
        yhat = detector.fit_predict(numeric_df)
    elif method == "Local Outlier Factor":
        detector = LocalOutlierFactor(contamination=contam)
        yhat = detector.fit_predict(numeric_df)
    elif method == "Elliptic Envelope":
        detector = EllipticEnvelope(contamination=contam, random_state=42)
        yhat = detector.fit_predict(numeric_df)
    else:
        return df, 0
        
    mask = yhat != -1
    clean_df = df[mask].reset_index(drop=True)
    return clean_df, int((~mask).sum())

def apply_scaling(df, target_col, method="StandardScaler"):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col, errors='ignore')
    if numeric_cols.empty: return df
    
    if method == "StandardScaler": scaler = StandardScaler()
    elif method == "MinMaxScaler": scaler = MinMaxScaler()
    else: scaler = RobustScaler()
    
    safe_numeric = _sanitize_numeric_frame(df[numeric_cols])
    df[numeric_cols] = scaler.fit_transform(safe_numeric)
    return df

def apply_augmentation(df, target_col, method="SMOTE", ptype="Classification"):
    if method in ["SMOTE", "ADASYN"]:
        if ptype != "Classification":
            raise ValueError("SMOTE and ADASYN require a Classification target.")
        if not HAS_IMBLEARN:
            raise ImportError("imblearn library is missing.")
            
        X = df.drop(columns=[target_col])
        y = _sanitize_target(df[target_col], ptype="Classification")
        X_num = _sanitize_numeric_frame(X)
        
        sampler = SMOTE(random_state=42) if method == "SMOTE" else ADASYN(random_state=42)
        X_res, y_res = sampler.fit_resample(X_num, y)
        new_df = pd.concat([X_res, pd.Series(y_res, name=target_col)], axis=1)
        return new_df
    elif method == "Gaussian Noise":
        X = df.drop(columns=[target_col])
        y = _sanitize_target(df[target_col], ptype=ptype)
        X_num = _sanitize_numeric_frame(X)
        
        noise = np.random.normal(0, X_num.std() * 0.05, X_num.shape)
        X_noisy = X_num + noise
        
        new_df = pd.concat([X_noisy, pd.Series(y, name=target_col)], axis=1)
        return pd.concat([df, new_df], axis=0).reset_index(drop=True)
    elif method == "Bootstrap Resample":
        sampled = df.sample(len(df), replace=True, random_state=42).reset_index(drop=True)
        return pd.concat([df, sampled], axis=0).reset_index(drop=True)
    elif method == "Mixup Numeric":
        X = df.drop(columns=[target_col])
        y = _sanitize_target(df[target_col], ptype=ptype)
        X_num = _sanitize_numeric_frame(X).copy()
        if len(X_num) < 2:
            return df
        rng = np.random.default_rng(42)
        idx_a = rng.integers(0, len(X_num), len(X_num))
        idx_b = rng.integers(0, len(X_num), len(X_num))
        lam = rng.beta(0.4, 0.4, len(X_num))
        lam_col = lam.reshape(-1, 1)
        mixed_x = X_num.to_numpy()[idx_a] * lam_col + X_num.to_numpy()[idx_b] * (1 - lam_col)
        if ptype == "Regression":
            mixed_y = y.to_numpy()[idx_a] * lam + y.to_numpy()[idx_b] * (1 - lam)
        else:
            mixed_y = y.to_numpy()[idx_a]
        mix_df = pd.DataFrame(mixed_x, columns=X_num.columns)
        mix_df[target_col] = mixed_y
        return pd.concat([df, mix_df], axis=0).reset_index(drop=True)
    else:
        return df


def _ecdf_distance(a, b):
    a = np.asarray(pd.Series(a).dropna(), dtype=float)
    b = np.asarray(pd.Series(b).dropna(), dtype=float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    grid = np.unique(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def _split_original_synthetic(df_old, df_new):
    n_old = len(df_old)
    original_like = df_new.iloc[: min(n_old, len(df_new))].copy()
    synthetic = df_new.iloc[n_old:].copy() if len(df_new) > n_old else pd.DataFrame(columns=df_new.columns)
    return original_like.reset_index(drop=True), synthetic.reset_index(drop=True)


def generate_augmentation_diagnostics(df_old, df_new, target_col, ptype="Regression", fmt="png", dpi=300):
    apply_nature_style()
    plots = {}
    original_like, synthetic = _split_original_synthetic(df_old, df_new)
    if synthetic.empty:
        synthetic = df_new.copy().reset_index(drop=True)
    old_num = _sanitize_numeric_frame(df_old.drop(columns=[target_col], errors="ignore"))
    new_num = _sanitize_numeric_frame(df_new.drop(columns=[target_col], errors="ignore"))
    syn_num = _sanitize_numeric_frame(synthetic.drop(columns=[target_col], errors="ignore"))
    common = [c for c in old_num.columns if c in new_num.columns]
    if not common:
        return plots
    common = common[: min(30, len(common))]

    # 1. Row and target balance.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].bar(["Original", "Augmented", "Synthetic"], [len(df_old), len(df_new), max(0, len(df_new) - len(df_old))], color=[NATURE_COLORS["blue"], NATURE_COLORS["teal"], NATURE_COLORS["orange"]])
    axes[0].set_ylabel("Rows")
    axes[0].set_title("Synthetic Sample Expansion")
    if target_col in df_old.columns and target_col in df_new.columns:
        if ptype == "Classification" or df_old[target_col].nunique() <= 15:
            before = df_old[target_col].value_counts(normalize=True)
            after = df_new[target_col].value_counts(normalize=True)
            target_frame = pd.DataFrame({"Original": before, "Augmented": after}).fillna(0)
            target_frame.plot(kind="bar", ax=axes[1], color=[NATURE_COLORS["blue"], NATURE_COLORS["orange"]])
            axes[1].set_ylabel("Class share")
        else:
            axes[1].hist(pd.to_numeric(df_old[target_col], errors="coerce").dropna(), bins=18, alpha=0.55, density=True, label="Original", color=NATURE_COLORS["blue"])
            axes[1].hist(pd.to_numeric(df_new[target_col], errors="coerce").dropna(), bins=18, alpha=0.45, density=True, label="Augmented", color=NATURE_COLORS["orange"])
            axes[1].legend()
            axes[1].set_ylabel("Density")
        axes[1].set_title("Target Distribution Preservation")
    plots["Synthetic Sample and Target Balance"] = fig_to_base64(fig, fmt, dpi)

    # 2. Mean shift.
    mean_shift = ((new_num[common].mean() - old_num[common].mean()) / (old_num[common].std().replace(0, 1.0) + 1e-9)).abs().sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.35 * len(mean_shift) + 1.5)))
    ax.barh(mean_shift.index, mean_shift.values, color=NATURE_COLORS["red"])
    ax.set_xlabel("|standardized mean shift|")
    ax.set_title("Synthetic Feature Mean Shift")
    plots["Synthetic Feature Mean Shift"] = fig_to_base64(fig, fmt, dpi)

    # 3. Variance ratio.
    var_ratio = (new_num[common].std() / (old_num[common].std().replace(0, np.nan))).replace([np.inf, -np.inf], np.nan).dropna().sort_values()
    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.35 * min(15, len(var_ratio)) + 1.5)))
    show = var_ratio.iloc[np.r_[0:min(7, len(var_ratio)), max(0, len(var_ratio)-8):len(var_ratio)]].drop_duplicates()
    ax.barh(show.index, show.values, color=NATURE_COLORS["purple"])
    ax.axvline(1, color=NATURE_COLORS["slate"], linestyle="--")
    ax.set_xlabel("Augmented std / original std")
    ax.set_title("Synthetic Variance Preservation")
    plots["Synthetic Variance Preservation"] = fig_to_base64(fig, fmt, dpi)

    # 4. ECDF drift.
    drift = pd.Series({col: _ecdf_distance(old_num[col], new_num[col]) for col in common}).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.35 * len(drift) + 1.5)))
    ax.barh(drift.index, drift.values, color=NATURE_COLORS["orange"])
    ax.set_xlabel("ECDF distance")
    ax.set_title("Synthetic Distribution Drift Sentinel")
    plots["Synthetic Distribution Drift Sentinel"] = fig_to_base64(fig, fmt, dpi)

    # 5. Correlation preservation.
    corr_old = old_num[common].corr().fillna(0)
    corr_new = new_num[common].corr().fillna(0)
    delta = corr_new - corr_old
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(delta.iloc[:15, :15], cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "Correlation delta"})
    ax.set_title("Synthetic Correlation Structure Delta")
    plots["Synthetic Correlation Structure Delta"] = fig_to_base64(fig, fmt, dpi)

    # 6. PCA coverage map.
    scaler = StandardScaler()
    combined = pd.concat([old_num[common], syn_num[common]], ignore_index=True).replace([np.inf, -np.inf], np.nan).fillna(0)
    coords = PCA(n_components=2, random_state=42).fit_transform(scaler.fit_transform(combined))
    labels = np.array(["Original"] * len(old_num) + ["Synthetic"] * len(syn_num))
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    for label, color in [("Original", NATURE_COLORS["blue"]), ("Synthetic", NATURE_COLORS["red"])]:
        mask = labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=32, alpha=0.65, color=color, label=label, edgecolors="white", linewidth=0.25)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Synthetic PCA Coverage Map")
    ax.legend()
    plots["Synthetic PCA Coverage Map"] = fig_to_base64(fig, fmt, dpi)

    # 7. Nearest-neighbor distance.
    if len(old_num) >= 3 and len(syn_num) >= 3:
        scaler = StandardScaler()
        old_scaled = scaler.fit_transform(old_num[common].fillna(0))
        syn_scaled = scaler.transform(syn_num[common].fillna(0))
        nn = NearestNeighbors(n_neighbors=min(5, len(old_scaled))).fit(old_scaled)
        dist, _ = nn.kneighbors(syn_scaled)
        min_dist = dist[:, 0]
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        sns.histplot(min_dist, bins=24, kde=True, color=NATURE_COLORS["teal"], ax=ax)
        ax.set_xlabel("Synthetic to nearest original distance")
        ax.set_title("Synthetic Nearest-Neighbor Distance")
        plots["Synthetic Nearest-Neighbor Distance"] = fig_to_base64(fig, fmt, dpi)

        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        sorted_dist = np.sort(min_dist)
        ax.plot(np.arange(1, len(sorted_dist) + 1), sorted_dist, color=NATURE_COLORS["red"], linewidth=2)
        ax.axhline(np.quantile(sorted_dist, 0.05), color=NATURE_COLORS["slate"], linestyle="--", label="P5 proximity")
        ax.set_xlabel("Synthetic samples sorted by proximity")
        ax.set_ylabel("Nearest-original distance")
        ax.set_title("Synthetic Memorization Risk Curve")
        ax.legend()
        plots["Synthetic Memorization Risk Curve"] = fig_to_base64(fig, fmt, dpi)

    # 8. Target-feature correlation drift.
    if target_col in df_old.columns and ptype == "Regression":
        y_old = pd.to_numeric(df_old[target_col], errors="coerce")
        y_new = pd.to_numeric(df_new[target_col], errors="coerce")
        rows = []
        for col in common:
            c1 = old_num[col].corr(y_old) if old_num[col].nunique() > 1 else 0
            c2 = new_num[col].corr(y_new) if new_num[col].nunique() > 1 else 0
            rows.append((col, float((0 if pd.isna(c2) else c2) - (0 if pd.isna(c1) else c1))))
        corr_drift = pd.Series(dict(rows)).abs().sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(8, max(4.5, 0.35 * len(corr_drift) + 1.5)))
        ax.barh(corr_drift.index, corr_drift.values, color=NATURE_COLORS["gold"])
        ax.set_xlabel("|target-correlation delta|")
        ax.set_title("Synthetic Target Relationship Drift")
        plots["Synthetic Target Relationship Drift"] = fig_to_base64(fig, fmt, dpi)

    # 9. Density overlay top shifted features.
    top_density = drift.sort_values(ascending=False).head(min(6, len(drift))).index.tolist()
    rows = int(np.ceil(len(top_density) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(10, max(4.8, rows * 3)), squeeze=False)
    for ax, col in zip(axes.ravel(), top_density):
        sns.kdeplot(old_num[col], ax=ax, color=NATURE_COLORS["blue"], label="Original", fill=True, alpha=0.12)
        sns.kdeplot(new_num[col], ax=ax, color=NATURE_COLORS["red"], label="Augmented", fill=True, alpha=0.10)
        ax.set_title(col)
    for ax in axes.ravel()[len(top_density):]:
        ax.axis("off")
    axes.ravel()[0].legend()
    fig.suptitle("Synthetic Density Overlay of Most Shifted Features", y=1.01)
    plots["Synthetic Density Overlay"] = fig_to_base64(fig, fmt, dpi)

    # 10. Covariance eigen spectrum.
    eig_old = np.linalg.eigvalsh(np.cov(old_num[common].fillna(0).to_numpy(), rowvar=False))
    eig_new = np.linalg.eigvalsh(np.cov(new_num[common].fillna(0).to_numpy(), rowvar=False))
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(np.sort(eig_old)[::-1], marker="o", label="Original", color=NATURE_COLORS["blue"])
    ax.plot(np.sort(eig_new)[::-1], marker="o", label="Augmented", color=NATURE_COLORS["red"])
    ax.set_xlabel("Eigen component")
    ax.set_ylabel("Covariance eigenvalue")
    ax.set_title("Synthetic Covariance Spectrum")
    ax.legend()
    plots["Synthetic Covariance Spectrum"] = fig_to_base64(fig, fmt, dpi)

    # 11. Quality scorecard.
    score_rows = [
        ["Mean shift", float(mean_shift.mean() if len(mean_shift) else 0)],
        ["Distribution drift", float(drift.mean() if len(drift) else 0)],
        ["Correlation delta", float(np.nanmean(np.abs(delta.to_numpy())))],
        ["Variance ratio error", float(np.nanmean(np.abs(var_ratio - 1))) if len(var_ratio) else 0],
    ]
    score_df = pd.DataFrame(score_rows, columns=["Metric", "Value"])
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    sns.heatmap(score_df.set_index("Metric").T, annot=True, fmt=".3f", cmap="rocket_r", ax=ax, cbar=False)
    ax.set_title("Synthetic Data Quality Scorecard")
    plots["Synthetic Data Quality Scorecard"] = fig_to_base64(fig, fmt, dpi)

    return plots

def get_feature_importances(df, target_col, method="Random Forest", ptype="Classification", top_k=10):
    X = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
    # Basic encoding if strings left
    X = pd.get_dummies(X, drop_first=True)
    X = X.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = _sanitize_target(df[target_col], ptype=ptype)
    
    if X.empty: return []
    
    if method == "Random Forest":
        model = RandomForestClassifier(random_state=42, n_jobs=-1) if ptype == "Classification" else RandomForestRegressor(random_state=42, n_jobs=-1)
        model.fit(X, y)
        importances = model.feature_importances_
    elif method == "Extra Trees":
        model = ExtraTreesClassifier(random_state=42, n_jobs=-1) if ptype == "Classification" else ExtraTreesRegressor(random_state=42, n_jobs=-1)
        model.fit(X, y)
        importances = model.feature_importances_
    elif method == "F-Score":
        if ptype == "Classification":
            importances, _ = f_classif(X, y)
        else:
            importances, _ = f_regression(X, y)
        importances = np.nan_to_num(importances, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        if ptype == "Classification":
            importances = mutual_info_classif(X, y)
        else:
            importances = mutual_info_regression(X, y)
            
    impt_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    impt_df = impt_df.sort_values(by='Importance', ascending=False).head(top_k)
    return impt_df.to_dict(orient="records")
