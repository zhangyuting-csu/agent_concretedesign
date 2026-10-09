import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patheffects as pe
from scipy.stats import wilcoxon
from scipy.stats import kurtosis, skew
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_curve,
)

from core.plotting_utils import NATURE_COLORS, apply_nature_style, fig_to_base64


def _ptype_is_classification(ptype):
    return str(ptype).strip().lower() == "classification"


def _ptype_is_regression(ptype):
    return str(ptype).strip().lower() == "regression"


def _series_values(arr):
    return arr.values if isinstance(arr, pd.Series) else np.asarray(arr)


def _regression_metrics(y_true, pred, ref_std):
    rmse = np.sqrt(mean_squared_error(y_true, pred))
    mae = mean_absolute_error(y_true, pred)
    return {
        "R2 Score": max(0, r2_score(y_true, pred)),
        "RMSE Skill": max(0, 1 - (rmse / (ref_std + 1e-9))),
        "MAE Skill": max(0, 1 - (mae / (ref_std + 1e-9))),
    }


def _classification_metrics(y_true, pred):
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, pred, average="weighted", zero_division=0),
    }


def _distinct_colors(count):
    if count <= 0:
        return []
    cmap = plt.get_cmap("turbo")
    samples = np.linspace(0.02, 0.98, count)
    return [cmap(sample) for sample in samples]


def _midpoint_annotation(theta_a, radius_a, theta_b, radius_b, label):
    x_a, y_a = radius_a * np.cos(theta_a), radius_a * np.sin(theta_a)
    x_b, y_b = radius_b * np.cos(theta_b), radius_b * np.sin(theta_b)
    x_mid, y_mid = (x_a + x_b) / 2.0, (y_a + y_b) / 2.0
    theta_mid = np.arctan2(y_mid, x_mid)
    if theta_mid < 0:
        theta_mid += 2 * np.pi
    radius_mid = np.sqrt(x_mid**2 + y_mid**2)
    return theta_mid, radius_mid, label


def _safe_corr(y_true, pred):
    if np.std(pred) <= 1e-12 or np.std(y_true) <= 1e-12:
        return 0.0
    corr = np.corrcoef(y_true, pred)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(np.clip(corr, -1, 1))


def _cv_score_summary(model, X_train, y_train, ptype):
    scoring = "accuracy" if _ptype_is_classification(ptype) else "r2"
    try:
        scores = cross_val_score(clone(model), X_train, y_train, cv=3, scoring=scoring, n_jobs=1)
        return float(np.mean(scores)), float(np.std(scores))
    except Exception:
        return np.nan, np.nan


def _ecdf_points(values):
    data = np.sort(np.asarray(values, dtype=float))
    if len(data) == 0:
        return np.array([]), np.array([])
    y = np.arange(1, len(data) + 1) / len(data)
    return data, y


def _label_offsets(count):
    pattern = [
        (8, 8),
        (10, -10),
        (-10, 10),
        (-12, -12),
        (14, 0),
        (0, 14),
        (-14, 0),
        (0, -14),
    ]
    return [pattern[idx % len(pattern)] for idx in range(count)]


def _model_response(model, X):
    if hasattr(model, "predict_proba"):
        try:
            proba = np.asarray(model.predict_proba(X))
            if proba.ndim == 2 and proba.shape[1] > 1:
                return proba[:, 1]
        except Exception:
            pass
    return np.asarray(model.predict(X), dtype=float)


def _top_numeric_features_for_error(X, values, limit=6):
    if not isinstance(X, pd.DataFrame):
        return []
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    scores = []
    values = np.asarray(values, dtype=float)
    for col in numeric_cols:
        series = pd.to_numeric(X[col], errors="coerce")
        mask = series.notna() & np.isfinite(values)
        if mask.sum() < 5 or series[mask].nunique() < 2:
            continue
        corr = np.corrcoef(series[mask].to_numpy(dtype=float), values[mask])[0, 1]
        if not np.isnan(corr):
            scores.append((abs(float(corr)), col))
    return [col for _, col in sorted(scores, reverse=True)[:limit]]


def _plot_residual_feature_scan(X_test, residuals, fmt="png", dpi=300):
    cols = _top_numeric_features_for_error(X_test, residuals, limit=6)
    if not cols:
        return None
    rows = int(np.ceil(len(cols) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(10, max(4.5, rows * 3.0)), squeeze=False)
    residuals = np.asarray(residuals, dtype=float)
    for ax, col in zip(axes.ravel(), cols):
        x = pd.to_numeric(X_test[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(residuals)
        ax.scatter(x[mask], residuals[mask], alpha=0.55, s=26, color=NATURE_COLORS["blue"], edgecolors="white", linewidth=0.25)
        ax.axhline(0, color=NATURE_COLORS["slate"], linestyle="--", linewidth=1.0)
        if mask.sum() >= 6 and len(np.unique(x[mask])) >= 3:
            deg = 2 if len(np.unique(x[mask])) > 3 else 1
            coef = np.polyfit(x[mask], residuals[mask], deg)
            grid = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 120)
            ax.plot(grid, np.poly1d(coef)(grid), color=NATURE_COLORS["red"], linewidth=1.8)
        ax.set_title(f"Residual vs {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Residual")
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    fig.suptitle("Residual-Feature Bias Scan", y=1.01)
    return fig_to_base64(fig, fmt, dpi)


def _plot_error_hotspot_heatmap(X_test, y_true, pred, fmt="png", dpi=300):
    abs_err = np.abs(np.asarray(y_true, dtype=float) - np.asarray(pred, dtype=float))
    cols = _top_numeric_features_for_error(X_test, abs_err, limit=2)
    if len(cols) < 2:
        return None
    work = pd.DataFrame({cols[0]: X_test[cols[0]], cols[1]: X_test[cols[1]], "Absolute Error": abs_err}).dropna()
    if len(work) < 12:
        return None
    try:
        work["X Bin"] = pd.qcut(work[cols[0]], q=min(6, work[cols[0]].nunique()), duplicates="drop")
        work["Y Bin"] = pd.qcut(work[cols[1]], q=min(6, work[cols[1]].nunique()), duplicates="drop")
        pivot = work.pivot_table(index="Y Bin", columns="X Bin", values="Absolute Error", aggfunc="mean", observed=False)
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    sns.heatmap(pivot, cmap="mako_r", annot=True, fmt=".2g", linewidths=0.45, ax=ax)
    ax.set_title(f"Error Hotspot Map: {cols[0]} x {cols[1]}")
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    return fig_to_base64(fig, fmt, dpi)


def _plot_uncertainty_error_alignment(bootstrap_diag, fmt="png", dpi=300):
    if not bootstrap_diag:
        return None
    y_true = np.asarray(bootstrap_diag.get("y_true"), dtype=float)
    pred_mean = np.asarray(bootstrap_diag.get("pred_mean"), dtype=float)
    low = np.asarray(bootstrap_diag.get("pred_low"), dtype=float)
    high = np.asarray(bootstrap_diag.get("pred_high"), dtype=float)
    if not len(y_true):
        return None
    width = high - low
    abs_err = np.abs(y_true - pred_mean)
    covered = (y_true >= low) & (y_true <= high)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    sc = axes[0].scatter(width, abs_err, c=covered.astype(int), cmap="coolwarm_r", s=42, alpha=0.78, edgecolors="white", linewidth=0.35)
    axes[0].set_xlabel("Bootstrap interval width")
    axes[0].set_ylabel("Absolute error")
    axes[0].set_title("Uncertainty vs Error Alignment")
    try:
        coef = np.polyfit(width, abs_err, 1)
        grid = np.linspace(width.min(), width.max(), 100)
        axes[0].plot(grid, np.poly1d(coef)(grid), color=NATURE_COLORS["slate"], linestyle="--")
    except Exception:
        pass
    order = np.argsort(width)
    rolling = pd.Series(covered[order].astype(float)).rolling(max(5, len(order) // 8), min_periods=3).mean()
    axes[1].plot(np.arange(len(order)), rolling, color=NATURE_COLORS["teal"], linewidth=2)
    axes[1].axhline(float(np.mean(covered)), color=NATURE_COLORS["red"], linestyle="--", label=f"Overall coverage={np.mean(covered):.2f}")
    axes[1].set_xlabel("Samples sorted by uncertainty")
    axes[1].set_ylabel("Rolling coverage")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Coverage Stability by Uncertainty")
    axes[1].legend()
    return fig_to_base64(fig, fmt, dpi)


def _plot_binary_threshold_diagnostics(y_true, proba, fmt="png", dpi=300):
    y_arr = np.asarray(y_true)
    if len(np.unique(y_arr)) != 2:
        return None
    classes = np.unique(y_arr)
    y_bin = (y_arr == classes[-1]).astype(int)
    thresholds = np.linspace(0.02, 0.98, 80)
    rows = []
    for thr in thresholds:
        pred = (proba >= thr).astype(int)
        rows.append({
            "Threshold": thr,
            "Precision": precision_score(y_bin, pred, zero_division=0),
            "Recall": recall_score(y_bin, pred, zero_division=0),
            "F1": f1_score(y_bin, pred, zero_division=0),
            "Positive Rate": float(pred.mean()),
        })
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for metric, color in [("Precision", NATURE_COLORS["blue"]), ("Recall", NATURE_COLORS["orange"]), ("F1", NATURE_COLORS["red"]), ("Positive Rate", NATURE_COLORS["teal"])]:
        ax.plot(df["Threshold"], df[metric], label=metric, color=color, linewidth=2)
    best = df.iloc[df["F1"].idxmax()]
    ax.axvline(best["Threshold"], color=NATURE_COLORS["slate"], linestyle="--", label=f"Best F1 threshold={best['Threshold']:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Threshold Utility Diagnostics")
    ax.legend()
    return fig_to_base64(fig, fmt, dpi)


def _plot_error_quantile_matrix(X_test, y_true, pred, fmt="png", dpi=300):
    abs_err = np.abs(np.asarray(y_true, dtype=float) - np.asarray(pred, dtype=float))
    cols = _top_numeric_features_for_error(X_test, abs_err, limit=8)
    if not cols:
        return None
    rows = []
    for col in cols:
        work = pd.DataFrame({"feature": pd.to_numeric(X_test[col], errors="coerce"), "abs_error": abs_err}).dropna()
        if len(work) < 10 or work["feature"].nunique() < 3:
            continue
        try:
            work["bin"] = pd.qcut(work["feature"], q=min(6, work["feature"].nunique()), duplicates="drop")
            profile = work.groupby("bin", observed=False)["abs_error"].mean()
            for idx, value in enumerate(profile):
                rows.append({"Feature": col, "Quantile Bin": f"Q{idx+1}", "Mean Absolute Error": float(value)})
        except Exception:
            continue
    matrix_df = pd.DataFrame(rows)
    if matrix_df.empty:
        return None
    pivot = matrix_df.pivot(index="Feature", columns="Quantile Bin", values="Mean Absolute Error")
    fig, ax = plt.subplots(figsize=(8.2, max(4.2, 0.45 * len(pivot) + 2)))
    sns.heatmap(pivot, annot=True, fmt=".2g", cmap="rocket_r", linewidths=0.4, ax=ax)
    ax.set_title("Feature Quantile Error Matrix")
    return fig_to_base64(fig, fmt, dpi)


def _plot_worst_error_table(y_true, pred, fmt="png", dpi=300):
    err = np.asarray(pred, dtype=float) - np.asarray(y_true, dtype=float)
    abs_err = np.abs(err)
    order = np.argsort(abs_err)[::-1][: min(20, len(abs_err))]
    table = pd.DataFrame({
        "Rank": np.arange(1, len(order) + 1),
        "Observed": np.asarray(y_true, dtype=float)[order],
        "Predicted": np.asarray(pred, dtype=float)[order],
        "Signed Error": err[order],
        "Absolute Error": abs_err[order],
    })
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.32 * len(table) + 2)))
    colors = [NATURE_COLORS["red"] if v > 0 else NATURE_COLORS["blue"] for v in table["Signed Error"]]
    ax.barh(table["Rank"].astype(str)[::-1], table["Signed Error"][::-1], color=colors[::-1])
    ax.axvline(0, color=NATURE_COLORS["slate"], linestyle="--", linewidth=1.0)
    ax.set_xlabel("Prediction - observed")
    ax.set_ylabel("Worst-error sample rank")
    ax.set_title("Worst Error Waterfall")
    return fig_to_base64(fig, fmt, dpi)


def _plot_residual_autocorrelation(residuals, fmt="png", dpi=300):
    residuals = np.asarray(residuals, dtype=float)
    if len(residuals) < 8:
        return None
    max_lag = min(24, len(residuals) // 2)
    lags = np.arange(1, max_lag + 1)
    acf = []
    centered = residuals - np.mean(residuals)
    denom = np.dot(centered, centered) + 1e-12
    for lag in lags:
        acf.append(float(np.dot(centered[:-lag], centered[lag:]) / denom))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot(np.arange(len(residuals)), residuals, color=NATURE_COLORS["blue"], linewidth=1.4, marker="o", markersize=3)
    rolling = pd.Series(residuals).rolling(max(4, len(residuals) // 12), min_periods=2).mean()
    axes[0].plot(rolling.index, rolling.values, color=NATURE_COLORS["red"], linewidth=2.0, label="Rolling mean")
    axes[0].axhline(0, color=NATURE_COLORS["slate"], linestyle="--", linewidth=1)
    axes[0].set_title("Residual Run Chart")
    axes[0].set_xlabel("Test sample order")
    axes[0].set_ylabel("Residual")
    axes[0].legend()
    axes[1].bar(lags, acf, color=NATURE_COLORS["purple"])
    axes[1].axhline(0, color=NATURE_COLORS["slate"], linewidth=1)
    axes[1].axhline(1.96 / np.sqrt(len(residuals)), color=NATURE_COLORS["orange"], linestyle="--", linewidth=1)
    axes[1].axhline(-1.96 / np.sqrt(len(residuals)), color=NATURE_COLORS["orange"], linestyle="--", linewidth=1)
    axes[1].set_title("Residual Autocorrelation")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("ACF")
    return fig_to_base64(fig, fmt, dpi)


def _plot_classwise_error_profile(y_true, pred, fmt="png", dpi=300):
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(pred)]))
    if len(labels) < 2:
        return None
    rows = []
    for label in labels:
        y_bin = np.asarray(y_true) == label
        p_bin = np.asarray(pred) == label
        tp = int(np.sum(y_bin & p_bin))
        fp = int(np.sum(~y_bin & p_bin))
        fn = int(np.sum(y_bin & ~p_bin))
        support = int(np.sum(y_bin))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append({"Class": str(label), "Precision": precision, "Recall": recall, "F1": f1, "Support": support})
    df = pd.DataFrame(rows).set_index("Class")
    fig, axes = plt.subplots(1, 2, figsize=(11, max(4.2, 0.3 * len(df) + 2)))
    sns.heatmap(df[["Precision", "Recall", "F1"]], annot=True, fmt=".2f", cmap="crest", vmin=0, vmax=1, ax=axes[0])
    axes[0].set_title("Class-wise Reliability")
    axes[1].barh(df.index, df["Support"], color=NATURE_COLORS["orange"])
    axes[1].set_title("Class Support")
    axes[1].set_xlabel("Samples")
    return fig_to_base64(fig, fmt, dpi)


def _plot_conformal_interval_diagnostics(y_train, train_pred, y_test, test_pred, fmt="png", dpi=300, alpha=0.1):
    if train_pred is None:
        return None
    y_train_v = np.asarray(y_train, dtype=float)
    train_pred_v = np.asarray(train_pred, dtype=float)
    y_test_v = np.asarray(y_test, dtype=float)
    test_pred_v = np.asarray(test_pred, dtype=float)
    if len(y_train_v) < 10 or len(y_test_v) < 5:
        return None
    q = float(np.quantile(np.abs(y_train_v - train_pred_v), 1 - alpha))
    low = test_pred_v - q
    high = test_pred_v + q
    covered = (y_test_v >= low) & (y_test_v <= high)
    order = np.argsort(test_pred_v)
    show = order[: min(80, len(order))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(show))
    axes[0].fill_between(x, low[show], high[show], color=NATURE_COLORS["blue"], alpha=0.16, label=f"{int((1-alpha)*100)}% conformal interval")
    axes[0].plot(x, test_pred_v[show], color=NATURE_COLORS["blue"], linewidth=1.8, label="Prediction")
    axes[0].scatter(x, y_test_v[show], c=covered[show], cmap="coolwarm_r", s=35, edgecolors="white", linewidth=0.3, label="Observed")
    axes[0].set_title("Split-Conformal Prediction Intervals")
    axes[0].set_xlabel("Test samples sorted by prediction")
    axes[0].set_ylabel("Target")
    axes[0].legend()
    try:
        bins = pd.qcut(pd.Series(y_test_v), q=min(6, len(np.unique(y_test_v))), duplicates="drop")
        cov = pd.DataFrame({"bin": bins, "covered": covered.astype(float), "width": high - low}).groupby("bin", observed=False).agg(coverage=("covered", "mean"), width=("width", "mean"))
        xpos = np.arange(len(cov))
        axes[1].bar(xpos, cov["coverage"], color=NATURE_COLORS["teal"], alpha=0.75, label="Coverage")
        axes[1].axhline(1 - alpha, color=NATURE_COLORS["red"], linestyle="--", label="Nominal")
        ax2 = axes[1].twinx()
        ax2.plot(xpos, cov["width"], color=NATURE_COLORS["purple"], marker="o", label="Mean width")
        axes[1].set_xticks(xpos)
        axes[1].set_xticklabels([f"Q{i+1}" for i in xpos])
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title("Coverage by Target Quantile")
        axes[1].set_xlabel("Observed target quantile")
        axes[1].set_ylabel("Coverage")
        ax2.set_ylabel("Interval width")
        axes[1].legend(loc="lower left")
        ax2.legend(loc="upper right")
    except Exception:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, f"Overall coverage = {covered.mean():.3f}", ha="center", va="center")
    return fig_to_base64(fig, fmt, dpi)


def _plot_slice_bias_heatmap(X_test, residuals, fmt="png", dpi=300):
    cols = _top_numeric_features_for_error(X_test, residuals, limit=10)
    if not cols:
        return None
    rows = []
    residuals = np.asarray(residuals, dtype=float)
    for col in cols:
        work = pd.DataFrame({"feature": pd.to_numeric(X_test[col], errors="coerce"), "residual": residuals}).dropna()
        if len(work) < 10 or work["feature"].nunique() < 3:
            continue
        try:
            work["bin"] = pd.qcut(work["feature"], q=min(6, work["feature"].nunique()), duplicates="drop")
            profile = work.groupby("bin", observed=False)["residual"].agg(["mean", "count"])
            for idx, row in enumerate(profile.itertuples()):
                rows.append({"Feature": col, "Quantile Bin": f"Q{idx+1}", "Mean Residual": float(row.mean), "Count": int(row.count)})
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    pivot = df.pivot(index="Feature", columns="Quantile Bin", values="Mean Residual")
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.42 * len(pivot) + 2)))
    sns.heatmap(pivot, annot=True, fmt=".2g", cmap="coolwarm", center=0, linewidths=0.45, ax=ax)
    ax.set_title("Feature Slice Bias Heatmap")
    return fig_to_base64(fig, fmt, dpi)


def _plot_prediction_risk_stratification(X_train, X_test, y_true, pred, bootstrap_diag, fmt="png", dpi=300):
    abs_err = np.abs(np.asarray(y_true, dtype=float) - np.asarray(pred, dtype=float))
    parts = {"Absolute Error": abs_err}
    try:
        train_num = pd.DataFrame(X_train).select_dtypes(include=[np.number])
        test_num = pd.DataFrame(X_test).select_dtypes(include=[np.number])
        common = [c for c in test_num.columns if c in train_num.columns]
        if common and len(train_num) >= 3:
            nbrs = NearestNeighbors(n_neighbors=min(5, len(train_num))).fit(train_num[common])
            dist, _ = nbrs.kneighbors(test_num[common])
            parts["AD Distance"] = dist.mean(axis=1)
    except Exception:
        pass
    if bootstrap_diag:
        try:
            idx = np.asarray(bootstrap_diag.get("indices"), dtype=int)
            width = np.asarray(bootstrap_diag.get("pred_high"), dtype=float) - np.asarray(bootstrap_diag.get("pred_low"), dtype=float)
            full_width = np.full(len(abs_err), np.nan)
            full_width[idx] = width
            parts["Bootstrap Width"] = full_width
        except Exception:
            pass
    risk_df = pd.DataFrame(parts)
    scaled = pd.DataFrame(index=risk_df.index)
    for col in risk_df.columns:
        vals = risk_df[col].astype(float)
        scaled[col] = (vals - np.nanmin(vals)) / (np.nanmax(vals) - np.nanmin(vals) + 1e-9)
    risk = scaled.drop(columns=["Absolute Error"], errors="ignore").mean(axis=1)
    if risk.isna().all():
        return None
    risk = risk.fillna(risk.median())
    try:
        bins = pd.qcut(risk, q=min(6, risk.nunique()), duplicates="drop")
        profile = pd.DataFrame({"risk": risk, "abs_error": abs_err, "bin": bins}).groupby("bin", observed=False).agg(mean_error=("abs_error", "mean"), median_risk=("risk", "median"), n=("abs_error", "size"))
    except Exception:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].scatter(risk, abs_err, color=NATURE_COLORS["red"], alpha=0.7, s=42, edgecolors="white", linewidth=0.3)
    axes[0].set_xlabel("Composite risk score")
    axes[0].set_ylabel("Absolute error")
    axes[0].set_title("Prediction Risk vs Error")
    xpos = np.arange(len(profile))
    axes[1].bar(xpos, profile["mean_error"], color=NATURE_COLORS["orange"], alpha=0.82)
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels([f"R{i+1}" for i in xpos])
    axes[1].set_xlabel("Risk stratum")
    axes[1].set_ylabel("Mean absolute error")
    axes[1].set_title("Risk-Stratified Error")
    return fig_to_base64(fig, fmt, dpi)


def _plot_feature_interaction_screen(model, X_train, fmt="png", dpi=300):
    if not isinstance(X_train, pd.DataFrame):
        return None
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()[:10]
    if len(numeric_cols) < 2 or len(X_train) < 20:
        return None
    sample = X_train.sample(min(400, len(X_train)), random_state=42).copy()
    base_pred = _model_response(model, sample)
    scores = []
    for i, a in enumerate(numeric_cols):
        for b in numeric_cols[i + 1:]:
            try:
                shuffled_a = sample.copy()
                shuffled_b = sample.copy()
                shuffled_ab = sample.copy()
                shuffled_a[a] = np.random.default_rng(42).permutation(shuffled_a[a].to_numpy())
                shuffled_b[b] = np.random.default_rng(43).permutation(shuffled_b[b].to_numpy())
                shuffled_ab[a] = np.random.default_rng(44).permutation(shuffled_ab[a].to_numpy())
                shuffled_ab[b] = np.random.default_rng(45).permutation(shuffled_ab[b].to_numpy())
                da = np.mean((base_pred - _model_response(model, shuffled_a)) ** 2)
                db = np.mean((base_pred - _model_response(model, shuffled_b)) ** 2)
                dab = np.mean((base_pred - _model_response(model, shuffled_ab)) ** 2)
                interaction = max(0.0, float(dab - da - db))
                scores.append({"Feature A": a, "Feature B": b, "Interaction Strength": interaction})
            except Exception:
                continue
    df = pd.DataFrame(scores)
    if df.empty or df["Interaction Strength"].max() <= 0:
        return None
    matrix = pd.DataFrame(0.0, index=numeric_cols, columns=numeric_cols)
    for _, row in df.iterrows():
        matrix.loc[row["Feature A"], row["Feature B"]] = row["Interaction Strength"]
        matrix.loc[row["Feature B"], row["Feature A"]] = row["Interaction Strength"]
    fig, ax = plt.subplots(figsize=(8, 6.8))
    sns.heatmap(matrix, cmap="YlOrRd", linewidths=0.4, ax=ax)
    ax.set_title("Permutation Interaction Strength Screen")
    return fig_to_base64(fig, fmt, dpi)


def _plot_prediction_diagnostic_gallery(y_true, pred, residuals, fmt="png", dpi=300):
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    abs_err = np.abs(residuals)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    sns.kdeplot(x=y_true, y=pred, fill=True, cmap="Blues", thresh=0.05, ax=axes[0, 0])
    axes[0, 0].scatter(y_true, pred, s=12, alpha=0.35, color=NATURE_COLORS["slate"])
    axes[0, 0].set_title("Prediction Density Map")
    sns.ecdfplot(abs_err, ax=axes[0, 1], color=NATURE_COLORS["red"], linewidth=2)
    axes[0, 1].set_title("Absolute Error ECDF Detail")
    axes[0, 1].set_xlabel("Absolute error")
    axes[1, 0].scatter(pred, np.sqrt(abs_err), color=NATURE_COLORS["purple"], alpha=0.65, edgecolors="white", linewidth=0.25)
    axes[1, 0].set_title("Scale-Location Diagnostic")
    axes[1, 0].set_xlabel("Predicted")
    axes[1, 0].set_ylabel("sqrt(|residual|)")
    signed_rank = pd.Series(residuals).rank().to_numpy()
    axes[1, 1].scatter(np.arange(len(residuals)), signed_rank, c=residuals, cmap="coolwarm", s=28, edgecolors="white", linewidth=0.25)
    axes[1, 1].set_title("Residual Rank Signature")
    axes[1, 1].set_xlabel("Sample order")
    axes[1, 1].set_ylabel("Residual rank")
    return fig_to_base64(fig, fmt, dpi)


def _plot_distribution_shift_predictions(y_train, train_pred, y_test, test_pred, fmt="png", dpi=300):
    if train_pred is None:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    sns.kdeplot(y_train, ax=axes[0], label="Train observed", color=NATURE_COLORS["blue"], fill=True, alpha=0.12)
    sns.kdeplot(y_test, ax=axes[0], label="Test observed", color=NATURE_COLORS["red"], fill=True, alpha=0.12)
    axes[0].set_title("Observed Target Shift")
    axes[0].legend()
    sns.kdeplot(train_pred, ax=axes[1], label="Train predicted", color=NATURE_COLORS["blue"], fill=True, alpha=0.12)
    sns.kdeplot(test_pred, ax=axes[1], label="Test predicted", color=NATURE_COLORS["red"], fill=True, alpha=0.12)
    axes[1].set_title("Prediction Distribution Shift")
    axes[1].legend()
    sns.kdeplot(np.asarray(y_train)-np.asarray(train_pred), ax=axes[2], label="Train residual", color=NATURE_COLORS["blue"], fill=True, alpha=0.12)
    sns.kdeplot(np.asarray(y_test)-np.asarray(test_pred), ax=axes[2], label="Test residual", color=NATURE_COLORS["red"], fill=True, alpha=0.12)
    axes[2].set_title("Residual Distribution Shift")
    axes[2].legend()
    return fig_to_base64(fig, fmt, dpi)


def _plot_segment_performance_dashboard(y_true, pred, fmt="png", dpi=300):
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    try:
        bins = pd.qcut(pd.Series(y_true), q=min(8, len(np.unique(y_true))), duplicates="drop")
    except Exception:
        return None
    df = pd.DataFrame({"Observed": y_true, "Predicted": pred, "Bin": bins})
    rows = []
    for i, (_, g) in enumerate(df.groupby("Bin", observed=False), start=1):
        rows.append({
            "Segment": f"S{i}",
            "R2": r2_score(g["Observed"], g["Predicted"]) if len(g) > 2 else np.nan,
            "RMSE": np.sqrt(mean_squared_error(g["Observed"], g["Predicted"])),
            "MAE": mean_absolute_error(g["Observed"], g["Predicted"]),
            "Bias": float((g["Predicted"] - g["Observed"]).mean()),
        })
    mat = pd.DataFrame(rows).set_index("Segment")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    sns.heatmap(mat, annot=True, fmt=".2g", cmap="Spectral_r", center=0, linewidths=0.4, ax=ax)
    ax.set_title("Target-Segment Performance Dashboard")
    return fig_to_base64(fig, fmt, dpi)


def _plot_feature_response_sensitivity_grid(model, X_train, fmt="png", dpi=300):
    if not isinstance(X_train, pd.DataFrame):
        return None
    cols = X_train.select_dtypes(include=[np.number]).columns.tolist()[:8]
    if not cols:
        return None
    base = X_train.median(numeric_only=True).to_dict()
    base_row = X_train.iloc[[0]].copy()
    for col, val in base.items():
        if col in base_row:
            base_row[col] = val
    rows = int(np.ceil(len(cols) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(10, max(4.8, rows * 3.0)), squeeze=False)
    for ax, col in zip(axes.ravel(), cols):
        values = np.linspace(X_train[col].quantile(0.02), X_train[col].quantile(0.98), 60)
        X_rep = pd.concat([base_row] * len(values), ignore_index=True)
        X_rep[col] = values
        yhat = _model_response(model, X_rep)
        ax.plot(values, yhat, color=NATURE_COLORS["blue"], linewidth=2)
        ax.axvline(base_row[col].iloc[0], color=NATURE_COLORS["red"], linestyle="--", linewidth=1)
        ax.set_title(f"Response: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Prediction")
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    fig.suptitle("One-Factor Model Response Sensitivity Grid", y=1.01)
    return fig_to_base64(fig, fmt, dpi)


def _plot_prediction_interval_residual_map(y_true, pred, fmt="png", dpi=300):
    residual = np.asarray(y_true, dtype=float) - np.asarray(pred, dtype=float)
    pred = np.asarray(pred, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    hb = ax.hexbin(pred, residual, gridsize=24, cmap="viridis", mincnt=1)
    fig.colorbar(hb, ax=ax, label="Sample count")
    ax.axhline(0, color=NATURE_COLORS["red"], linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residual Density Hexbin")
    return fig_to_base64(fig, fmt, dpi)


def _plot_regression_calibration_belt(y_true, pred, fmt="png", dpi=300):
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if len(y_true) < 12:
        return None
    try:
        bins = pd.qcut(pd.Series(pred), q=min(10, len(np.unique(pred))), duplicates="drop")
    except Exception:
        return None
    df = pd.DataFrame({"Observed": y_true, "Predicted": pred, "Bin": bins})
    grouped = df.groupby("Bin", observed=False).agg(
        pred_mean=("Predicted", "mean"),
        obs_mean=("Observed", "mean"),
        obs_std=("Observed", "std"),
        n=("Observed", "size"),
    ).dropna()
    if grouped.empty:
        return None
    grouped["se"] = grouped["obs_std"] / np.sqrt(grouped["n"].clip(lower=1))
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    ax.errorbar(grouped["pred_mean"], grouped["obs_mean"], yerr=1.96 * grouped["se"], fmt="o", color=NATURE_COLORS["blue"], ecolor=NATURE_COLORS["gray"], capsize=3, label="Binned mean +/- 95% CI")
    lo = min(np.min(pred), np.min(y_true))
    hi = max(np.max(pred), np.max(y_true))
    ax.plot([lo, hi], [lo, hi], color=NATURE_COLORS["red"], linestyle="--", linewidth=1.8, label="Perfect calibration")
    ax.set_xlabel("Mean predicted value")
    ax.set_ylabel("Mean observed value")
    ax.set_title("Regression Calibration Belt")
    ax.legend()
    return fig_to_base64(fig, fmt, dpi)


def _plot_error_violin_by_target_segment(y_true, pred, fmt="png", dpi=300):
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - y_true
    try:
        bins = pd.qcut(pd.Series(y_true), q=min(8, len(np.unique(y_true))), duplicates="drop")
    except Exception:
        return None
    df = pd.DataFrame({"Segment": bins, "Signed Error": err, "Absolute Error": np.abs(err)})
    df["Segment Label"] = df["Segment"].cat.codes.map(lambda x: f"Q{x+1}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.violinplot(data=df, x="Segment Label", y="Signed Error", ax=axes[0], color="#c7dcef", inner="quartile")
    axes[0].axhline(0, color=NATURE_COLORS["red"], linestyle="--")
    axes[0].set_title("Signed Error Distribution by Target Segment")
    sns.boxplot(data=df, x="Segment Label", y="Absolute Error", ax=axes[1], color="#b7d6c6")
    axes[1].set_title("Absolute Error Spread by Target Segment")
    return fig_to_base64(fig, fmt, dpi)


def _plot_standardized_residual_control_chart(residuals, fmt="png", dpi=300):
    residuals = np.asarray(residuals, dtype=float)
    if len(residuals) < 8:
        return None
    z = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-9)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(np.arange(len(z)), z, color=NATURE_COLORS["blue"], marker="o", markersize=3.5, linewidth=1.3)
    for level, color, label in [(0, NATURE_COLORS["slate"], "center"), (2, NATURE_COLORS["orange"], "+/-2 sigma"), (-2, NATURE_COLORS["orange"], None), (3, NATURE_COLORS["red"], "+/-3 sigma"), (-3, NATURE_COLORS["red"], None)]:
        ax.axhline(level, color=color, linestyle="--" if level else "-", linewidth=1.2, label=label)
    out = np.where(np.abs(z) > 3)[0]
    if len(out):
        ax.scatter(out, z[out], color=NATURE_COLORS["red"], s=70, edgecolors="white", zorder=4, label="Out of control")
    ax.set_xlabel("Test sample order")
    ax.set_ylabel("Standardized residual")
    ax.set_title("Standardized Residual Control Chart")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    return fig_to_base64(fig, fmt, dpi)


def _plot_residual_statistics_panel(residuals, fmt="png", dpi=300):
    residuals = np.asarray(residuals, dtype=float)
    if len(residuals) < 6:
        return None
    stats = pd.DataFrame(
        {
            "Metric": ["Mean", "Median", "Std", "MAD", "Skewness", "Kurtosis", "P95 Abs Error"],
            "Value": [
                np.mean(residuals),
                np.median(residuals),
                np.std(residuals),
                np.median(np.abs(residuals - np.median(residuals))),
                skew(residuals),
                kurtosis(residuals),
                np.quantile(np.abs(residuals), 0.95),
            ],
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].axis("off")
    table = axes[0].table(cellText=np.round(stats["Value"].to_numpy(dtype=float), 4).reshape(-1, 1), rowLabels=stats["Metric"], colLabels=["Value"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    axes[0].set_title("Residual Summary Statistics")
    sns.boxenplot(x=residuals, ax=axes[1], color="#c7dcef")
    axes[1].axvline(0, color=NATURE_COLORS["red"], linestyle="--")
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residual Tail Shape")
    return fig_to_base64(fig, fmt, dpi)


def _plot_observed_predicted_marginal_joint(y_true, pred, fmt="png", dpi=300):
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if len(y_true) < 8:
        return None
    df = pd.DataFrame({"Observed": y_true, "Predicted": pred})
    g = sns.jointplot(data=df, x="Observed", y="Predicted", kind="hex", height=7, cmap="viridis", marginal_kws={"bins": 24})
    lo = min(y_true.min(), pred.min())
    hi = max(y_true.max(), pred.max())
    g.ax_joint.plot([lo, hi], [lo, hi], color=NATURE_COLORS["red"], linestyle="--", linewidth=1.8)
    g.fig.suptitle("Observed-Predicted Joint Marginal Map", y=1.02)
    return fig_to_base64(g.fig, fmt, dpi)


def _plot_error_exceedance_curve(y_true, pred, fmt="png", dpi=300):
    abs_err = np.sort(np.abs(np.asarray(y_true, dtype=float) - np.asarray(pred, dtype=float)))
    if len(abs_err) < 5:
        return None
    exceed = 1 - np.arange(1, len(abs_err) + 1) / len(abs_err)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.step(abs_err, exceed, where="post", color=NATURE_COLORS["red"], linewidth=2.2)
    for q in [0.5, 0.8, 0.9, 0.95]:
        val = np.quantile(abs_err, q)
        ax.axvline(val, color=NATURE_COLORS["gray"], linestyle="--", linewidth=1)
        ax.text(val, 0.95 - q * 0.25, f"P{int(q*100)}={val:.2g}", rotation=90, fontsize=8, color=NATURE_COLORS["slate"])
    ax.set_xlabel("Absolute error threshold")
    ax.set_ylabel("Exceedance probability")
    ax.set_title("Error Exceedance Curve")
    return fig_to_base64(fig, fmt, dpi)


def _safe_qbin(values, q=8):
    values = pd.Series(values)
    try:
        return pd.qcut(values, q=min(q, values.nunique()), duplicates="drop")
    except Exception:
        return pd.cut(values, bins=min(q, max(2, values.nunique())), duplicates="drop")


def _plot_additional_regression_diagnostics(model, X_test, y_true, pred, fmt="png", dpi=300):
    plots = {}
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    residual = y_true - pred
    signed_error = pred - y_true
    abs_err = np.abs(residual)
    n = len(y_true)
    if n < 6:
        return plots

    def put(name, fig):
        plots[name] = fig_to_base64(fig, fmt, dpi)

    # 1
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    if n > 1:
        ax.scatter(residual[:-1], residual[1:], color=NATURE_COLORS["blue"], alpha=0.72, edgecolors="white", linewidth=0.3)
        ax.axhline(0, color=NATURE_COLORS["slate"], linestyle="--")
        ax.axvline(0, color=NATURE_COLORS["slate"], linestyle="--")
    ax.set_xlabel("Residual(t)")
    ax.set_ylabel("Residual(t+1)")
    ax.set_title("Residual Lag Scatter")
    put("Residual Lag Scatter", fig)

    # 2
    fig, ax = plt.subplots(figsize=(8, 4.6))
    window = max(4, n // 10)
    rolling_rmse = pd.Series(residual).rolling(window, min_periods=2).apply(lambda x: float(np.sqrt(np.mean(x**2))))
    ax.plot(rolling_rmse.index, rolling_rmse.values, color=NATURE_COLORS["red"], linewidth=2)
    ax.set_xlabel("Sample order")
    ax.set_ylabel("Rolling RMSE")
    ax.set_title("Rolling RMSE Trace")
    put("Rolling RMSE Trace", fig)

    # 3
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    sizes = 36 + 260 * (abs_err / (abs_err.max() + 1e-9))
    ax.scatter(y_true, pred, s=sizes, c=abs_err, cmap="rocket_r", alpha=0.75, edgecolors="white", linewidth=0.35)
    lo, hi = min(y_true.min(), pred.min()), max(y_true.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], color=NATURE_COLORS["slate"], linestyle="--")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title("Prediction Error Bubble Map")
    put("Prediction Error Bubble Map", fig)

    # 4
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    sorted_err = np.sort(abs_err)[::-1]
    share = np.cumsum(sorted_err) / (sorted_err.sum() + 1e-9)
    ax.bar(np.arange(1, len(sorted_err) + 1), sorted_err, color=NATURE_COLORS["orange"], alpha=0.65)
    ax2 = ax.twinx()
    ax2.plot(np.arange(1, len(sorted_err) + 1), share, color=NATURE_COLORS["red"], linewidth=2)
    ax.set_xlabel("Samples sorted by absolute error")
    ax.set_ylabel("Absolute error")
    ax2.set_ylabel("Cumulative error share")
    ax.set_title("Absolute Error Pareto Chart")
    put("Absolute Error Pareto Chart", fig)

    # 5
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    dec = _safe_qbin(abs_err, 10)
    prof = pd.DataFrame({"decile": dec, "abs_error": abs_err}).groupby("decile", observed=False)["abs_error"].mean()
    ax.plot(np.arange(1, len(prof) + 1), prof.values, marker="o", color=NATURE_COLORS["purple"], linewidth=2.2)
    ax.set_xlabel("Absolute-error decile")
    ax.set_ylabel("Mean absolute error")
    ax.set_title("Error Decile Lift Chart")
    put("Error Decile Lift Chart", fig)

    # 6
    fig, ax = plt.subplots(figsize=(8, 4.8))
    order = np.argsort(pred)
    x = np.arange(n)
    ax.plot(x, y_true[order], color=NATURE_COLORS["slate"], linewidth=1.6, label="Observed")
    ax.plot(x, pred[order], color=NATURE_COLORS["red"], linewidth=1.6, label="Predicted")
    ax.fill_between(x, pred[order] - np.quantile(abs_err, 0.9), pred[order] + np.quantile(abs_err, 0.9), color=NATURE_COLORS["red"], alpha=0.12, label="P90 error band")
    ax.set_title("Target Coverage Error Band")
    ax.set_xlabel("Samples sorted by prediction")
    ax.legend()
    put("Target Coverage Error Band", fig)

    # 7
    fig, ax = plt.subplots(figsize=(8, 4.8))
    pbin = _safe_qbin(pred, 8)
    bias = pd.DataFrame({"bin": pbin, "bias": signed_error}).groupby("bin", observed=False)["bias"].agg(["mean", "std", "count"])
    xpos = np.arange(len(bias))
    ax.errorbar(xpos, bias["mean"], yerr=1.96 * bias["std"] / np.sqrt(bias["count"].clip(lower=1)), fmt="o", color=NATURE_COLORS["blue"], capsize=3)
    ax.axhline(0, color=NATURE_COLORS["red"], linestyle="--")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"P{i+1}" for i in xpos])
    ax.set_xlabel("Predicted-value segment")
    ax.set_ylabel("Mean signed error")
    ax.set_title("Prediction-Segment Bias with CI")
    put("Prediction-Segment Bias with CI", fig)

    # 8
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    sns.histplot(residual, kde=True, ax=axes[0], color=NATURE_COLORS["blue"])
    axes[0].axvline(0, color=NATURE_COLORS["red"], linestyle="--")
    axes[0].set_title("Residual Normality Histogram")
    quantiles = np.linspace(0, 1, n, endpoint=False) + 0.5 / n
    theo = np.quantile(np.random.default_rng(42).normal(size=5000), quantiles)
    sample = np.quantile(np.sort((residual - residual.mean()) / (residual.std() + 1e-9)), quantiles)
    axes[1].scatter(theo, sample, color=NATURE_COLORS["purple"], s=22)
    axes[1].plot([theo.min(), theo.max()], [theo.min(), theo.max()], color=NATURE_COLORS["slate"], linestyle="--")
    axes[1].set_title("Standardized Residual Q-Q Detail")
    put("Residual Normality Detail Panel", fig)

    # 9
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    neg_x, neg_y = _ecdf_points(-residual[residual < 0])
    pos_x, pos_y = _ecdf_points(residual[residual > 0])
    ax.step(pos_x, pos_y, where="post", label="Positive residual", color=NATURE_COLORS["red"])
    ax.step(neg_x, neg_y, where="post", label="Negative residual magnitude", color=NATURE_COLORS["blue"])
    ax.set_xlabel("Residual magnitude")
    ax.set_ylabel("ECDF")
    ax.set_title("Signed Residual ECDF")
    ax.legend()
    put("Signed Residual ECDF", fig)

    # 10
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    q = np.linspace(0.05, 0.95, 19)
    pos = np.quantile(np.maximum(residual, 0), q)
    neg = np.quantile(np.maximum(-residual, 0), q)
    ax.scatter(neg, pos, color=NATURE_COLORS["teal"], s=45, edgecolors="white")
    lim = max(pos.max(), neg.max()) if len(pos) else 1
    ax.plot([0, lim], [0, lim], color=NATURE_COLORS["slate"], linestyle="--")
    ax.set_xlabel("Negative residual quantiles")
    ax.set_ylabel("Positive residual quantiles")
    ax.set_title("Error Symmetry Plot")
    put("Error Symmetry Plot", fig)

    # 11
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    contribution = sorted_err / (sorted_err.sum() + 1e-9)
    ax.plot(np.arange(1, len(contribution) + 1), np.cumsum(contribution), color=NATURE_COLORS["red"], linewidth=2.2)
    ax.axhline(0.8, color=NATURE_COLORS["slate"], linestyle="--")
    ax.set_xlabel("Top-k worst samples")
    ax.set_ylabel("Cumulative error contribution")
    ax.set_title("Cumulative Error Contribution Curve")
    put("Cumulative Error Contribution Curve", fig)

    # 12
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    obs_rank = pd.Series(y_true).rank().to_numpy()
    pred_rank = pd.Series(pred).rank().to_numpy()
    ax.scatter(obs_rank, pred_rank, c=abs_err, cmap="rocket_r", s=36, alpha=0.75, edgecolors="white")
    ax.plot([obs_rank.min(), obs_rank.max()], [obs_rank.min(), obs_rank.max()], color=NATURE_COLORS["slate"], linestyle="--")
    ax.set_xlabel("Observed rank")
    ax.set_ylabel("Predicted rank")
    ax.set_title("Prediction Rank Concordance")
    put("Prediction Rank Concordance", fig)

    # 13
    fig, ax = plt.subplots(figsize=(8, 4.8))
    rolling_std = pd.Series(residual).rolling(window, min_periods=2).std()
    ax.plot(rolling_std.index, rolling_std.values, color=NATURE_COLORS["orange"], linewidth=2)
    ax.set_xlabel("Sample order")
    ax.set_ylabel("Rolling residual std")
    ax.set_title("Local Error Volatility")
    put("Local Error Volatility", fig)

    # 14
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    centered = residual - residual.mean()
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    freq = np.fft.rfftfreq(len(centered))
    ax.plot(freq[1:], spectrum[1:], color=NATURE_COLORS["purple"], linewidth=1.8)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Power")
    ax.set_title("Residual Periodogram Spectrum")
    put("Residual Periodogram Spectrum", fig)

    # 15
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    mean_vals = (pred + y_true) / 2
    diff_vals = pred - y_true
    hb = ax.hexbin(mean_vals, diff_vals, gridsize=24, cmap="mako", mincnt=1)
    fig.colorbar(hb, ax=ax, label="Count")
    ax.axhline(diff_vals.mean(), color=NATURE_COLORS["red"], linewidth=1.6)
    ax.axhline(diff_vals.mean() + 1.96 * diff_vals.std(), color=NATURE_COLORS["slate"], linestyle="--")
    ax.axhline(diff_vals.mean() - 1.96 * diff_vals.std(), color=NATURE_COLORS["slate"], linestyle="--")
    ax.set_title("Bland-Altman Density Map")
    ax.set_xlabel("Mean of observed and predicted")
    ax.set_ylabel("Prediction - observed")
    put("Bland-Altman Density Map", fig)

    # 16
    fig, ax = plt.subplots(figsize=(8, 4.8))
    calib = pd.DataFrame({"pred": pred, "resid": residual, "abs": abs_err, "bin": _safe_qbin(pred, 8)})
    grouped = calib.groupby("bin", observed=False).agg(pred_mean=("pred", "mean"), resid_mean=("resid", "mean"), resid_std=("resid", "std"), n=("resid", "size"))
    ax.fill_between(grouped["pred_mean"], grouped["resid_mean"] - grouped["resid_std"], grouped["resid_mean"] + grouped["resid_std"], color=NATURE_COLORS["blue"], alpha=0.15)
    ax.plot(grouped["pred_mean"], grouped["resid_mean"], marker="o", color=NATURE_COLORS["blue"])
    ax.axhline(0, color=NATURE_COLORS["red"], linestyle="--")
    ax.set_xlabel("Predicted segment mean")
    ax.set_ylabel("Mean residual +/- SD")
    ax.set_title("Calibration Residual Belt")
    put("Calibration Residual Belt", fig)

    # 17
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    boxdf = pd.DataFrame({"Segment": _safe_qbin(pred, 8), "Absolute Error": abs_err})
    boxdf["Segment Label"] = boxdf["Segment"].cat.codes.map(lambda x: f"P{x+1}")
    sns.boxenplot(data=boxdf, x="Segment Label", y="Absolute Error", ax=ax, color="#c7dcef")
    ax.set_title("Error Boxen by Prediction Segment")
    put("Error Boxen by Prediction Segment", fig)

    # 18
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    rel = abs_err / (np.abs(y_true) + 1e-9)
    sns.histplot(np.clip(rel, 0, np.nanquantile(rel, 0.98)), kde=True, color=NATURE_COLORS["teal"], ax=ax)
    ax.set_xlabel("Relative absolute error")
    ax.set_title("Relative Error Distribution")
    put("Relative Error Distribution", fig)

    # 19
    fig, ax = plt.subplots(figsize=(8, 4.8))
    mape_df = pd.DataFrame({"Segment": _safe_qbin(y_true, 8), "MAPE": rel * 100})
    mape = mape_df.groupby("Segment", observed=False)["MAPE"].median()
    ax.bar(np.arange(len(mape)), mape.values, color=NATURE_COLORS["gold"])
    ax.set_xticks(np.arange(len(mape)))
    ax.set_xticklabels([f"Q{i+1}" for i in range(len(mape))])
    ax.set_xlabel("Observed target segment")
    ax.set_ylabel("Median absolute percentage error (%)")
    ax.set_title("MAPE by Target Segment")
    put("MAPE by Target Segment", fig)

    # 20
    if isinstance(X_test, pd.DataFrame) and len(X_test.select_dtypes(include=[np.number]).columns):
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        numeric_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()
        base = np.asarray(pred, dtype=float)
        levels = [0.01, 0.03, 0.05, 0.08, 0.12]
        drops = []
        rng = np.random.default_rng(42)
        scale = X_test[numeric_cols].std().replace(0, 1.0)
        for level in levels:
            noisy = X_test.copy()
            for col in numeric_cols:
                noisy[col] = noisy[col] + rng.normal(0, scale[col] * level, len(noisy))
            noisy_pred = _model_response(model, noisy)
            drops.append(float(np.mean(np.abs(noisy_pred - base))))
        ax.plot(np.array(levels) * 100, drops, marker="o", color=NATURE_COLORS["red"], linewidth=2.2)
        ax.set_xlabel("Feature noise level (% of std)")
        ax.set_ylabel("Mean prediction perturbation")
        ax.set_title("Input Noise Robustness Curve")
        put("Input Noise Robustness Curve", fig)

    # 21
    if isinstance(X_test, pd.DataFrame):
        numeric_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()[:12]
        if numeric_cols:
            rows = []
            for col in numeric_cols:
                s = pd.to_numeric(X_test[col], errors="coerce")
                mask = s.notna()
                if mask.sum() > 5 and s[mask].nunique() > 2:
                    rows.append({"Feature": col, "Corr With Abs Error": np.corrcoef(s[mask], abs_err[mask])[0, 1]})
            corrdf = pd.DataFrame(rows).dropna()
            if not corrdf.empty:
                fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(corrdf) + 1.6)))
                corrdf = corrdf.sort_values("Corr With Abs Error")
                colors = [NATURE_COLORS["blue"] if v < 0 else NATURE_COLORS["red"] for v in corrdf["Corr With Abs Error"]]
                ax.barh(corrdf["Feature"], corrdf["Corr With Abs Error"], color=colors)
                ax.axvline(0, color=NATURE_COLORS["slate"], linestyle="--")
                ax.set_title("Feature-Error Correlation Bars")
                put("Feature-Error Correlation Bars", fig)

    # 22
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    thresholds = np.linspace(0, np.quantile(abs_err, 0.98), 60)
    utility = [np.mean(abs_err <= t) - 0.02 * t for t in thresholds]
    ax.plot(thresholds, utility, color=NATURE_COLORS["purple"], linewidth=2)
    ax.set_xlabel("Acceptable absolute error threshold")
    ax.set_ylabel("Coverage - 0.02 * threshold")
    ax.set_title("Error Tolerance Utility Curve")
    put("Error Tolerance Utility Curve", fig)

    return plots


def _plot_deep_regression_diagnostics(model, X_test, y_true, pred, fmt="png", dpi=300):
    plots = {}
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    residual = y_true - pred
    signed_error = pred - y_true
    abs_err = np.abs(residual)
    n = len(y_true)
    if n < 8:
        return plots

    def put(name, fig):
        plots[name] = fig_to_base64(fig, fmt, dpi)

    X_num = None
    numeric_cols = []
    if isinstance(X_test, pd.DataFrame):
        X_num = X_test.select_dtypes(include=[np.number]).copy()
        numeric_cols = X_num.columns.tolist()

    # 1
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    q = np.linspace(0.01, 0.99, n)
    theo = np.quantile(np.random.default_rng(7).normal(size=8000), q)
    sample = np.quantile((residual - residual.mean()) / (residual.std() + 1e-9), q)
    tail_color = np.where((q < 0.1) | (q > 0.9), NATURE_COLORS["red"], NATURE_COLORS["blue"])
    ax.scatter(theo, sample, c=tail_color, s=24, alpha=0.78, edgecolors="white", linewidth=0.2)
    ax.plot([theo.min(), theo.max()], [theo.min(), theo.max()], color=NATURE_COLORS["slate"], linestyle="--")
    ax.set_xlabel("Normal quantile")
    ax.set_ylabel("Standardized residual quantile")
    ax.set_title("Residual Tail QQ Plot")
    put("Residual Tail QQ Plot", fig)

    # 2
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    signs = np.where(residual >= 0, 1, 0)
    mat = np.zeros((2, 2), dtype=int)
    for a, b in zip(signs[:-1], signs[1:]):
        mat[a, b] += 1
    sns.heatmap(pd.DataFrame(mat, index=["Neg/Under", "Pos/Over"], columns=["Neg/Under", "Pos/Over"]), annot=True, fmt="d", cmap="crest", ax=ax, cbar=False)
    ax.set_xlabel("Next residual sign")
    ax.set_ylabel("Current residual sign")
    ax.set_title("Residual Sign Transition Matrix")
    put("Residual Sign Transition Matrix", fig)

    # 3
    fig, ax = plt.subplots(figsize=(8, 4.8))
    roll_q = pd.Series(residual).rolling(max(6, n // 8), min_periods=4)
    ax.plot(roll_q.quantile(0.1), color=NATURE_COLORS["blue"], label="P10")
    ax.plot(roll_q.quantile(0.5), color=NATURE_COLORS["slate"], label="P50")
    ax.plot(roll_q.quantile(0.9), color=NATURE_COLORS["red"], label="P90")
    ax.axhline(0, color=NATURE_COLORS["gray"], linestyle="--", linewidth=1)
    ax.set_xlabel("Sample order")
    ax.set_ylabel("Rolling residual quantile")
    ax.set_title("Residual Quantile Trend")
    ax.legend()
    put("Residual Quantile Trend", fig)

    # 4
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    pbin = _safe_qbin(pred, 10)
    cal = pd.DataFrame({"bin": pbin, "Observed": y_true, "Predicted": pred}).groupby("bin", observed=False).mean()
    x = np.arange(len(cal))
    ax.plot(x, cal["Observed"], marker="o", color=NATURE_COLORS["blue"], label="Observed mean")
    ax.plot(x, cal["Predicted"], marker="o", color=NATURE_COLORS["red"], label="Predicted mean")
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{i+1}" for i in x])
    ax.set_xlabel("Prediction quantile")
    ax.set_ylabel("Mean value")
    ax.set_title("Prediction Quantile Calibration Ladder")
    ax.legend()
    put("Prediction Quantile Calibration Ladder", fig)

    # 5
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    tbin = _safe_qbin(y_true, 10)
    bias = pd.DataFrame({"bin": tbin, "bias": signed_error}).groupby("bin", observed=False)["bias"].median()
    ax.bar(np.arange(len(bias)), bias.values, color=[NATURE_COLORS["red"] if v > 0 else NATURE_COLORS["blue"] for v in bias.values])
    ax.axhline(0, color=NATURE_COLORS["slate"], linestyle="--")
    ax.set_xticks(np.arange(len(bias)))
    ax.set_xticklabels([f"T{i+1}" for i in range(len(bias))])
    ax.set_xlabel("Observed target quantile")
    ax.set_ylabel("Median signed error")
    ax.set_title("Target Quantile Bias Ladder")
    put("Target Quantile Bias Ladder", fig)

    if X_num is not None and numeric_cols:
        clean = X_num.replace([np.inf, -np.inf], np.nan)
        norm = (clean - clean.mean()) / (clean.std().replace(0, 1.0) + 1e-9)
        cols = numeric_cols[: min(12, len(numeric_cols))]

        # 6
        worst_idx = np.argsort(abs_err)[-min(20, n):]
        fig, ax = plt.subplots(figsize=(max(7.6, 0.55 * len(cols) + 2), max(4.8, 0.22 * len(worst_idx) + 1.6)))
        sns.heatmap(norm.iloc[worst_idx][cols], cmap="vlag", center=0, ax=ax, cbar_kws={"label": "Z-score"})
        ax.set_xlabel("Feature")
        ax.set_ylabel("Worst-error sample")
        ax.set_title("Hard-Sample Fingerprint Heatmap")
        put("Hard-Sample Fingerprint Heatmap", fig)

        # 7
        fig, ax = plt.subplots(figsize=(8, max(4.2, 0.33 * len(cols) + 1.5)))
        hard = abs_err >= np.quantile(abs_err, 0.8)
        easy = abs_err <= np.quantile(abs_err, 0.2)
        shift = (norm.loc[hard, cols].mean() - norm.loc[easy, cols].mean()).dropna().sort_values()
        ax.barh(shift.index, shift.values, color=[NATURE_COLORS["red"] if v > 0 else NATURE_COLORS["blue"] for v in shift.values])
        ax.axvline(0, color=NATURE_COLORS["slate"], linestyle="--")
        ax.set_xlabel("Hard cohort mean z-score - easy cohort mean z-score")
        ax.set_title("Easy-vs-Hard Feature Shift")
        put("Easy-vs-Hard Feature Shift", fig)

        # 8
        try:
            filled = norm[cols].fillna(0).to_numpy()
            u, s, vt = np.linalg.svd(filled, full_matrices=False)
            coords = u[:, :2] * s[:2]
            fig, ax = plt.subplots(figsize=(7.2, 5.4))
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=abs_err, cmap="rocket_r", s=44, alpha=0.78, edgecolors="white", linewidth=0.25)
            fig.colorbar(sc, ax=ax, label="Absolute error")
            ax.set_xlabel("Latent component 1")
            ax.set_ylabel("Latent component 2")
            ax.set_title("Residual Leverage Proxy Map")
            put("Residual Leverage Proxy Map", fig)
        except Exception:
            pass

        # 9
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        center_dist = np.sqrt(np.nanmean(np.square(norm[cols].to_numpy()), axis=1))
        dbin = _safe_qbin(center_dist, 8)
        dprof = pd.DataFrame({"bin": dbin, "dist": center_dist, "error": abs_err}).groupby("bin", observed=False).agg(dist=("dist", "mean"), error=("error", "mean"))
        ax.plot(dprof["dist"], dprof["error"], marker="o", color=NATURE_COLORS["purple"], linewidth=2.2)
        ax.set_xlabel("Distance from feature center")
        ax.set_ylabel("Mean absolute error")
        ax.set_title("Distance-to-Center Error Curve")
        put("Distance-to-Center Error Curve", fig)

        # 10
        try:
            filled = norm[cols].fillna(0).to_numpy()
            k = min(8, max(2, n - 1))
            nn = NearestNeighbors(n_neighbors=k).fit(filled)
            _, idx = nn.kneighbors(filled)
            neigh_err = np.mean(abs_err[idx[:, 1:]], axis=1)
            fig, ax = plt.subplots(figsize=(6.8, 5.2))
            ax.scatter(neigh_err, abs_err, color=NATURE_COLORS["teal"], alpha=0.72, edgecolors="white", linewidth=0.25)
            lo, hi = min(neigh_err.min(), abs_err.min()), max(neigh_err.max(), abs_err.max())
            ax.plot([lo, hi], [lo, hi], color=NATURE_COLORS["slate"], linestyle="--")
            ax.set_xlabel("Neighbor mean absolute error")
            ax.set_ylabel("Sample absolute error")
            ax.set_title("Neighbor Error Smoothness")
            put("Neighbor Error Smoothness", fig)
        except Exception:
            pass

        # 11
        if len(cols) >= 2:
            fig, ax = plt.subplots(figsize=(7.2, 5.2))
            f1, f2 = cols[:2]
            surf = pd.DataFrame({"x": _safe_qbin(clean[f1], 8), "y": _safe_qbin(clean[f2], 8), "err": abs_err}).groupby(["x", "y"], observed=False)["err"].mean().unstack()
            sns.heatmap(surf, cmap="rocket_r", ax=ax, cbar_kws={"label": "Mean absolute error"})
            ax.set_xlabel(f2)
            ax.set_ylabel(f1)
            ax.set_title("Two-Feature Error Surface")
            put("Two-Feature Error Surface", fig)

            fig, ax = plt.subplots(figsize=(7.2, 5.2))
            rh = pd.DataFrame({"x": _safe_qbin(clean[f1], 8), "y": _safe_qbin(clean[f2], 8), "resid": residual}).groupby(["x", "y"], observed=False)["resid"].mean().unstack()
            sns.heatmap(rh, cmap="vlag", center=0, ax=ax, cbar_kws={"label": "Mean residual"})
            ax.set_xlabel(f2)
            ax.set_ylabel(f1)
            ax.set_title("Feature-Binned Residual Heatmap")
            put("Feature-Binned Residual Heatmap", fig)

        # 12
        fig, ax = plt.subplots(figsize=(8, max(4.2, 0.32 * len(cols) + 1.5)))
        rows = []
        for col in cols:
            s = clean[col]
            out = np.abs((s - s.mean()) / (s.std() + 1e-9)) > 2
            if out.sum() and (~out).sum():
                rows.append({"Feature": col, "Error risk ratio": abs_err[out].mean() / (abs_err[~out].mean() + 1e-9)})
        risk = pd.DataFrame(rows)
        if not risk.empty:
            risk = risk.sort_values("Error risk ratio")
            ax.barh(risk["Feature"], risk["Error risk ratio"], color=NATURE_COLORS["orange"])
            ax.axvline(1, color=NATURE_COLORS["slate"], linestyle="--")
            ax.set_xlabel("Outlier mean error / non-outlier mean error")
            ax.set_title("Feature Outlier Error Risk")
            put("Feature Outlier Error Risk", fig)

        # 13
        fig, ax = plt.subplots(figsize=(8, max(4.2, 0.33 * len(cols) + 1.5)))
        cover_rows = []
        tol = np.quantile(abs_err, 0.75)
        for col in cols:
            seg = _safe_qbin(clean[col], 4)
            rates = pd.DataFrame({"seg": seg, "ok": abs_err <= tol}).groupby("seg", observed=False)["ok"].mean()
            cover_rows.append({"Feature": col, "Coverage spread": rates.max() - rates.min()})
        cover = pd.DataFrame(cover_rows).dropna().sort_values("Coverage spread")
        ax.barh(cover["Feature"], cover["Coverage spread"], color=NATURE_COLORS["red"])
        ax.set_xlabel("Max-min acceptable-error coverage")
        ax.set_title("Conditional Coverage by Feature Segment")
        put("Conditional Coverage by Feature Segment", fig)

    # 14
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    sns.kdeplot(pred, ax=ax, color=NATURE_COLORS["blue"], fill=True, alpha=0.13, label="Prediction density")
    ax2 = ax.twinx()
    ax2.scatter(pred, residual, s=18, color=NATURE_COLORS["red"], alpha=0.45)
    ax2.axhline(0, color=NATURE_COLORS["slate"], linestyle="--")
    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Density")
    ax2.set_ylabel("Residual")
    ax.set_title("Prediction Density Residual Rug")
    put("Prediction Density Residual Rug", fig)

    # 15
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    deltas = np.linspace(0.1, np.quantile(abs_err, 0.95) + 1e-9, 40)
    huber = [np.mean(np.where(abs_err <= d, 0.5 * abs_err**2, d * (abs_err - 0.5 * d))) for d in deltas]
    ax.plot(deltas, huber, color=NATURE_COLORS["purple"], linewidth=2.2)
    ax.set_xlabel("Huber delta")
    ax.set_ylabel("Mean Huber loss")
    ax.set_title("Robust Loss Sensitivity Curve")
    put("Robust Loss Sensitivity Curve", fig)

    # 16
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    bins = [0, np.quantile(abs_err, 0.5), np.quantile(abs_err, 0.8), np.quantile(abs_err, 0.95), abs_err.max() + 1e-9]
    labels = ["Low", "Moderate", "High", "Extreme"]
    severity = pd.cut(abs_err, bins=np.unique(bins), labels=labels[: max(1, len(np.unique(bins)) - 1)], include_lowest=True)
    counts = pd.Series(severity).value_counts().reindex(labels).dropna()
    ax.bar(counts.index.astype(str), counts.values, color=[NATURE_COLORS["teal"], NATURE_COLORS["gold"], NATURE_COLORS["orange"], NATURE_COLORS["red"]][: len(counts)])
    ax.set_ylabel("Sample count")
    ax.set_title("Error Severity Composition")
    put("Error Severity Composition", fig)

    # 17
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    cutoffs = np.quantile(abs_err, np.linspace(0.55, 0.99, 20))
    worst_mean = [abs_err[abs_err >= c].mean() for c in cutoffs]
    ax.plot(np.linspace(45, 1, len(cutoffs)), worst_mean, color=NATURE_COLORS["red"], linewidth=2.2)
    ax.set_xlabel("Worst sample share (%)")
    ax.set_ylabel("Mean error in worst tail")
    ax.set_title("Worst-Case Error Frontier")
    put("Worst-Case Error Frontier", fig)

    # 18
    fig, ax = plt.subplots(figsize=(8, 4.8))
    order = np.argsort(abs_err)
    colors = np.where(signed_error[order] >= 0, NATURE_COLORS["red"], NATURE_COLORS["blue"])
    ax.bar(np.arange(n), signed_error[order], color=colors, width=1.0, alpha=0.75)
    ax.axhline(0, color=NATURE_COLORS["slate"], linestyle="--")
    ax.set_xlabel("Samples sorted by absolute error")
    ax.set_ylabel("Signed prediction error")
    ax.set_title("Signed Error Waterfall")
    put("Signed Error Waterfall", fig)

    # 19
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    top_share = float(abs_err[abs_err >= np.quantile(abs_err, 0.9)].sum() / (abs_err.sum() + 1e-9))
    vals = [top_share, 1 - top_share]
    ax.pie(vals, labels=["Top 10% samples", "Remaining samples"], autopct="%1.1f%%", colors=[NATURE_COLORS["red"], NATURE_COLORS["gray"]], startangle=90)
    ax.set_title("Error Tail Concentration Donut")
    put("Error Tail Concentration Donut", fig)

    # 20
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    pred_q = pd.Series(pred).rank(pct=True).to_numpy()
    obs_q = pd.Series(y_true).rank(pct=True).to_numpy()
    ax.hexbin(pred_q, obs_q, gridsize=20, cmap="mako", mincnt=1)
    ax.plot([0, 1], [0, 1], color=NATURE_COLORS["red"], linestyle="--")
    ax.set_xlabel("Predicted percentile")
    ax.set_ylabel("Observed percentile")
    ax.set_title("Percentile Calibration Map")
    put("Percentile Calibration Map", fig)

    return plots


def _bootstrap_diagnostics(model, X_train, y_train, X_test, y_test, ptype, n_boot=24, sample_limit=80):
    X_train_df = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
    X_test_df = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test)
    y_test_v = _series_values(y_test)
    idx_subset = np.linspace(0, len(X_test_df) - 1, min(sample_limit, len(X_test_df)), dtype=int)
    preds = []
    scores = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sample_idx = rng.integers(0, len(X_train_df), len(X_train_df))
        try:
            fitted = clone(model).fit(X_train_df.iloc[sample_idx], _series_values(y_train)[sample_idx])
            pred_test = _model_response(fitted, X_test_df)
            preds.append(pred_test[idx_subset])
            if _ptype_is_classification(ptype):
                label_pred = fitted.predict(X_test_df)
                scores.append(float(accuracy_score(y_test, label_pred)))
            else:
                label_pred = np.asarray(pred_test, dtype=float)
                scores.append(float(r2_score(y_test_v, label_pred)))
        except Exception:
            continue
    if not preds or not scores:
        return None
    pred_arr = np.asarray(preds, dtype=float)
    return {
        "indices": idx_subset,
        "y_true": y_test_v[idx_subset],
        "pred_mean": pred_arr.mean(axis=0),
        "pred_low": np.quantile(pred_arr, 0.025, axis=0),
        "pred_high": np.quantile(pred_arr, 0.975, axis=0),
        "scores": np.asarray(scores, dtype=float),
        "pred_samples": pred_arr,
    }


def _plot_bootstrap_stability(diag, ptype, fmt="png", dpi=300):
    if diag is None:
        return None
    apply_nature_style()
    metric_name = "Accuracy" if _ptype_is_classification(ptype) else "R2"
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    sns.violinplot(x=diag["scores"], ax=ax, color=NATURE_COLORS["teal"], inner=None, linewidth=0)
    sns.stripplot(x=diag["scores"], ax=ax, color=NATURE_COLORS["slate"], alpha=0.45, size=4)
    ax.axvline(float(np.mean(diag["scores"])), color=NATURE_COLORS["red"], linestyle="--", linewidth=1.5, label=f"Mean = {np.mean(diag['scores']):.3f}")
    ax.set_xlabel(f"Bootstrap {metric_name}")
    ax.set_title("Bootstrap Stability")
    ax.legend()
    return fig_to_base64(fig, fmt, dpi)


def _plot_bootstrap_uncertainty_band(diag, ptype, fmt="png", dpi=300):
    if diag is None:
        return None
    apply_nature_style()
    order = np.argsort(diag["y_true"] if not _ptype_is_classification(ptype) else diag["pred_mean"])
    x_axis = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    ax.fill_between(x_axis, diag["pred_low"][order], diag["pred_high"][order], color=NATURE_COLORS["teal"], alpha=0.18, label="95% bootstrap band")
    ax.plot(x_axis, diag["pred_mean"][order], color=NATURE_COLORS["teal"], linewidth=2.0, label="Bootstrap mean prediction")
    if _ptype_is_classification(ptype):
        encoded_true = pd.factorize(diag["y_true"][order])[0]
        ax.scatter(x_axis, encoded_true, color=NATURE_COLORS["orange"], s=28, alpha=0.7, label="True label (encoded)")
        ax.set_ylabel("Positive-class probability / encoded label")
        ax.set_title("Bootstrap Classification Uncertainty Band")
    else:
        ax.plot(x_axis, diag["y_true"][order], color=NATURE_COLORS["slate"], linewidth=1.8, label="Observed")
        coverage = np.mean((diag["y_true"] >= diag["pred_low"]) & (diag["y_true"] <= diag["pred_high"]))
        ax.set_ylabel("Target / prediction")
        ax.set_title(f"Bootstrap Regression Uncertainty Band (coverage={coverage:.2f})")
    ax.set_xlabel("Ordered test sample")
    ax.legend()
    return fig_to_base64(fig, fmt, dpi)


def _ensemble_prediction_interval(model, X, lower_q=0.05, upper_q=0.95):
    estimators = getattr(model, "estimators_", None)
    if estimators is None:
        return None
    try:
        members = [est for est in np.ravel(estimators) if est is not None and hasattr(est, "predict")]
        if len(members) < 5:
            return None
        X_member = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        member_preds = np.vstack([np.asarray(est.predict(X_member), dtype=float) for est in members])
        return {
            "mean": np.nanmean(member_preds, axis=0),
            "median": np.nanquantile(member_preds, 0.50, axis=0),
            "lower": np.nanquantile(member_preds, lower_q, axis=0),
            "upper": np.nanquantile(member_preds, upper_q, axis=0),
            "std": np.nanstd(member_preds, axis=0),
            "member_count": len(members),
            "member_predictions": member_preds,
        }
    except Exception:
        return None


def _plot_probabilistic_prediction_interval(model, X_test, y_true, pred, fmt="png", dpi=300):
    interval = _ensemble_prediction_interval(model, X_test)
    if not interval:
        return None
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    low = np.asarray(interval["lower"], dtype=float)
    high = np.asarray(interval["upper"], dtype=float)
    mean_pred = np.asarray(interval["mean"], dtype=float)
    width = high - low
    covered = (y_true >= low) & (y_true <= high)
    order = np.argsort(mean_pred)
    show = order[: min(120, len(order))]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    x = np.arange(len(show))
    axes[0].fill_between(x, low[show], high[show], color=NATURE_COLORS["purple"], alpha=0.18, label="5%-95% ensemble interval")
    axes[0].plot(x, mean_pred[show], color=NATURE_COLORS["purple"], linewidth=2.0, label="Ensemble mean")
    axes[0].plot(x, pred[show], color=NATURE_COLORS["blue"], linewidth=1.3, alpha=0.85, label="Model prediction")
    axes[0].scatter(x, y_true[show], c=covered[show], cmap="coolwarm_r", s=34, edgecolors="white", linewidth=0.3, label="Observed")
    axes[0].set_xlabel("Test samples sorted by ensemble mean")
    axes[0].set_ylabel("Target")
    axes[0].set_title(f"Probabilistic Prediction Interval (coverage={covered.mean():.2f})")
    axes[0].legend(fontsize=8)

    try:
        bins = pd.qcut(pd.Series(width), q=min(6, len(np.unique(width))), duplicates="drop")
        profile = pd.DataFrame({"bin": bins, "covered": covered.astype(float), "width": width, "absolute_error": np.abs(y_true - pred)}).groupby("bin", observed=False).agg(
            coverage=("covered", "mean"),
            mean_width=("width", "mean"),
            mean_abs_error=("absolute_error", "mean"),
        )
        xpos = np.arange(len(profile))
        axes[1].bar(xpos - 0.18, profile["coverage"], width=0.36, color=NATURE_COLORS["teal"], label="Coverage")
        axes[1].bar(xpos + 0.18, profile["mean_abs_error"] / (profile["mean_abs_error"].max() + 1e-9), width=0.36, color=NATURE_COLORS["orange"], label="Normalized MAE")
        ax2 = axes[1].twinx()
        ax2.plot(xpos, profile["mean_width"], color=NATURE_COLORS["purple"], marker="o", linewidth=1.8, label="Mean interval width")
        axes[1].set_xticks(xpos)
        axes[1].set_xticklabels([f"W{i+1}" for i in xpos])
        axes[1].set_ylim(0, 1.05)
        axes[1].set_xlabel("Interval-width stratum")
        axes[1].set_ylabel("Coverage / normalized MAE")
        ax2.set_ylabel("Mean interval width")
        axes[1].set_title("Interval Reliability by Width")
        axes[1].legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
    except Exception:
        axes[1].scatter(width, np.abs(y_true - pred), c=covered.astype(int), cmap="coolwarm_r", s=34, edgecolors="white", linewidth=0.3)
        axes[1].set_xlabel("Interval width")
        axes[1].set_ylabel("Absolute error")
        axes[1].set_title("Uncertainty Width vs Error")
    return fig_to_base64(fig, fmt, dpi)


def _applicability_domain_payload(X_train, X_test, residuals=None, k=5):
    X_train_arr = np.asarray(X_train, dtype=float)
    X_test_arr = np.asarray(X_test, dtype=float)
    if len(X_train_arr) < 8:
        return None
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X_train_arr)))
    nn.fit(X_train_arr)
    train_dist, _ = nn.kneighbors(X_train_arr)
    train_mean = train_dist[:, 1:].mean(axis=1) if train_dist.shape[1] > 1 else train_dist[:, 0]
    test_nn = NearestNeighbors(n_neighbors=min(k, len(X_train_arr)))
    test_nn.fit(X_train_arr)
    test_dist, _ = test_nn.kneighbors(X_test_arr)
    test_mean = test_dist.mean(axis=1)
    threshold = float(np.quantile(train_mean, 0.95))
    payload = {
        "train_distance": train_mean,
        "test_distance": test_mean,
        "threshold": threshold,
        "test_in_domain": test_mean <= threshold,
    }
    if residuals is not None:
        payload["abs_residual"] = np.abs(np.asarray(residuals, dtype=float))
    return payload


def _plot_ad_histogram(ad_payload, fmt="png", dpi=300):
    if not ad_payload:
        return None
    apply_nature_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    sns.kdeplot(ad_payload["train_distance"], ax=ax, fill=True, alpha=0.18, color=NATURE_COLORS["blue"], label="Train")
    sns.kdeplot(ad_payload["test_distance"], ax=ax, fill=True, alpha=0.18, color=NATURE_COLORS["orange"], label="Test")
    ax.axvline(ad_payload["threshold"], color=NATURE_COLORS["red"], linestyle="--", linewidth=1.4, label=f"AD threshold = {ad_payload['threshold']:.3f}")
    ax.set_xlabel("Mean kNN distance to training domain")
    ax.set_title("Applicability Domain Distance")
    ax.legend()
    return fig_to_base64(fig, fmt, dpi)


def _plot_ad_reliability(ad_payload, fmt="png", dpi=300):
    if not ad_payload or "abs_residual" not in ad_payload:
        return None
    apply_nature_style()
    test_dist = np.asarray(ad_payload["test_distance"], dtype=float)
    abs_res = np.asarray(ad_payload["abs_residual"], dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    colors = np.where(test_dist <= ad_payload["threshold"], NATURE_COLORS["teal"], NATURE_COLORS["red"])
    ax.scatter(test_dist, abs_res, c=colors, alpha=0.72, edgecolors="white", linewidth=0.35)
    ax.axvline(ad_payload["threshold"], color=NATURE_COLORS["red"], linestyle="--", linewidth=1.4)
    if len(test_dist) >= 10:
        order = np.argsort(test_dist)
        smooth = pd.Series(abs_res[order]).rolling(max(4, len(abs_res) // 8), min_periods=1).mean()
        ax.plot(test_dist[order], smooth, color=NATURE_COLORS["slate"], linewidth=2.0)
    ax.set_xlabel("Distance from training domain")
    ax.set_ylabel("Absolute residual")
    ax.set_title("Applicability Domain Reliability")
    return fig_to_base64(fig, fmt, dpi)


def _plot_ad_coverage(ad_payload, fmt="png", dpi=300):
    if not ad_payload:
        return None
    apply_nature_style()
    in_count = int(np.sum(ad_payload["test_in_domain"]))
    out_count = int(len(ad_payload["test_in_domain"]) - in_count)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.bar(["In-domain", "Out-of-domain"], [in_count, out_count], color=[NATURE_COLORS["teal"], NATURE_COLORS["orange"]], width=0.58)
    ax.set_ylabel("Test samples")
    ax.set_title("Applicability Domain Coverage")
    total = max(in_count + out_count, 1)
    ax.text(0, in_count + total * 0.02, f"{in_count / total:.1%}", ha="center", fontsize=10, fontweight="bold")
    ax.text(1, out_count + total * 0.02, f"{out_count / total:.1%}", ha="center", fontsize=10, fontweight="bold")
    return fig_to_base64(fig, fmt, dpi)


def _bh_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.array([])
    order = np.argsort(pvals)
    ranked = pvals[order]
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = ranked[i] * n / rank
        running = min(running, value)
        adjusted[i] = min(running, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return out


def _paired_cohens_d(a, b):
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sd = np.std(diff, ddof=1) if len(diff) > 1 else np.std(diff)
    if sd < 1e-12:
        return 0.0
    return float(np.mean(diff) / sd)


def _per_sample_loss(y_true, pred, ptype):
    y_true_v = _series_values(y_true)
    pred_v = _series_values(pred)
    if _ptype_is_classification(ptype):
        return (y_true_v != pred_v).astype(float)
    return np.abs(y_true_v - pred_v)


def _build_significance_df(trained_models_dict, X_test, y_test, ptype, baseline_name):
    baseline_pred = trained_models_dict[baseline_name].predict(X_test)
    baseline_loss = _per_sample_loss(y_test, baseline_pred, ptype)
    rows = []
    for model_name, model_obj in trained_models_dict.items():
        if model_name == baseline_name or not hasattr(model_obj, "predict"):
            continue
        pred = model_obj.predict(X_test)
        loss = _per_sample_loss(y_test, pred, ptype)
        diff = loss - baseline_loss
        try:
            _, p_val = wilcoxon(diff)
            p_val = float(p_val)
        except Exception:
            p_val = 1.0
        rows.append(
            {
                "Model": model_name,
                "Mean Loss Diff": float(np.mean(diff)),
                "PValue": p_val,
                "CohensD": _paired_cohens_d(loss, baseline_loss),
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["AdjustedPValue"] = _bh_adjust(out["PValue"].to_numpy(dtype=float))
    out["Significant"] = out["AdjustedPValue"] < 0.05
    return out.sort_values("AdjustedPValue").reset_index(drop=True)


def _build_wilcoxon_plot(significance_df, baseline_name, ptype, fmt="png", dpi=300):
    apply_nature_style()
    fig, ax = plt.subplots(figsize=(8.8, max(4.6, 0.38 * len(significance_df) + 2.3)))
    plot_df = significance_df.copy().iloc[::-1]
    ypos = np.arange(len(plot_df))
    colors = [NATURE_COLORS["red"] if sig else NATURE_COLORS["gray"] for sig in plot_df["Significant"]]
    ax.barh(ypos, -np.log10(np.clip(plot_df["AdjustedPValue"], 1e-12, 1.0)), color=colors, alpha=0.9)
    ax.set_yticks(ypos)
    ax.set_yticklabels(plot_df["Model"])
    ax.axvline(-np.log10(0.05), color=NATURE_COLORS["slate"], linestyle="--", linewidth=1.2, label="FDR 0.05")
    ax.set_xlabel("-log10 adjusted p-value")
    ax.set_title(f"Wilcoxon Significance vs {baseline_name}")
    for y, (_, row) in zip(ypos, plot_df.iterrows()):
        ax.text(
            0.02,
            y,
            f"p={row['PValue']:.3g} | BH={row['AdjustedPValue']:.3g}",
            va="center",
            ha="left",
            fontsize=7.5,
            color="white" if row["Significant"] else NATURE_COLORS["slate"],
            transform=ax.get_yaxis_transform(),
        )
    ax.legend(loc="lower right")
    return fig_to_base64(fig, fmt, dpi)


def _build_effect_size_plot(significance_df, baseline_name, fmt="png", dpi=300):
    apply_nature_style()
    fig, ax = plt.subplots(figsize=(8.8, max(4.6, 0.38 * len(significance_df) + 2.3)))
    plot_df = significance_df.copy().iloc[::-1]
    ypos = np.arange(len(plot_df))
    colors = [NATURE_COLORS["blue"] if d < 0 else NATURE_COLORS["orange"] for d in plot_df["CohensD"]]
    ax.barh(ypos, plot_df["CohensD"], color=colors, alpha=0.92)
    ax.set_yticks(ypos)
    ax.set_yticklabels(plot_df["Model"])
    ax.axvline(0, color=NATURE_COLORS["slate"], linewidth=1.2)
    for thr in [0.2, 0.5, 0.8]:
        ax.axvline(thr, color="#cbd5e1", linestyle="--", linewidth=0.9)
        ax.axvline(-thr, color="#cbd5e1", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Paired Cohen's d")
    ax.set_title(f"Effect Size vs {baseline_name}")
    return fig_to_base64(fig, fmt, dpi)


def _parse_interval_edges(y_values, interval_bins=4, interval_edges=None):
    series = np.asarray(y_values, dtype=float)
    if interval_edges:
        if isinstance(interval_edges, str):
            edges = [float(x.strip()) for x in str(interval_edges).split(",") if x.strip()]
        else:
            edges = [float(x) for x in interval_edges]
        if len(edges) < 2:
            raise ValueError("Custom interval edges require at least two values.")
        edges = sorted(edges)
        return np.asarray(edges, dtype=float)
    bins = max(2, int(interval_bins or 4))
    low, high = float(np.min(series)), float(np.max(series))
    if abs(high - low) < 1e-12:
        high = low + 1.0
    return np.linspace(low, high, bins + 1)


def _interval_mae_frame(y_train, train_pred, y_test, test_pred, interval_bins=4, interval_edges=None):
    edges = _parse_interval_edges(np.concatenate([_series_values(y_train), _series_values(y_test)]), interval_bins, interval_edges)
    labels = [f"{edges[i]:.2f} ~ {edges[i+1]:.2f}" for i in range(len(edges) - 1)]

    def mae_by_bins(y_true, y_hat):
        idx = np.digitize(np.asarray(y_true, dtype=float), edges[1:-1], right=False)
        vals = []
        for i in range(len(labels)):
            mask = idx == i
            if np.any(mask):
                vals.append(float(np.mean(np.abs(np.asarray(y_true)[mask] - np.asarray(y_hat)[mask]))))
            else:
                vals.append(0.0)
        return vals

    return pd.DataFrame(
        {
            "Label": labels,
            "Train MAE": mae_by_bins(_series_values(y_train), train_pred),
            "Test MAE": mae_by_bins(_series_values(y_test), test_pred),
        }
    )


def _plot_interval_radial(interval_df, title="Target Interval Stability", fmt="png", dpi=300):
    apply_nature_style()
    labels = interval_df["Label"].tolist()
    train = interval_df["Train MAE"].to_numpy(dtype=float)
    test = interval_df["Test MAE"].to_numpy(dtype=float)
    sector_centers = np.deg2rad([45, -45, -135, 135])
    if len(labels) != 4:
        sector_centers = np.linspace(np.pi / 4, np.pi / 4 - 2 * np.pi, len(labels), endpoint=False)
    max_val = max(float(np.max(train)), float(np.max(test)), 1.0)
    fig, ax = plt.subplots(figsize=(7.6, 6.2), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, max_val * 1.45)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(color="#d0d0d0", linestyle="--", alpha=0.8)
    ring_vals = np.linspace(0, max_val * 1.2, 5)[1:]
    for val in ring_vals:
        ax.plot(np.linspace(0, 2 * np.pi, 240), np.full(240, val), color="#d0d0d0", linewidth=0.7, linestyle="--", alpha=0.8)
    for idx, theta in enumerate(sector_centers):
        width = np.deg2rad(26)
        ax.bar(theta - np.deg2rad(8), train[idx], width=width, color="#5660A9", alpha=0.9, edgecolor="#555555", linewidth=1.0)
        ax.bar(theta + np.deg2rad(8), test[idx], width=width, color="#D56B57", alpha=0.9, edgecolor="#555555", linewidth=1.0)
        ax.text(theta, max_val * 1.34, labels[idx], ha="center", va="center", fontsize=8.5, color=NATURE_COLORS["slate"], path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        ax.text(theta - np.deg2rad(8), train[idx] + max_val * 0.05, f"{train[idx]:.2f}", ha="center", va="center", fontsize=8, color="white" if train[idx] > 1.0 else "black", path_effects=[pe.withStroke(linewidth=2, foreground="#5660A9")])
        ax.text(theta + np.deg2rad(8), test[idx] + max_val * 0.05, f"{test[idx]:.2f}", ha="center", va="center", fontsize=8, color="white" if test[idx] > 1.0 else "black", path_effects=[pe.withStroke(linewidth=2, foreground="#D56B57")])
    ax.set_title(title, va="bottom", pad=22)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="#5660A9", lw=8, label="Train MAE"),
            plt.Line2D([0], [0], color="#D56B57", lw=8, label="Test MAE"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
    )
    return fig_to_base64(fig, fmt, dpi)


def _plot_interval_bar(interval_df, fmt="png", dpi=300):
    apply_nature_style()
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    xpos = np.arange(len(interval_df))
    width = 0.36
    ax.bar(xpos - width / 2, interval_df["Train MAE"], width, color="#5660A9", label="Train")
    ax.bar(xpos + width / 2, interval_df["Test MAE"], width, color="#D56B57", label="Test")
    ax.set_xticks(xpos)
    ax.set_xticklabels(interval_df["Label"], rotation=15, ha="right")
    ax.set_ylabel("MAE")
    ax.set_title("Interval Stability Analysis")
    ax.legend()
    return fig_to_base64(fig, fmt, dpi)


def _plot_kfold_stability(model, X_train, y_train, ptype, fmt="png", dpi=300):
    apply_nature_style()
    scoring = "accuracy" if _ptype_is_classification(ptype) else "r2"
    try:
        scores = cross_val_score(clone(model), X_train, y_train, cv=5, scoring=scoring, n_jobs=1)
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(range(1, len(scores) + 1), scores, marker="o", color=NATURE_COLORS["teal"], linewidth=2)
    ax.axhline(np.mean(scores), color=NATURE_COLORS["red"], linestyle="--", linewidth=1.5, label=f"Mean = {np.mean(scores):.3f}")
    ax.fill_between(range(1, len(scores) + 1), np.mean(scores) - np.std(scores), np.mean(scores) + np.std(scores), color=NATURE_COLORS["teal"], alpha=0.15)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy" if _ptype_is_classification(ptype) else "R2")
    ax.set_title("K-Fold Stability")
    ax.legend()
    return fig_to_base64(fig, fmt, dpi)


def _plot_y_randomization(model, X_train, y_train, X_test, y_test, ptype, fmt="png", dpi=300):
    apply_nature_style()
    rng = np.random.default_rng(42)
    null_scores = []
    for _ in range(16):
        y_perm = rng.permutation(_series_values(y_train))
        try:
            fitted = clone(model).fit(X_train, y_perm)
            pred = fitted.predict(X_test)
            score = accuracy_score(y_test, pred) if _ptype_is_classification(ptype) else r2_score(y_test, pred)
            null_scores.append(float(score))
        except Exception:
            continue
    if not null_scores:
        return None
    actual_pred = model.predict(X_test)
    actual_score = accuracy_score(y_test, actual_pred) if _ptype_is_classification(ptype) else r2_score(y_test, actual_pred)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sns.histplot(null_scores, bins=10, kde=True, ax=ax, color=NATURE_COLORS["gray"])
    ax.axvline(actual_score, color=NATURE_COLORS["red"], linewidth=2, label=f"Actual = {actual_score:.3f}")
    ax.set_xlabel("Randomized score")
    ax.set_title("Y-Randomization Test")
    ax.legend()
    return fig_to_base64(fig, fmt, dpi)


def _plot_williams(model, X_train, y_train, X_test, y_test, fmt="png", dpi=300):
    apply_nature_style()
    X_train_arr = np.asarray(X_train, dtype=float)
    X_test_arr = np.asarray(X_test, dtype=float)
    x_aug = np.column_stack([np.ones(len(X_train_arr)), X_train_arr])
    try:
        inv_xtx = np.linalg.pinv(x_aug.T @ x_aug)
    except Exception:
        return None
    train_pred = _series_values(model.predict(X_train))
    test_pred = _series_values(model.predict(X_test))
    train_resid = _series_values(y_train) - train_pred
    test_resid = _series_values(y_test) - test_pred
    resid_all = np.concatenate([train_resid, test_resid])
    std_resid = resid_all / (np.std(resid_all) + 1e-9)
    all_x = np.vstack([X_train_arr, X_test_arr])
    h = np.sum((np.column_stack([np.ones(len(all_x)), all_x]) @ inv_xtx) * np.column_stack([np.ones(len(all_x)), all_x]), axis=1)
    h_star = 3 * (x_aug.shape[1]) / max(len(X_train_arr), 1)
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.scatter(h[: len(X_train_arr)], std_resid[: len(X_train_arr)], color=NATURE_COLORS["blue"], alpha=0.5, label="Train", edgecolors="white", linewidth=0.3)
    ax.scatter(h[len(X_train_arr):], std_resid[len(X_train_arr):], color=NATURE_COLORS["orange"], alpha=0.75, label="Test", edgecolors="white", linewidth=0.3)
    ax.axhline(3, color=NATURE_COLORS["red"], linestyle="--", linewidth=1.2)
    ax.axhline(-3, color=NATURE_COLORS["red"], linestyle="--", linewidth=1.2)
    ax.axvline(h_star, color=NATURE_COLORS["purple"], linestyle="--", linewidth=1.2, label=f"h* = {h_star:.3f}")
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Standardized residual")
    ax.set_title("Williams Plot")
    ax.legend()
    return fig_to_base64(fig, fmt, dpi)


def _plot_meta_history(meta_history, fmt="png", dpi=300):
    if not meta_history:
        return None
    apply_nature_style()
    history_df = pd.DataFrame(meta_history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    axes[0].plot(history_df["epoch"], history_df["score"], marker="o", color=NATURE_COLORS["teal"], label="Score")
    axes[0].plot(history_df["epoch"], history_df["best_score"], marker="s", color=NATURE_COLORS["red"], label="Best")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("CV score")
    axes[0].set_title("Hyperparameter Search Score")
    axes[0].legend()
    axes[1].plot(history_df["epoch"], history_df["n_estimators"], marker="o", color=NATURE_COLORS["blue"], label="n_estimators")
    axes[1].plot(history_df["epoch"], history_df["max_depth"], marker="s", color=NATURE_COLORS["gold"], label="max_depth")
    axes[1].set_xlabel("Iteration")
    axes[1].set_title("Hyperparameter Search Trajectory")
    axes[1].legend()
    return fig_to_base64(fig, fmt, dpi)


def _plot_meta_heatmap(meta_history, fmt="png", dpi=300):
    if not meta_history:
        return None
    history_df = pd.DataFrame(meta_history)
    if history_df["n_estimators"].nunique() < 2 or history_df["max_depth"].nunique() < 2:
        return None
    pivot = history_df.pivot_table(index="max_depth", columns="n_estimators", values="score", aggfunc="max")
    apply_nature_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sns.heatmap(pivot.sort_index(ascending=False), cmap="viridis", annot=True, fmt=".3f", ax=ax)
    ax.set_title("Hyperparameter Search Heatmap")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("max_depth")
    return fig_to_base64(fig, fmt, dpi)


def _plot_meta_surface(meta_history, fmt="png", dpi=300):
    if not meta_history:
        return None
    history_df = pd.DataFrame(meta_history)
    if history_df["n_estimators"].nunique() < 3 or history_df["max_depth"].nunique() < 3:
        return None
    apply_nature_style()
    fig = plt.figure(figsize=(8.4, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    x = history_df["n_estimators"].to_numpy(dtype=float)
    y = history_df["max_depth"].to_numpy(dtype=float)
    z = history_df["score"].to_numpy(dtype=float)
    tri = ax.plot_trisurf(x, y, z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.scatter(x, y, z, color="black", s=24, alpha=0.75)
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("max_depth")
    ax.set_zlabel("CV score")
    ax.set_title("Hyperparameter Response Surface")
    fig.colorbar(tri, ax=ax, shrink=0.6, aspect=12)
    return fig_to_base64(fig, fmt, dpi)


def _plot_meta_top_trials(meta_history, fmt="png", dpi=300):
    if not meta_history:
        return None
    history_df = pd.DataFrame(meta_history).sort_values("score", ascending=False).head(8).copy()
    apply_nature_style()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    labels = [f"n={int(r.n_estimators)} | d={int(r.max_depth)}" for _, r in history_df.iterrows()]
    ax.barh(labels[::-1], history_df["score"].to_numpy(dtype=float)[::-1], color=NATURE_COLORS["purple"])
    ax.set_xlabel("CV score")
    ax.set_title("Top Hyperparameter Trials")
    return fig_to_base64(fig, fmt, dpi)


def _plot_meta_score_distribution(meta_history, fmt="png", dpi=300):
    if not meta_history:
        return None
    df = pd.DataFrame(meta_history)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sns.histplot(df["score"], kde=True, color=NATURE_COLORS["teal"], ax=ax)
    ax.axvline(df["score"].max(), color=NATURE_COLORS["red"], linestyle="--", label="Best")
    ax.set_title("Meta Search Score Distribution")
    ax.set_xlabel("CV score")
    ax.legend()
    return fig_to_base64(fig, fmt, dpi)


def _plot_meta_improvement_curve(meta_history, fmt="png", dpi=300):
    if not meta_history:
        return None
    df = pd.DataFrame(meta_history)
    first = float(df["best_score"].iloc[0])
    df["improvement"] = df["best_score"] - first
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.step(df["epoch"], df["improvement"], where="post", color=NATURE_COLORS["red"], linewidth=2.2)
    ax.scatter(df["epoch"], df["improvement"], color=NATURE_COLORS["red"], s=36, edgecolors="white")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best score improvement")
    ax.set_title("Meta Search Improvement Curve")
    return fig_to_base64(fig, fmt, dpi)


def _plot_meta_exploration_exploitation(meta_history, fmt="png", dpi=300):
    if not meta_history:
        return None
    df = pd.DataFrame(meta_history).copy()
    df["param_step"] = np.sqrt(df["n_estimators"].diff().fillna(0) ** 2 + df["max_depth"].diff().fillna(0) ** 2)
    df["score_gain"] = df["best_score"].diff().fillna(0)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    axes[0].plot(df["epoch"], df["param_step"], color=NATURE_COLORS["blue"], marker="o")
    axes[0].set_title("Search Step Size")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Parameter movement")
    axes[1].scatter(df["param_step"], df["score_gain"], color=NATURE_COLORS["purple"], s=48, edgecolors="white")
    axes[1].axhline(0, color=NATURE_COLORS["slate"], linestyle="--")
    axes[1].set_title("Exploration vs Improvement")
    axes[1].set_xlabel("Parameter movement")
    axes[1].set_ylabel("Best-score gain")
    return fig_to_base64(fig, fmt, dpi)


def _plot_meta_algorithm_overview(meta_histories, fmt="png", dpi=300):
    if not meta_histories:
        return {}
    rows = []
    for name, history in meta_histories.items():
        for row in history or []:
            item = dict(row)
            item["combo"] = name
            rows.append(item)
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    plots = {}
    if {"algorithm", "model", "best_score"}.issubset(df.columns):
        final = df.sort_values("epoch").groupby(["algorithm", "model"], as_index=False).tail(1)
        pivot = final.pivot_table(index="algorithm", columns="model", values="best_score", aggfunc="max")
        fig, ax = plt.subplots(figsize=(9, max(4.8, 0.35 * len(pivot) + 2)))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", linewidths=0.45, ax=ax)
        ax.set_title("Meta Algorithm x Model Best Score Matrix")
        plots["Meta Algorithm Model Score Matrix"] = fig_to_base64(fig, fmt, dpi)

        rank = final.copy()
        rank["Rank"] = rank["best_score"].rank(ascending=False, method="min")
        rank = rank.sort_values("Rank").head(30)
        fig, ax = plt.subplots(figsize=(9, max(4.6, 0.32 * len(rank) + 2)))
        labels = rank["algorithm"] + "-" + rank["model"]
        ax.barh(labels[::-1], rank["best_score"][::-1], color=NATURE_COLORS["orange"])
        ax.set_xlabel("Best CV score")
        ax.set_title("Meta-Model Combination Leaderboard")
        plots["Meta-Model Combination Leaderboard"] = fig_to_base64(fig, fmt, dpi)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for combo, grp in df.groupby("combo"):
        ax.plot(grp["epoch"], grp["best_score"], linewidth=1.7, alpha=0.78, label=combo)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best CV score")
    ax.set_title("Meta Optimization Convergence Comparison")
    if df["combo"].nunique() <= 12:
        ax.legend(fontsize=7, ncol=2)
    plots["Meta Convergence Comparison"] = fig_to_base64(fig, fmt, dpi)

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    sc = ax.scatter(df["n_estimators"], df["max_depth"], c=df["score"], cmap="viridis", s=55, alpha=0.78, edgecolors="white")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("max_depth")
    ax.set_title("Meta Search Landscape Scatter")
    fig.colorbar(sc, ax=ax, label="CV score")
    plots["Meta Search Landscape Scatter"] = fig_to_base64(fig, fmt, dpi)

    final = df.sort_values("epoch").groupby("combo", as_index=False).tail(1).copy()
    final["Efficiency"] = final["best_score"] / final["epoch"].clip(lower=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.boxplot(data=df, x="algorithm" if "algorithm" in df.columns else "combo", y="score", ax=axes[0], color="#c7dcef")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_title("Meta Score Stability by Algorithm")
    top_eff = final.sort_values("Efficiency", ascending=False).head(20)
    axes[1].barh(top_eff["combo"][::-1], top_eff["Efficiency"][::-1], color=NATURE_COLORS["teal"])
    axes[1].set_title("Search Efficiency Ranking")
    axes[1].set_xlabel("Best score / iterations")
    plots["Meta Stability and Efficiency"] = fig_to_base64(fig, fmt, dpi)
    return plots


def _feature_importance_plot(model, feature_names, fmt="png", dpi=300):
    values = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        values = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
    if values is None or len(values) != len(feature_names):
        return None
    order = np.argsort(values)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.42 * len(order) + 2)))
    ax.barh(np.array(feature_names)[order][::-1], values[order][::-1], color=NATURE_COLORS["teal"])
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importance")
    return fig_to_base64(fig, fmt, dpi)


def _learning_curve_plot(model, X_train, y_train, ptype, fmt="png", dpi=300):
    try:
        scoring = "accuracy" if _ptype_is_classification(ptype) else "r2"
        sizes, train_scores, test_scores = learning_curve(
            model,
            X_train,
            y_train,
            cv=3,
            scoring=scoring,
            train_sizes=np.linspace(0.2, 1.0, 5),
            n_jobs=1,
        )
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.plot(sizes, train_scores.mean(axis=1), marker="o", color=NATURE_COLORS["blue"], label="Train")
        ax.fill_between(
            sizes,
            train_scores.mean(axis=1) - train_scores.std(axis=1),
            train_scores.mean(axis=1) + train_scores.std(axis=1),
            color=NATURE_COLORS["blue"],
            alpha=0.15,
        )
        ax.plot(sizes, test_scores.mean(axis=1), marker="s", color=NATURE_COLORS["orange"], label="Validation")
        ax.fill_between(
            sizes,
            test_scores.mean(axis=1) - test_scores.std(axis=1),
            test_scores.mean(axis=1) + test_scores.std(axis=1),
            color=NATURE_COLORS["orange"],
            alpha=0.15,
        )
        ax.set_xlabel("Training samples")
        ax.set_ylabel("Score")
        ax.set_title("Learning Curve")
        ax.legend()
        return fig_to_base64(fig, fmt, dpi)
    except Exception:
        return None


def generate_model_plots(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    ptype,
    fmt="png",
    dpi=300,
    interval_bins=4,
    interval_edges=None,
    meta_history=None,
):
    apply_nature_style()
    plots = {}

    ypred = model.predict(X_test)
    train_pred = model.predict(X_train) if hasattr(model, "predict") else None
    bootstrap_diag = _bootstrap_diagnostics(model, X_train, y_train, X_test, y_test, ptype)

    if _ptype_is_classification(ptype):
        labels = np.unique(np.concatenate([_series_values(y_train), _series_values(y_test)]))
        cm = confusion_matrix(y_test, ypred, labels=labels)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title("Confusion Matrix")
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        plots["Confusion Matrix"] = fig_to_base64(fig, fmt, dpi)

        fig, ax = plt.subplots(figsize=(6, 5))
        norm_cm = confusion_matrix(y_test, ypred, labels=labels, normalize="true")
        sns.heatmap(norm_cm, annot=True, fmt=".2f", cmap="mako", ax=ax, cbar=False)
        ax.set_title("Normalized Confusion Matrix")
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        plots["Normalized Confusion Matrix"] = fig_to_base64(fig, fmt, dpi)

        class_profile = _plot_classwise_error_profile(y_test, ypred, fmt, dpi)
        if class_profile:
            plots["Class-wise Error Profile"] = class_profile

        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)
                if y_prob.shape[1] == 2:
                    positive_prob = y_prob[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, positive_prob)
                    roc_auc = auc(fpr, tpr)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.plot(fpr, tpr, color=NATURE_COLORS["red"], label=f"AUC = {roc_auc:.3f}")
                    ax.plot([0, 1], [0, 1], linestyle="--", color=NATURE_COLORS["slate"])
                    ax.set_xlabel("False positive rate")
                    ax.set_ylabel("True positive rate")
                    ax.set_title("ROC Curve")
                    ax.legend(loc="lower right")
                    plots["ROC Curve"] = fig_to_base64(fig, fmt, dpi)

                    precision, recall, _ = precision_recall_curve(y_test, positive_prob)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.plot(recall, precision, color=NATURE_COLORS["purple"])
                    ax.set_xlabel("Recall")
                    ax.set_ylabel("Precision")
                    ax.set_title("Precision-Recall Curve")
                    plots["Precision-Recall Curve"] = fig_to_base64(fig, fmt, dpi)

                    frac_pos, mean_pred = calibration_curve(y_test, positive_prob, n_bins=8)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.plot(mean_pred, frac_pos, marker="o", color=NATURE_COLORS["teal"])
                    ax.plot([0, 1], [0, 1], linestyle="--", color=NATURE_COLORS["slate"])
                    ax.set_xlabel("Mean predicted probability")
                    ax.set_ylabel("Observed frequency")
                    ax.set_title("Calibration Curve")
                    plots["Calibration Curve"] = fig_to_base64(fig, fmt, dpi)

                    fig, ax = plt.subplots(figsize=(6.5, 4.5))
                    ax.hist(positive_prob[y_test == labels.max()], bins=15, alpha=0.65, label="Positive", color=NATURE_COLORS["red"])
                    ax.hist(positive_prob[y_test == labels.min()], bins=15, alpha=0.65, label="Negative", color=NATURE_COLORS["blue"])
                    ax.set_xlabel("Predicted positive probability")
                    ax.set_ylabel("Count")
                    ax.set_title("Probability Separation")
                    ax.legend()
                    plots["Probability Separation"] = fig_to_base64(fig, fmt, dpi)

                    order = np.argsort(-positive_prob)
                    ranked_y = _series_values(y_test)[order]
                    cum_pos = np.cumsum(ranked_y == labels.max()) / max(1, np.sum(ranked_y == labels.max()))
                    sample_share = np.arange(1, len(ranked_y) + 1) / len(ranked_y)
                    fig, ax = plt.subplots(figsize=(6.5, 5))
                    ax.plot(sample_share, cum_pos, color=NATURE_COLORS["purple"], linewidth=2.2, label="Model")
                    ax.plot([0, 1], [0, 1], linestyle="--", color=NATURE_COLORS["slate"], label="Baseline")
                    ax.set_xlabel("Fraction of samples")
                    ax.set_ylabel("Captured positives")
                    ax.set_title("Cumulative Gain Curve")
                    ax.legend()
                    plots["Cumulative Gain Curve"] = fig_to_base64(fig, fmt, dpi)

                    pos_mask = _series_values(y_test) == labels.max()
                    neg_mask = _series_values(y_test) == labels.min()
                    pos_sorted = np.sort(positive_prob[pos_mask])
                    neg_sorted = np.sort(positive_prob[neg_mask])
                    grid = np.linspace(0, 1, 200)
                    tpr_curve = np.searchsorted(pos_sorted, grid, side="right") / max(1, len(pos_sorted))
                    fpr_curve = np.searchsorted(neg_sorted, grid, side="right") / max(1, len(neg_sorted))
                    ks = np.max(np.abs(tpr_curve - fpr_curve))
                    fig, ax = plt.subplots(figsize=(6.5, 5))
                    ax.plot(grid, tpr_curve, color=NATURE_COLORS["red"], linewidth=2, label="TPR CDF")
                    ax.plot(grid, fpr_curve, color=NATURE_COLORS["blue"], linewidth=2, label="FPR CDF")
                    ax.plot(grid, np.abs(tpr_curve - fpr_curve), color=NATURE_COLORS["teal"], linewidth=1.8, label=f"KS = {ks:.3f}")
                    ax.set_xlabel("Threshold")
                    ax.set_ylabel("Rate")
                    ax.set_title("KS Diagnostic Curve")
                    ax.legend()
                    plots["KS Diagnostic Curve"] = fig_to_base64(fig, fmt, dpi)
            except Exception:
                pass

        if train_pred is not None:
            train_metrics = _classification_metrics(y_train, train_pred)
            test_metrics = _classification_metrics(y_test, ypred)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.bar(["Train", "Test"], [train_metrics["Accuracy"], test_metrics["Accuracy"]], color=[NATURE_COLORS["blue"], NATURE_COLORS["orange"]], width=0.55)
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1.05)
            ax.set_title("Train vs Test Accuracy")
            plots["Train vs Test Accuracy"] = fig_to_base64(fig, fmt, dpi)

            metrics = list(train_metrics.keys())
            fig, ax = plt.subplots(figsize=(6.8, 5))
            xpos = np.arange(len(metrics))
            width = 0.36
            ax.bar(xpos - width / 2, [train_metrics[m] for m in metrics], width, label="Train", color=NATURE_COLORS["blue"])
            ax.bar(xpos + width / 2, [test_metrics[m] for m in metrics], width, label="Test", color=NATURE_COLORS["orange"])
            ax.set_xticks(xpos)
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1.05)
            ax.set_title("Metric Comparison")
            ax.legend()
            plots["Metric Comparison"] = fig_to_base64(fig, fmt, dpi)

            gap_values = [train_metrics[m] - test_metrics[m] for m in metrics]
            fig, ax = plt.subplots(figsize=(7, 4.8))
            bars = ax.bar(metrics, gap_values, color=[NATURE_COLORS["red"] if val > 0.03 else NATURE_COLORS["teal"] for val in gap_values], width=0.6)
            ax.axhline(0, color=NATURE_COLORS["slate"], linewidth=1.2)
            ax.set_ylabel("Train - Test")
            ax.set_title("Generalization Gap by Metric")
            for bar, value in zip(bars, gap_values):
                ax.text(bar.get_x() + bar.get_width() / 2, value + (0.01 if value >= 0 else -0.03), f"{value:.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
            plots["Generalization Gap by Metric"] = fig_to_base64(fig, fmt, dpi)

            summary_frame = pd.DataFrame(
                {
                    "Accuracy": [train_metrics["Accuracy"], test_metrics["Accuracy"], train_metrics["Accuracy"] - test_metrics["Accuracy"]],
                    "F1": [train_metrics["F1"], test_metrics["F1"], train_metrics["F1"] - test_metrics["F1"]],
                    "Precision": [train_metrics["Precision"], test_metrics["Precision"], train_metrics["Precision"] - test_metrics["Precision"]],
                    "Recall": [train_metrics["Recall"], test_metrics["Recall"], train_metrics["Recall"] - test_metrics["Recall"]],
                },
                index=["Train", "Test", "Gap"],
            )
            fig, ax = plt.subplots(figsize=(7.8, 3.8))
            sns.heatmap(summary_frame, annot=True, fmt=".3f", cmap="crest", ax=ax, cbar=False)
            ax.set_title("Generalization Scorecard")
            plots["Generalization Scorecard"] = fig_to_base64(fig, fmt, dpi)

            if hasattr(model, "predict_proba"):
                try:
                    train_prob = model.predict_proba(X_train)
                    test_prob = model.predict_proba(X_test)
                    train_conf = np.max(train_prob, axis=1)
                    test_conf = np.max(test_prob, axis=1)
                    fig, ax = plt.subplots(figsize=(7, 4.8))
                    sns.kdeplot(train_conf, ax=ax, label="Train confidence", color=NATURE_COLORS["blue"], fill=True, alpha=0.15)
                    sns.kdeplot(test_conf, ax=ax, label="Test confidence", color=NATURE_COLORS["orange"], fill=True, alpha=0.15)
                    ax.set_xlabel("Max class probability")
                    ax.set_title("Confidence Shift")
                    ax.legend()
                    plots["Confidence Shift"] = fig_to_base64(fig, fmt, dpi)
                except Exception:
                    pass
                try:
                    test_prob = model.predict_proba(X_test)
                    if np.asarray(test_prob).ndim == 2 and np.asarray(test_prob).shape[1] == 2:
                        threshold_plot = _plot_binary_threshold_diagnostics(y_test, np.asarray(test_prob)[:, 1], fmt, dpi)
                        if threshold_plot:
                            plots["Threshold Utility Diagnostics"] = threshold_plot
                except Exception:
                    pass

        importance_plot = _feature_importance_plot(model, list(X_test.columns), fmt, dpi) if isinstance(X_test, pd.DataFrame) else None
        if importance_plot:
            plots["Feature Importance"] = importance_plot
        learning_plot = _learning_curve_plot(model, X_train, y_train, ptype, fmt, dpi)
        if learning_plot:
            plots["Learning Curve"] = learning_plot
        kfold_plot = _plot_kfold_stability(model, X_train, y_train, ptype, fmt, dpi)
        if kfold_plot:
            plots["K-Fold Stability"] = kfold_plot
        y_rand_plot = _plot_y_randomization(model, X_train, y_train, X_test, y_test, ptype, fmt, dpi)
        if y_rand_plot:
            plots["Y-Randomization Test"] = y_rand_plot
        bootstrap_plot = _plot_bootstrap_stability(bootstrap_diag, ptype, fmt, dpi)
        if bootstrap_plot:
            plots["Bootstrap Stability"] = bootstrap_plot
        band_plot = _plot_bootstrap_uncertainty_band(bootstrap_diag, ptype, fmt, dpi)
        if band_plot:
            plots["Bootstrap Uncertainty Band"] = band_plot
    else:
        y_train_v = _series_values(y_train)
        y_test_v = _series_values(y_test)
        ypred_v = _series_values(ypred)
        train_pred_v = _series_values(train_pred) if train_pred is not None else None

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        min_val = min(y_test_v.min(), ypred_v.min(), y_train_v.min() if train_pred_v is not None else y_test_v.min(), train_pred_v.min() if train_pred_v is not None else ypred_v.min())
        max_val = max(y_test_v.max(), ypred_v.max(), y_train_v.max() if train_pred_v is not None else y_test_v.max(), train_pred_v.max() if train_pred_v is not None else ypred_v.max())
        if train_pred_v is not None:
            ax.scatter(y_train_v, train_pred_v, alpha=0.25, label=f"Train (R2={r2_score(y_train_v, train_pred_v):.3f})", color=NATURE_COLORS["blue"], edgecolors="white", linewidth=0.3)
        ax.scatter(y_test_v, ypred_v, alpha=0.75, label=f"Test (R2={r2_score(y_test_v, ypred_v):.3f})", color=NATURE_COLORS["red"], edgecolors="white", linewidth=0.3)
        ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color=NATURE_COLORS["slate"], linewidth=2, label="Identity")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.set_title("Predicted vs Observed")
        ax.legend()
        plots["Prediction vs True"] = fig_to_base64(fig, fmt, dpi)

        residuals = y_test_v - ypred_v
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.scatter(ypred_v, residuals, alpha=0.72, color=NATURE_COLORS["blue"], edgecolors="white", linewidth=0.3)
        ax.axhline(0, color=NATURE_COLORS["red"], linestyle="--", linewidth=1.8)
        ax.set_xlabel("Predicted values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals Plot")
        plots["Residuals Plot"] = fig_to_base64(fig, fmt, dpi)
        for name, plot in [
            ("Regression Calibration Belt", _plot_regression_calibration_belt(y_test_v, ypred_v, fmt, dpi)),
            ("Error Violin by Target Segment", _plot_error_violin_by_target_segment(y_test_v, ypred_v, fmt, dpi)),
            ("Standardized Residual Control Chart", _plot_standardized_residual_control_chart(residuals, fmt, dpi)),
            ("Residual Statistics Panel", _plot_residual_statistics_panel(residuals, fmt, dpi)),
            ("Observed-Predicted Joint Marginal Map", _plot_observed_predicted_marginal_joint(y_test_v, ypred_v, fmt, dpi)),
            ("Error Exceedance Curve", _plot_error_exceedance_curve(y_test_v, ypred_v, fmt, dpi)),
        ]:
            if plot:
                plots[name] = plot
        gallery = _plot_prediction_diagnostic_gallery(y_test_v, ypred_v, residuals, fmt, dpi)
        if gallery:
            plots["Prediction Diagnostic Gallery"] = gallery
        residual_hex = _plot_prediction_interval_residual_map(y_test_v, ypred_v, fmt, dpi)
        if residual_hex:
            plots["Residual Density Hexbin"] = residual_hex
        plots.update(_plot_additional_regression_diagnostics(model, X_test, y_test_v, ypred_v, fmt, dpi))
        plots.update(_plot_deep_regression_diagnostics(model, X_test, y_test_v, ypred_v, fmt, dpi))
        segment_dashboard = _plot_segment_performance_dashboard(y_test_v, ypred_v, fmt, dpi)
        if segment_dashboard:
            plots["Target-Segment Performance Dashboard"] = segment_dashboard

        residual_scan = _plot_residual_feature_scan(X_test, residuals, fmt, dpi)
        if residual_scan:
            plots["Residual-Feature Bias Scan"] = residual_scan
        hotspot_plot = _plot_error_hotspot_heatmap(X_test, y_test_v, ypred_v, fmt, dpi)
        if hotspot_plot:
            plots["Error Hotspot Map"] = hotspot_plot
        quantile_error = _plot_error_quantile_matrix(X_test, y_test_v, ypred_v, fmt, dpi)
        if quantile_error:
            plots["Feature Quantile Error Matrix"] = quantile_error
        worst_error = _plot_worst_error_table(y_test_v, ypred_v, fmt, dpi)
        if worst_error:
            plots["Worst Error Waterfall"] = worst_error
        residual_acf = _plot_residual_autocorrelation(residuals, fmt, dpi)
        if residual_acf:
            plots["Residual Run and Autocorrelation"] = residual_acf
        slice_bias = _plot_slice_bias_heatmap(X_test, residuals, fmt, dpi)
        if slice_bias:
            plots["Feature Slice Bias Heatmap"] = slice_bias

        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.scatter(y_test_v, residuals, alpha=0.72, color=NATURE_COLORS["teal"], edgecolors="white", linewidth=0.3)
        ax.axhline(0, color=NATURE_COLORS["red"], linestyle="--", linewidth=1.6)
        ax.set_xlabel("Observed values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Observed")
        plots["Residuals vs Observed"] = fig_to_base64(fig, fmt, dpi)

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        sns.histplot(residuals, kde=True, ax=ax, color=NATURE_COLORS["orange"])
        ax.set_xlabel("Residual")
        ax.set_title("Residual Distribution")
        plots["Residuals Density"] = fig_to_base64(fig, fmt, dpi)

        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        quantiles = np.linspace(0, 1, len(residuals), endpoint=False) + 0.5 / len(residuals)
        theo = np.quantile(np.random.normal(size=5000), quantiles)
        sample = np.quantile(np.sort(residuals), quantiles)
        ax.scatter(theo, sample, color=NATURE_COLORS["purple"], s=20, alpha=0.8)
        diag_min = min(theo.min(), sample.min())
        diag_max = max(theo.max(), sample.max())
        ax.plot([diag_min, diag_max], [diag_min, diag_max], linestyle="--", color=NATURE_COLORS["slate"])
        ax.set_xlabel("Theoretical normal quantiles")
        ax.set_ylabel("Observed residual quantiles")
        ax.set_title("Residual Q-Q Plot")
        plots["Residual Q-Q Plot"] = fig_to_base64(fig, fmt, dpi)

        mean_vals = (ypred_v + y_test_v) / 2
        diff_vals = ypred_v - y_test_v
        bias = np.mean(diff_vals)
        sd = np.std(diff_vals)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.scatter(mean_vals, diff_vals, color=NATURE_COLORS["teal"], alpha=0.72, edgecolors="white", linewidth=0.3)
        ax.axhline(bias, color=NATURE_COLORS["red"], linewidth=1.8, label=f"Bias = {bias:.3f}")
        ax.axhline(bias + 1.96 * sd, color=NATURE_COLORS["slate"], linestyle="--", linewidth=1.4)
        ax.axhline(bias - 1.96 * sd, color=NATURE_COLORS["slate"], linestyle="--", linewidth=1.4)
        ax.set_xlabel("Mean of observed and predicted")
        ax.set_ylabel("Prediction - observed")
        ax.set_title("Bland-Altman Plot")
        ax.legend()
        plots["Bland-Altman Plot"] = fig_to_base64(fig, fmt, dpi)

        if train_pred_v is not None:
            shift_plot = _plot_distribution_shift_predictions(y_train_v, train_pred_v, y_test_v, ypred_v, fmt, dpi)
            if shift_plot:
                plots["Observed-Predicted Distribution Shift"] = shift_plot
            conformal_plot = _plot_conformal_interval_diagnostics(y_train_v, train_pred_v, y_test_v, ypred_v, fmt, dpi)
            if conformal_plot:
                plots["Conformal Prediction Diagnostics"] = conformal_plot
            probabilistic_interval = _plot_probabilistic_prediction_interval(model, X_test, y_test_v, ypred_v, fmt, dpi)
            if probabilistic_interval:
                plots["Probabilistic Prediction Interval"] = probabilistic_interval
            train_r2 = r2_score(y_train_v, train_pred_v)
            test_r2 = r2_score(y_test_v, ypred_v)
            fig, ax = plt.subplots(figsize=(5.5, 5))
            ax.bar(["Train", "Test"], [train_r2, test_r2], color=[NATURE_COLORS["teal"], NATURE_COLORS["purple"]], width=0.55)
            ax.set_ylabel("R2")
            ax.set_title("Train vs Test R2")
            plots["Train vs Test R2"] = fig_to_base64(fig, fmt, dpi)

            train_residuals = y_train_v - train_pred_v
            train_rmse = float(np.sqrt(mean_squared_error(y_train_v, train_pred_v)))
            test_rmse = float(np.sqrt(mean_squared_error(y_test_v, ypred_v)))
            gap_r2 = train_r2 - test_r2

            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            summary_data = pd.DataFrame(
                {
                    "Train": [train_r2, train_rmse, float(np.mean(np.abs(train_residuals)))],
                    "Test": [test_r2, test_rmse, float(np.mean(np.abs(residuals)))],
                },
                index=["R2", "RMSE", "MAE"],
            )
            summary_data.T.plot(kind="bar", ax=ax, color=[NATURE_COLORS["teal"], NATURE_COLORS["orange"], NATURE_COLORS["purple"]], width=0.7)
            ax.set_title("Generalization Summary")
            ax.set_ylabel("Metric value")
            ax.legend(loc="upper right")
            plots["Generalization Summary"] = fig_to_base64(fig, fmt, dpi)

            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            sns.kdeplot(train_residuals, ax=ax, label="Train residuals", color=NATURE_COLORS["blue"], fill=True, alpha=0.15)
            sns.kdeplot(residuals, ax=ax, label="Test residuals", color=NATURE_COLORS["orange"], fill=True, alpha=0.15)
            ax.axvline(0, color=NATURE_COLORS["slate"], linestyle="--", linewidth=1.2)
            ax.set_xlabel("Residual")
            ax.set_title("Train-Test Residual Density")
            ax.legend()
            plots["Train-Test Residual Density"] = fig_to_base64(fig, fmt, dpi)

            train_abs = np.abs(train_residuals)
            test_abs = np.abs(residuals)
            x_train_ecdf, y_train_ecdf = _ecdf_points(train_abs)
            x_test_ecdf, y_test_ecdf = _ecdf_points(test_abs)
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            ax.step(x_train_ecdf, y_train_ecdf, where="post", color=NATURE_COLORS["blue"], linewidth=2, label="Train")
            ax.step(x_test_ecdf, y_test_ecdf, where="post", color=NATURE_COLORS["red"], linewidth=2, label="Test")
            ax.set_xlabel("Absolute error")
            ax.set_ylabel("ECDF")
            ax.set_title("Absolute Error ECDF")
            ax.legend()
            plots["Absolute Error ECDF"] = fig_to_base64(fig, fmt, dpi)

            fig, ax = plt.subplots(figsize=(7.2, 5))
            ax.scatter(train_pred_v, train_residuals, alpha=0.28, color=NATURE_COLORS["blue"], edgecolors="white", linewidth=0.25, label="Train")
            ax.scatter(ypred_v, residuals, alpha=0.68, color=NATURE_COLORS["orange"], edgecolors="white", linewidth=0.25, label="Test")
            ax.axhline(0, color=NATURE_COLORS["slate"], linestyle="--", linewidth=1.2)
            ax.set_xlabel("Predicted values")
            ax.set_ylabel("Residual")
            ax.set_title("Residual Split Comparison")
            ax.legend()
            plots["Residual Split Comparison"] = fig_to_base64(fig, fmt, dpi)

            fig, ax = plt.subplots(figsize=(6.8, 3.6))
            labels = ["Train R2", "Test R2", "Gap", "RMSE Ratio"]
            values = [train_r2, test_r2, gap_r2, test_rmse / (train_rmse + 1e-9)]
            sns.heatmap(
                pd.DataFrame([values], index=["Model"], columns=labels),
                annot=True,
                fmt=".3f",
                cmap="rocket_r",
                ax=ax,
                cbar=False,
            )
            ax.set_title("Overfitting Fingerprint")
            plots["Overfitting Fingerprint"] = fig_to_base64(fig, fmt, dpi)

        fig, ax = plt.subplots(figsize=(8, 4))
        samples = min(60, len(y_test_v))
        order = np.arange(samples)
        ax.plot(order, y_test_v[:samples], marker="o", label="Observed", color=NATURE_COLORS["slate"])
        ax.plot(order, ypred_v[:samples], marker="x", linestyle="--", label="Predicted", color=NATURE_COLORS["red"])
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Target")
        ax.set_title("Series Overlay")
        ax.legend()
        plots["Series Overlay"] = fig_to_base64(fig, fmt, dpi)

        quantiles = pd.qcut(pd.Series(y_test_v), q=min(10, len(y_test_v)), duplicates="drop")
        quantile_df = pd.DataFrame({"Observed": y_test_v, "Predicted": ypred_v, "Bin": quantiles})
        profile = quantile_df.groupby("Bin", observed=False).agg(
            observed_mean=("Observed", "mean"),
            pred_mean=("Predicted", "mean"),
            rmse=("Predicted", lambda x: 0.0),
        )
        profile["rmse"] = quantile_df.groupby("Bin", observed=False).apply(
            lambda frame: float(np.sqrt(np.mean((frame["Observed"] - frame["Predicted"]) ** 2)))
        ).to_numpy()
        fig, ax = plt.subplots(figsize=(8, 4.8))
        xpos = np.arange(len(profile))
        ax.plot(xpos, profile["observed_mean"], marker="o", color=NATURE_COLORS["slate"], label="Observed mean")
        ax.plot(xpos, profile["pred_mean"], marker="s", color=NATURE_COLORS["red"], label="Predicted mean")
        ax2 = ax.twinx()
        ax2.bar(xpos, profile["rmse"], alpha=0.22, color=NATURE_COLORS["gold"], width=0.58, label="RMSE")
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"Q{i+1}" for i in range(len(profile))])
        ax.set_xlabel("Observed quantile bin")
        ax.set_ylabel("Target level")
        ax2.set_ylabel("RMSE")
        ax.set_title("Quantile Error Profile")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plots["Quantile Error Profile"] = fig_to_base64(fig, fmt, dpi)

        importance_plot = _feature_importance_plot(model, list(X_test.columns), fmt, dpi) if isinstance(X_test, pd.DataFrame) else None
        if importance_plot:
            plots["Feature Importance"] = importance_plot
        learning_plot = _learning_curve_plot(model, X_train, y_train, ptype, fmt, dpi)
        if learning_plot:
            plots["Learning Curve"] = learning_plot
        kfold_plot = _plot_kfold_stability(model, X_train, y_train, ptype, fmt, dpi)
        if kfold_plot:
            plots["K-Fold Stability"] = kfold_plot
        y_rand_plot = _plot_y_randomization(model, X_train, y_train, X_test, y_test, ptype, fmt, dpi)
        if y_rand_plot:
            plots["Y-Randomization Test"] = y_rand_plot
        bootstrap_plot = _plot_bootstrap_stability(bootstrap_diag, ptype, fmt, dpi)
        if bootstrap_plot:
            plots["Bootstrap Stability"] = bootstrap_plot
        band_plot = _plot_bootstrap_uncertainty_band(bootstrap_diag, ptype, fmt, dpi)
        if band_plot:
            plots["Bootstrap Uncertainty Band"] = band_plot
        uncertainty_alignment = _plot_uncertainty_error_alignment(bootstrap_diag, fmt, dpi)
        if uncertainty_alignment:
            plots["Uncertainty Error Alignment"] = uncertainty_alignment
        risk_plot = _plot_prediction_risk_stratification(X_train, X_test, y_test_v, ypred_v, bootstrap_diag, fmt, dpi)
        if risk_plot:
            plots["Prediction Risk Stratification"] = risk_plot
        interaction_screen = _plot_feature_interaction_screen(model, X_train, fmt, dpi)
        if interaction_screen:
            plots["Permutation Interaction Screen"] = interaction_screen
        response_grid = _plot_feature_response_sensitivity_grid(model, X_train, fmt, dpi)
        if response_grid:
            plots["One-Factor Response Sensitivity Grid"] = response_grid
        williams_plot = _plot_williams(model, X_train, y_train, X_test, y_test, fmt, dpi)
        if williams_plot:
            plots["Williams Plot"] = williams_plot
        ad_payload = _applicability_domain_payload(X_train, X_test, residuals=residuals)
        ad_hist = _plot_ad_histogram(ad_payload, fmt, dpi)
        if ad_hist:
            plots["Applicability Domain Distance"] = ad_hist
        ad_rel = _plot_ad_reliability(ad_payload, fmt, dpi)
        if ad_rel:
            plots["Applicability Domain Reliability"] = ad_rel
        ad_cov = _plot_ad_coverage(ad_payload, fmt, dpi)
        if ad_cov:
            plots["Applicability Domain Coverage"] = ad_cov
        if train_pred_v is not None:
            interval_df = _interval_mae_frame(y_train_v, train_pred_v, y_test_v, ypred_v, interval_bins, interval_edges)
            plots["Interval Stability Analysis"] = _plot_interval_bar(interval_df, fmt, dpi)
            if len(interval_df) == 4:
                plots["Interval Stability Radial"] = _plot_interval_radial(interval_df, "Target Interval Stability", fmt, dpi)

    meta_plot = _plot_meta_history(meta_history, fmt, dpi)
    if meta_plot:
        plots["Hyperparameter Search History"] = meta_plot
    meta_heatmap = _plot_meta_heatmap(meta_history, fmt, dpi)
    if meta_heatmap:
        plots["Hyperparameter Search Heatmap"] = meta_heatmap
    meta_surface = _plot_meta_surface(meta_history, fmt, dpi)
    if meta_surface:
        plots["Hyperparameter Response Surface"] = meta_surface
    meta_rank = _plot_meta_top_trials(meta_history, fmt, dpi)
    if meta_rank:
        plots["Top Hyperparameter Trials"] = meta_rank
    meta_dist = _plot_meta_score_distribution(meta_history, fmt, dpi)
    if meta_dist:
        plots["Meta Score Distribution"] = meta_dist
    meta_improve = _plot_meta_improvement_curve(meta_history, fmt, dpi)
    if meta_improve:
        plots["Meta Improvement Curve"] = meta_improve
    meta_explore = _plot_meta_exploration_exploitation(meta_history, fmt, dpi)
    if meta_explore:
        plots["Meta Exploration Exploitation"] = meta_explore

    return plots


def generate_global_plots(
    trained_models_dict,
    X_train,
    y_train,
    X_test,
    y_test,
    ptype,
    fmt="png",
    dpi=300,
    min_score=None,
    filter_enabled=False,
    meta_histories=None,
):
    apply_nature_style()
    plots = {}
    if not trained_models_dict:
        return plots

    y_test_v = _series_values(y_test)

    metric_table = []
    for m_name, model_obj in trained_models_dict.items():
        if not hasattr(model_obj, "predict"):
            continue
        train_pred = _series_values(model_obj.predict(X_train))
        pred = _series_values(model_obj.predict(X_test))
        cv_mean, cv_std = _cv_score_summary(model_obj, X_train, y_train, ptype)
        if _ptype_is_regression(ptype):
            primary = r2_score(y_test_v, pred)
            train_primary = r2_score(_series_values(y_train), train_pred)
            metric_table.append(
                {
                    "Model": m_name,
                    "Primary": primary,
                    "TrainPrimary": train_primary,
                    "Gap": train_primary - primary,
                    "CVMean": cv_mean,
                    "CVStd": cv_std,
                    "Secondary": np.sqrt(mean_squared_error(y_test_v, pred)),
                    "Tertiary": mean_absolute_error(y_test_v, pred),
                }
            )
        else:
            stats = _classification_metrics(y_test, pred)
            train_stats = _classification_metrics(y_train, train_pred)
            metric_table.append(
                {
                    "Model": m_name,
                    "Primary": stats["Accuracy"],
                    "TrainPrimary": train_stats["Accuracy"],
                    "Gap": train_stats["Accuracy"] - stats["Accuracy"],
                    "CVMean": cv_mean,
                    "CVStd": cv_std,
                    "Secondary": stats["Precision"],
                    "Tertiary": stats["F1"],
                }
            )

    if not metric_table:
        return plots

    metric_df = pd.DataFrame(metric_table).sort_values("Primary", ascending=False)
    if _ptype_is_regression(ptype) and filter_enabled and min_score is not None:
        metric_df = metric_df[metric_df["Primary"] >= float(min_score)].copy()
    if metric_df.empty:
        return plots

    colors = _distinct_colors(len(metric_df))
    metric_df["Color"] = colors
    ref_std = np.std(y_test_v) + 1e-9

    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.45 * len(metric_df) + 2)))
    ax.barh(metric_df["Model"], metric_df["Primary"], color=list(metric_df["Color"]))
    ax.set_title("Model Leaderboard")
    ax.set_xlabel("R2" if _ptype_is_regression(ptype) else "Accuracy")
    ax.set_ylabel("")
    plots["Model Leaderboard"] = fig_to_base64(fig, fmt, dpi)

    if _ptype_is_regression(ptype):
        taylor_df = metric_df.head(min(8, len(metric_df))).copy()
        fig = plt.figure(figsize=(8.8, 7.4))
        ax = fig.add_subplot(111)
        radius_max = ref_std * 1.45

        x_vals = np.linspace(0, radius_max, 220)
        for corr in [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]:
            y_vals = np.tan(np.arccos(corr)) * (x_vals - 0)
            mask = (y_vals >= 0) & (np.sqrt(x_vals**2 + y_vals**2) <= radius_max)
            ax.plot(x_vals[mask], y_vals[mask], linestyle=":", linewidth=0.9, color="#cbd5e1", alpha=0.95)
            if np.any(mask):
                ax.text(x_vals[mask][-1], y_vals[mask][-1], f"r={corr:.2f}", fontsize=7.5, color="#64748b", ha="left", va="bottom")
        xx, yy = np.meshgrid(np.linspace(0, radius_max, 240), np.linspace(0, radius_max, 240))
        crmse = np.sqrt((xx - ref_std) ** 2 + yy**2)
        radial = np.sqrt(xx**2 + yy**2)
        crmse = np.where(radial <= radius_max, crmse, np.nan)
        contours = ax.contour(xx, yy, crmse, levels=6, colors="#94a3b8", linestyles="--", linewidths=0.9, alpha=0.8)
        ax.clabel(contours, inline=True, fontsize=7, fmt="d=%1.2f")

        theta = np.linspace(0, np.pi / 2, 240)
        for ratio in [0.5, 0.75, 1.0, 1.25]:
            radius = ref_std * ratio
            ax.plot(radius * np.cos(theta), radius * np.sin(theta), color="#e2e8f0", linewidth=0.9)

        ax.scatter([ref_std], [0], marker="*", s=220, color="black", zorder=6)
        ax.text(ref_std, 0.02 * radius_max, "Reference", ha="center", va="bottom", fontsize=9, fontweight="bold", color=NATURE_COLORS["slate"])

        offsets = _label_offsets(len(taylor_df))
        for idx, ((_, row), (dx, dy)) in enumerate(zip(taylor_df.iterrows(), offsets), start=1):
            model_obj = trained_models_dict[row["Model"]]
            pred = _series_values(model_obj.predict(X_test))
            std_p = np.std(pred)
            corr = max(0, _safe_corr(y_test_v, pred))
            dist = np.sqrt(ref_std**2 + std_p**2 - 2 * ref_std * std_p * corr)
            color = row["Color"]
            x = std_p * corr
            y = std_p * np.sqrt(max(0, 1 - corr**2))
            ax.scatter([x], [y], s=150, color=color, edgecolors="white", linewidth=1.0, zorder=7)
            ax.plot([ref_std, x], [0, y], color=color, linewidth=1.1, alpha=0.72)
            short_label = row["Model"]
            if len(short_label) > 18:
                short_label = short_label[:16] + "..."
            ax.annotate(
                f"{short_label}\nd={dist:.3f}",
                xy=(x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7.4,
                color=color,
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, linewidth=0.6, alpha=0.9),
                arrowprops=dict(arrowstyle="-", color=color, lw=0.7, alpha=0.8),
                zorder=8,
            )

        ax.set_xlim(0, radius_max)
        ax.set_ylim(0, radius_max * 0.92)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Standard deviation projected on x-axis")
        ax.set_ylabel("Orthogonal variability")
        ax.set_title(f"Multi-Model Taylor Diagram (Top {len(taylor_df)})", pad=18)
        plots["Multi-Model Taylor Diagram"] = fig_to_base64(fig, fmt, dpi)

        performance_rows = []
        for m_name in metric_df["Model"]:
            model_obj = trained_models_dict[m_name]
            pred = _series_values(model_obj.predict(X_test))
            summary = _regression_metrics(y_test_v, pred, ref_std)
            performance_rows.append(
                {
                    "Model": m_name,
                    "Test R2": metric_df.loc[metric_df["Model"] == m_name, "Primary"].iloc[0],
                    "Train R2": metric_df.loc[metric_df["Model"] == m_name, "TrainPrimary"].iloc[0],
                    "CV Mean": metric_df.loc[metric_df["Model"] == m_name, "CVMean"].iloc[0],
                    "Gap Penalty": max(0.0, 1 - metric_df.loc[metric_df["Model"] == m_name, "Gap"].iloc[0]),
                    "RMSE Skill": summary["RMSE Skill"],
                    "MAE Skill": summary["MAE Skill"],
                }
            )
    else:
        performance_rows = []
        for m_name in metric_df["Model"]:
            model_obj = trained_models_dict[m_name]
            pred = model_obj.predict(X_test)
            stats = _classification_metrics(y_test, pred)
            performance_rows.append(
                {
                    "Model": m_name,
                    "Test Accuracy": metric_df.loc[metric_df["Model"] == m_name, "Primary"].iloc[0],
                    "Train Accuracy": metric_df.loc[metric_df["Model"] == m_name, "TrainPrimary"].iloc[0],
                    "CV Mean": metric_df.loc[metric_df["Model"] == m_name, "CVMean"].iloc[0],
                    "Gap Penalty": max(0.0, 1 - metric_df.loc[metric_df["Model"] == m_name, "Gap"].iloc[0]),
                    "Precision": stats["Precision"],
                    "F1": stats["F1"],
                }
            )

        fig, ax = plt.subplots(figsize=(8, max(4.5, 0.45 * len(metric_df) + 2)))
        metric_long = metric_df[["Model", "Primary", "Secondary", "Tertiary"]].copy()
        metric_long.columns = ["Model", "Accuracy", "Precision", "F1"]
        melted = metric_long.melt(id_vars="Model", var_name="Metric", value_name="Score")
        sns.barplot(data=melted, x="Score", y="Model", hue="Metric", ax=ax)
        ax.set_xlim(0, 1.05)
        ax.set_title("Classification Metrics Summary")
        plots["Classification Metrics Summary"] = fig_to_base64(fig, fmt, dpi)

    performance_df = pd.DataFrame(performance_rows).set_index("Model")
    fig, ax = plt.subplots(figsize=(9.2, max(4.2, 0.42 * len(performance_df) + 2.2)))
    sns.heatmap(performance_df, annot=True, fmt=".3f", cmap="crest", linewidths=0.5, linecolor="white", ax=ax, cbar=True)
    ax.set_title("Multi-Model Performance Matrix")
    plots["Multi-Model Performance Matrix"] = fig_to_base64(fig, fmt, dpi)

    fig, ax = plt.subplots(figsize=(8.4, max(4.6, 0.38 * len(metric_df) + 2.5)))
    scatter = ax.scatter(
        metric_df["Gap"],
        metric_df["Primary"],
        s=120 + 600 * np.nan_to_num(metric_df["CVStd"], nan=0.02),
        c=np.nan_to_num(metric_df["CVMean"], nan=metric_df["Primary"]),
        cmap="viridis",
        edgecolors="white",
        linewidth=0.8,
        alpha=0.9,
    )
    for _, row in metric_df.iterrows():
        ax.text(row["Gap"] + 0.003, row["Primary"] + 0.003, row["Model"], fontsize=7.5, color=NATURE_COLORS["slate"])
    ax.axvline(0.03 if _ptype_is_classification(ptype) else 0.05, linestyle="--", color=NATURE_COLORS["red"], linewidth=1.2, label="Overfit warning line")
    ax.set_xlabel("Generalization gap (train - test)")
    ax.set_ylabel("Test accuracy" if _ptype_is_classification(ptype) else "Test R2")
    ax.set_title("Overfitting Risk Map")
    ax.legend(loc="lower left")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("CV mean")
    plots["Overfitting Risk Map"] = fig_to_base64(fig, fmt, dpi)

    fig, ax = plt.subplots(figsize=(9, max(4.4, 0.4 * len(metric_df) + 2.4)))
    ypos = np.arange(len(metric_df))
    ax.hlines(ypos, metric_df["Primary"], metric_df["TrainPrimary"], color="#cbd5e1", linewidth=3)
    ax.scatter(metric_df["Primary"], ypos, color=NATURE_COLORS["orange"], s=72, label="Test", zorder=3)
    ax.scatter(metric_df["TrainPrimary"], ypos, color=NATURE_COLORS["blue"], s=72, label="Train", zorder=3)
    for y_val, (_, row) in zip(ypos, metric_df.iterrows()):
        ax.text(max(row["Primary"], row["TrainPrimary"]) + 0.005, y_val, f"gap={row['Gap']:.3f}", va="center", fontsize=7.5, color=NATURE_COLORS["slate"])
    ax.set_yticks(ypos)
    ax.set_yticklabels(metric_df["Model"])
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy" if _ptype_is_classification(ptype) else "R2")
    ax.set_title("Train-Test Generalization Ladder")
    ax.legend(loc="lower right")
    plots["Train-Test Generalization Ladder"] = fig_to_base64(fig, fmt, dpi)

    try:
        rank_df = performance_df.copy()
        higher_better = [c for c in rank_df.columns if c not in {"Gap Penalty"}]
        ranks = pd.DataFrame(index=rank_df.index)
        for col in rank_df.columns:
            ranks[col] = rank_df[col].rank(ascending=False if col in higher_better else False, method="min")
        ranks["Consensus Rank"] = ranks.mean(axis=1)
        ranks = ranks.sort_values("Consensus Rank")
        fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.38 * len(ranks) + 2)))
        sns.heatmap(ranks, annot=True, fmt=".1f", cmap="YlGnBu_r", linewidths=0.4, ax=ax)
        ax.set_title("Consensus Model Rank Matrix")
        plots["Consensus Model Rank Matrix"] = fig_to_base64(fig, fmt, dpi)
    except Exception:
        pass

    try:
        best_cols = [c for c in performance_df.columns if c != "Gap Penalty"][:5]
        radar_df = performance_df[best_cols].copy()
        radar_df = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min() + 1e-9)
        radar_df["Gap Penalty"] = performance_df["Gap Penalty"].clip(0, 1)
        radar_df = radar_df.head(min(5, len(radar_df)))
        labels = radar_df.columns.tolist()
        theta = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        theta = np.r_[theta, theta[0]]
        fig, ax = plt.subplots(figsize=(7.2, 7.2), subplot_kw={"projection": "polar"})
        colors = _distinct_colors(len(radar_df))
        for color, (model_name, row) in zip(colors, radar_df.iterrows()):
            vals = np.r_[row.to_numpy(dtype=float), row.to_numpy(dtype=float)[0]]
            ax.plot(theta, vals, linewidth=2, color=color, label=model_name)
            ax.fill(theta, vals, color=color, alpha=0.08)
        ax.set_xticks(theta[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title("Model Capability Radar")
        ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=8)
        plots["Model Capability Radar"] = fig_to_base64(fig, fmt, dpi)
    except Exception:
        pass

    try:
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        ax.scatter(metric_df["Gap"], metric_df["Primary"], s=110, c=np.nan_to_num(metric_df["CVStd"], nan=0), cmap="magma_r", edgecolors="white", linewidth=0.8)
        ordered = metric_df.sort_values("Gap")
        frontier = []
        best = -np.inf
        for _, row in ordered.iterrows():
            if row["Primary"] >= best:
                frontier.append(row)
                best = row["Primary"]
        if frontier:
            fr = pd.DataFrame(frontier)
            ax.plot(fr["Gap"], fr["Primary"], color=NATURE_COLORS["teal"], linewidth=2.2, marker="o", label="Pareto frontier")
        for _, row in metric_df.iterrows():
            ax.text(row["Gap"] + 0.002, row["Primary"] + 0.002, row["Model"], fontsize=7.2)
        ax.set_xlabel("Generalization gap")
        ax.set_ylabel("Primary test score")
        ax.set_title("Model Selection Pareto Frontier")
        ax.legend()
        cbar = fig.colorbar(ax.collections[0], ax=ax)
        cbar.set_label("CV std")
        plots["Model Selection Pareto Frontier"] = fig_to_base64(fig, fmt, dpi)
    except Exception:
        pass

    try:
        pred_map = {}
        for m_name in metric_df["Model"]:
            pred_map[m_name] = _series_values(trained_models_dict[m_name].predict(X_test))
        pred_df = pd.DataFrame(pred_map)
        if _ptype_is_regression(ptype):
            agreement = pred_df.corr()
            title = "Model Prediction Correlation Matrix"
            cmap = "vlag"
            center = 0
            fmt_text = ".2f"
        else:
            names = pred_df.columns.tolist()
            agreement = pd.DataFrame(index=names, columns=names, dtype=float)
            for a in names:
                for b in names:
                    agreement.loc[a, b] = float(np.mean(pred_df[a].to_numpy() == pred_df[b].to_numpy()))
            title = "Model Prediction Agreement Matrix"
            cmap = "crest"
            center = None
            fmt_text = ".2f"
        fig, ax = plt.subplots(figsize=(8.2, 6.8))
        sns.heatmap(agreement, annot=True, fmt=fmt_text, cmap=cmap, center=center, linewidths=0.4, ax=ax)
        ax.set_title(title)
        plots[title] = fig_to_base64(fig, fmt, dpi)
    except Exception:
        pass

    try:
        pred_map = {m_name: _series_values(trained_models_dict[m_name].predict(X_test)) for m_name in metric_df["Model"]}
        pred_df = pd.DataFrame(pred_map)
        if _ptype_is_regression(ptype):
            ensemble_mean = pred_df.mean(axis=1).to_numpy(dtype=float)
            disagreement = pred_df.std(axis=1).to_numpy(dtype=float)
            abs_error = np.abs(y_test_v - ensemble_mean)
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
            axes[0].scatter(disagreement, abs_error, color=NATURE_COLORS["red"], alpha=0.72, s=42, edgecolors="white", linewidth=0.3)
            axes[0].set_xlabel("Across-model prediction std")
            axes[0].set_ylabel("Absolute ensemble error")
            axes[0].set_title("Ensemble Disagreement vs Error")
            try:
                coef = np.polyfit(disagreement, abs_error, 1)
                grid = np.linspace(disagreement.min(), disagreement.max(), 120)
                axes[0].plot(grid, np.poly1d(coef)(grid), color=NATURE_COLORS["slate"], linestyle="--")
            except Exception:
                pass
            bins = pd.qcut(pd.Series(disagreement), q=min(6, len(np.unique(disagreement))), duplicates="drop")
            prof = pd.DataFrame({"bin": bins, "error": abs_error}).groupby("bin", observed=False)["error"].mean()
            axes[1].bar(np.arange(len(prof)), prof.values, color=NATURE_COLORS["orange"])
            axes[1].set_xticks(np.arange(len(prof)))
            axes[1].set_xticklabels([f"D{i+1}" for i in range(len(prof))])
            axes[1].set_xlabel("Disagreement stratum")
            axes[1].set_ylabel("Mean absolute error")
            axes[1].set_title("Error by Model Disagreement Stratum")
            plots["Ensemble Disagreement Error Analysis"] = fig_to_base64(fig, fmt, dpi)
        else:
            arr = pred_df.astype(str)
            vote_share = arr.apply(lambda row: row.value_counts(normalize=True).max(), axis=1)
            consensus = arr.mode(axis=1).iloc[:, 0].astype(str)
            correct = consensus.to_numpy() == pd.Series(y_test).astype(str).to_numpy()
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
            sns.histplot(vote_share, bins=10, ax=axes[0], color=NATURE_COLORS["blue"])
            axes[0].set_xlabel("Top-vote share")
            axes[0].set_title("Ensemble Vote Confidence")
            prof = pd.DataFrame({"vote_share": vote_share, "correct": correct.astype(float)})
            prof["bin"] = pd.cut(prof["vote_share"], bins=np.linspace(0, 1, 7), include_lowest=True)
            acc = prof.groupby("bin", observed=False)["correct"].mean()
            axes[1].bar(np.arange(len(acc)), acc.values, color=NATURE_COLORS["teal"])
            axes[1].set_ylim(0, 1.05)
            axes[1].set_xticks(np.arange(len(acc)))
            axes[1].set_xticklabels([f"B{i+1}" for i in range(len(acc))])
            axes[1].set_ylabel("Consensus accuracy")
            axes[1].set_title("Accuracy by Vote Confidence")
            plots["Ensemble Vote Confidence Analysis"] = fig_to_base64(fig, fmt, dpi)
    except Exception:
        pass

    try:
        error_map = {}
        for m_name in metric_df["Model"].head(12):
            pred = _series_values(trained_models_dict[m_name].predict(X_test))
            if _ptype_is_regression(ptype):
                error_map[m_name] = np.abs(y_test_v - pred)
            else:
                error_map[m_name] = (np.asarray(pred).astype(str) != pd.Series(y_test).astype(str).to_numpy()).astype(float)
        err_df = pd.DataFrame(error_map)
        err_df = err_df.iloc[np.argsort(err_df.mean(axis=1).to_numpy())[::-1][: min(80, len(err_df))]]
        fig, ax = plt.subplots(figsize=(9.2, max(4.8, 0.09 * len(err_df) + 2.5)))
        sns.heatmap(err_df, cmap="rocket_r", yticklabels=False, linewidths=0.0, ax=ax)
        ax.set_title("Model Error Signature Map")
        ax.set_xlabel("Model")
        ax.set_ylabel("Hardest test samples")
        plots["Model Error Signature Map"] = fig_to_base64(fig, fmt, dpi)
    except Exception:
        pass

    baseline_name = metric_df.iloc[0]["Model"]
    significance_df = _build_significance_df(trained_models_dict, X_test, y_test, ptype, baseline_name)
    if not significance_df.empty:
        plots["Wilcoxon Significance Analysis"] = _build_wilcoxon_plot(significance_df, baseline_name, ptype, fmt, dpi)
        plots["Cohens D Effect Size"] = _build_effect_size_plot(significance_df, baseline_name, fmt, dpi)
    if meta_histories:
        plots.update(_plot_meta_algorithm_overview(meta_histories, fmt, dpi))

    return plots
