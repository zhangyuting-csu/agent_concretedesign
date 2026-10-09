import warnings
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from core.plotting_utils import NATURE_COLORS, apply_nature_style, fig_to_base64


def _numeric_cols(df, target_col=None):
    return [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]


def _feature_cols(df, target_col=None):
    return [c for c in df.columns if c != target_col]


def _sample_df(df, max_rows=4000):
    if len(df) <= max_rows:
        return df.copy()
    return df.sample(max_rows, random_state=42).copy()


def _safe_corr(x, y):
    try:
        return float(pd.Series(x).corr(pd.Series(y)))
    except Exception:
        return np.nan


def _fig(name, fig, fmt, dpi):
    return name, fig_to_base64(fig, fmt, dpi)


def build_quality_tables(df, target_col=None):
    rows = []
    total_rows = max(len(df), 1)
    for col in df.columns:
        s = df[col]
        numeric = pd.api.types.is_numeric_dtype(s)
        rows.append(
            {
                "Feature": col,
                "Role": "Target" if col == target_col else "Feature",
                "DType": str(s.dtype),
                "Missing Count": int(s.isna().sum()),
                "Missing Rate": float(s.isna().mean()),
                "Unique Count": int(s.nunique(dropna=True)),
                "Unique Rate": float(s.nunique(dropna=True) / total_rows),
                "Zero Count": int((s == 0).sum()) if numeric else "",
                "Mean": float(s.mean()) if numeric else "",
                "Std": float(s.std()) if numeric else "",
                "Min": float(s.min()) if numeric else "",
                "Q1": float(s.quantile(0.25)) if numeric else "",
                "Median": float(s.median()) if numeric else "",
                "Q3": float(s.quantile(0.75)) if numeric else "",
                "Max": float(s.max()) if numeric else "",
                "Skew": float(s.skew()) if numeric else "",
                "Kurtosis": float(s.kurtosis()) if numeric else "",
                "Likely ID": bool(s.nunique(dropna=True) > 0.92 * total_rows and col != target_col),
                "Constant": bool(s.nunique(dropna=True) <= 1),
            }
        )
    quality = pd.DataFrame(rows)
    duplicate = pd.DataFrame(
        [
            {
                "Metric": "Duplicate Rows",
                "Value": int(df.duplicated().sum()),
                "Rate": float(df.duplicated().mean()) if len(df) else 0,
            }
        ]
    )
    return quality, duplicate


def build_leakage_table(df, target_col=None, problem_type="Regression"):
    if not target_col or target_col not in df.columns:
        return pd.DataFrame()
    rows = []
    target = df[target_col]
    y_numeric = pd.api.types.is_numeric_dtype(target)
    for col in _feature_cols(df, target_col):
        s = df[col]
        exact_match = bool(s.equals(target))
        corr = np.nan
        if pd.api.types.is_numeric_dtype(s) and y_numeric:
            corr = abs(_safe_corr(s, target))
        overlap = np.nan
        try:
            overlap = float((s.astype(str) == target.astype(str)).mean())
        except Exception:
            pass
        risk = "Low"
        reasons = []
        if exact_match or (not np.isnan(overlap) and overlap > 0.98):
            risk = "Critical"
            reasons.append("feature nearly equals target")
        if not np.isnan(corr) and corr > 0.98:
            risk = "High"
            reasons.append("absolute correlation with target > 0.98")
        if s.nunique(dropna=True) > 0.92 * max(len(df), 1):
            reasons.append("ID-like high-cardinality column")
            if risk == "Low":
                risk = "Medium"
        rows.append(
            {
                "Feature": col,
                "Leakage Risk": risk,
                "Abs Corr With Target": corr,
                "String Equality Rate": overlap,
                "Unique Count": int(s.nunique(dropna=True)),
                "Reason": "; ".join(reasons) if reasons else "No obvious leakage signal",
            }
        )
    return pd.DataFrame(rows).sort_values(["Leakage Risk", "Abs Corr With Target"], ascending=[True, False])


def build_vif_table(df, target_col=None):
    nums = _numeric_cols(df, target_col)
    if len(nums) < 2:
        return pd.DataFrame()
    work = df[nums].replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < 5:
        work = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(df[nums]), columns=nums)
    rows = []
    for col in nums:
        others = [c for c in nums if c != col]
        try:
            model = LinearRegression().fit(work[others], work[col])
            r2 = float(model.score(work[others], work[col]))
            vif = float(1 / max(1 - r2, 1e-8))
        except Exception:
            r2, vif = np.nan, np.nan
        rows.append({"Feature": col, "R2 Explained By Other Features": r2, "VIF": vif})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def build_target_signal_table(df, target_col=None, problem_type="Regression"):
    if not target_col or target_col not in df.columns:
        return pd.DataFrame()
    features = _feature_cols(df, target_col)
    if not features:
        return pd.DataFrame()
    work = df[features + [target_col]].copy()
    X = pd.get_dummies(work[features], dummy_na=True)
    y = work[target_col]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)
    try:
        if problem_type == "Classification":
            y_codes = pd.factorize(y)[0]
            scores = mutual_info_classif(X, y_codes, random_state=42)
        else:
            y_num = pd.to_numeric(y, errors="coerce")
            mask = y_num.notna()
            scores = mutual_info_regression(X.loc[mask], y_num.loc[mask], random_state=42)
            X = X.loc[mask]
        mapped = []
        for feature in features:
            prefix = f"{feature}_"
            score = float(np.sum([scores[i] for i, c in enumerate(X.columns) if c == feature or c.startswith(prefix)]))
            mapped.append({"Feature": feature, "Mutual Information Signal": score})
        return pd.DataFrame(mapped).sort_values("Mutual Information Signal", ascending=False)
    except Exception:
        return pd.DataFrame()


def _plot_missing_summary(quality, fmt, dpi):
    plot_df = quality.sort_values("Missing Rate", ascending=False).head(40)
    fig, ax = plt.subplots(figsize=(8.5, max(4, 0.28 * len(plot_df) + 1.8)))
    sns.barplot(data=plot_df, x="Missing Rate", y="Feature", ax=ax, color=NATURE_COLORS["orange"])
    ax.set_title("Missingness Summary")
    ax.set_xlabel("Missing rate")
    ax.set_ylabel("")
    return _fig("Missingness Summary", fig, fmt, dpi)


def _plot_quality_flags(quality, fmt, dpi):
    flags = quality.assign(
        MissingFlag=quality["Missing Rate"] > 0,
        ConstantFlag=quality["Constant"].astype(bool),
        IDFlag=quality["Likely ID"].astype(bool),
    )[["Feature", "MissingFlag", "ConstantFlag", "IDFlag"]].set_index("Feature")
    fig, ax = plt.subplots(figsize=(7.5, max(4, 0.25 * len(flags) + 1.5)))
    sns.heatmap(flags.astype(int), cmap=["#f8fafc", "#c44e52"], cbar=False, linewidths=0.4, ax=ax)
    ax.set_title("Data Quality Flag Map")
    ax.set_xlabel("Flag")
    ax.set_ylabel("")
    return _fig("Data Quality Flag Map", fig, fmt, dpi)


def _plot_leakage(leakage, fmt, dpi):
    if leakage.empty:
        return None
    order = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
    plot_df = leakage.assign(RiskScore=leakage["Leakage Risk"].map(order).fillna(0)).sort_values("RiskScore", ascending=False).head(40)
    fig, ax = plt.subplots(figsize=(8.5, max(4, 0.3 * len(plot_df) + 1.8)))
    sns.barplot(data=plot_df, x="RiskScore", y="Feature", hue="Leakage Risk", dodge=False, palette="rocket", ax=ax)
    ax.set_title("Potential Data Leakage Signals")
    ax.set_xlabel("Risk level score")
    ax.set_ylabel("")
    ax.legend(title="")
    return _fig("Leakage Risk Ranking", fig, fmt, dpi)


def _plot_vif(vif, fmt, dpi):
    if vif.empty:
        return None
    plot_df = vif.head(40).copy()
    fig, ax = plt.subplots(figsize=(8.5, max(4, 0.3 * len(plot_df) + 1.8)))
    sns.barplot(data=plot_df, x="VIF", y="Feature", ax=ax, color=NATURE_COLORS["purple"])
    ax.axvline(5, color=NATURE_COLORS["orange"], linestyle="--", linewidth=1.5, label="VIF = 5")
    ax.axvline(10, color=NATURE_COLORS["red"], linestyle="--", linewidth=1.5, label="VIF = 10")
    ax.set_xscale("log")
    ax.set_title("Multicollinearity Diagnostic (VIF)")
    ax.set_ylabel("")
    ax.legend()
    return _fig("VIF Multicollinearity Bar Chart", fig, fmt, dpi)


def _plot_corr_network(df, target_col, fmt, dpi):
    nums = _numeric_cols(df, target_col)
    if len(nums) < 2:
        return None
    corr = df[nums].corr().abs()
    edges = []
    for i, a in enumerate(nums):
        for b in nums[i + 1 :]:
            val = corr.loc[a, b]
            if val >= 0.7:
                edges.append((a, b, float(val)))
    if not edges:
        edges = sorted([(a, b, float(corr.loc[a, b])) for i, a in enumerate(nums) for b in nums[i + 1 :]], key=lambda x: x[2], reverse=True)[:12]
    try:
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(nums)
        graph.add_weighted_edges_from(edges)
        pos = nx.spring_layout(graph, seed=42, weight="weight")
        fig, ax = plt.subplots(figsize=(8, 6))
        weights = [graph[u][v]["weight"] for u, v in graph.edges()]
        nx.draw_networkx_nodes(graph, pos, node_color="#dceefb", edgecolors=NATURE_COLORS["blue"], node_size=900, ax=ax)
        nx.draw_networkx_edges(graph, pos, width=[1 + 4 * w for w in weights], alpha=0.55, edge_color=NATURE_COLORS["slate"], ax=ax)
        nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)
        ax.set_title("Feature Correlation Network")
        ax.axis("off")
        return _fig("Feature Correlation Network", fig, fmt, dpi)
    except Exception:
        return None


def _plot_pca(df, target_col, fmt, dpi):
    nums = _numeric_cols(df, target_col)
    if len(nums) < 2:
        return None
    work = _sample_df(df[nums + ([target_col] if target_col in df.columns else [])]).copy()
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(work[nums]), columns=nums)
    pcs = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(X))
    plot_df = pd.DataFrame({"PC1": pcs[:, 0], "PC2": pcs[:, 1]})
    if target_col in work.columns:
        plot_df[target_col] = work[target_col].to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    hue = target_col if target_col in plot_df.columns and plot_df[target_col].nunique() <= 12 else None
    if hue:
        sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue=hue, ax=ax, s=42, alpha=0.75, edgecolor="white")
    else:
        sc = ax.scatter(plot_df["PC1"], plot_df["PC2"], c=pd.to_numeric(plot_df.get(target_col), errors="coerce") if target_col in plot_df else NATURE_COLORS["blue"], cmap="viridis", s=42, alpha=0.75, edgecolor="white")
        if target_col in plot_df:
            plt.colorbar(sc, ax=ax, label=target_col)
    ax.set_title("PCA Projection of Numeric Features")
    return _fig("PCA Projection Scatter Plot", fig, fmt, dpi)


def _plot_target_lowess(df, target_col, fmt, dpi):
    if target_col not in df.columns or not pd.api.types.is_numeric_dtype(df[target_col]):
        return None
    nums = _numeric_cols(df, target_col)[:8]
    if not nums:
        return None
    rows = int(np.ceil(len(nums) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(10, max(4, 3.1 * rows)), squeeze=False)
    sample = _sample_df(df[nums + [target_col]].dropna())
    for ax, col in zip(axes.ravel(), nums):
        sns.regplot(data=sample, x=col, y=target_col, order=2, scatter_kws={"alpha": 0.35, "s": 20}, line_kws={"color": NATURE_COLORS["red"]}, ax=ax)
        ax.set_title(f"{col} vs {target_col}")
    for ax in axes.ravel()[len(nums) :]:
        ax.axis("off")
    fig.suptitle("Feature-Target Nonlinear Response Curves", y=1.01)
    return _fig("Feature Target Nonlinear Response Curves", fig, fmt, dpi)


def _plot_categorical_target(df, target_col, fmt, dpi):
    if target_col not in df.columns or not pd.api.types.is_numeric_dtype(df[target_col]):
        return None
    cats = [c for c in _feature_cols(df, target_col) if not pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique(dropna=True) <= 12][:8]
    if not cats:
        return None
    rows = int(np.ceil(len(cats) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(10, max(4, 3.1 * rows)), squeeze=False)
    for ax, col in zip(axes.ravel(), cats):
        work = df[[col, target_col]].dropna().copy()
        order = work.groupby(col)[target_col].mean().sort_values(ascending=False).index[:12]
        sns.pointplot(data=work, x=col, y=target_col, order=order, errorbar=("ci", 95), color=NATURE_COLORS["teal"], ax=ax)
        ax.tick_params(axis="x", rotation=30)
        ax.set_title(f"{col}: target mean with 95% CI")
    for ax in axes.ravel()[len(cats) :]:
        ax.axis("off")
    fig.suptitle("Categorical Feature Target Summary", y=1.01)
    return _fig("Categorical Target Mean CI Plot", fig, fmt, dpi)


def _plot_train_test_drift(df, target_col, fmt, dpi):
    nums = _numeric_cols(df, target_col)[:12]
    if len(nums) < 1 or len(df) < 20:
        return None, pd.DataFrame()
    train, test = train_test_split(df[nums], test_size=0.25, random_state=42)
    rows = []
    for col in nums:
        tr = train[col].dropna()
        te = test[col].dropna()
        rows.append(
            {
                "Feature": col,
                "Train Mean": float(tr.mean()) if len(tr) else np.nan,
                "Test Mean": float(te.mean()) if len(te) else np.nan,
                "Mean Difference": float(te.mean() - tr.mean()) if len(tr) and len(te) else np.nan,
                "Standardized Mean Difference": float((te.mean() - tr.mean()) / (tr.std() + 1e-9)) if len(tr) and len(te) else np.nan,
            }
        )
    drift = pd.DataFrame(rows).sort_values("Standardized Mean Difference", key=lambda s: s.abs(), ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, max(4, 0.32 * len(drift) + 1.8)))
    sns.barplot(data=drift, x="Standardized Mean Difference", y="Feature", ax=ax, color=NATURE_COLORS["red"])
    ax.axvline(0, color=NATURE_COLORS["slate"], linewidth=1)
    ax.axvline(0.2, color=NATURE_COLORS["orange"], linestyle="--", linewidth=1.3)
    ax.axvline(-0.2, color=NATURE_COLORS["orange"], linestyle="--", linewidth=1.3)
    ax.set_title("Train-Test Feature Drift Diagnostic")
    ax.set_ylabel("")
    return _fig("Train Test Drift Bar Chart", fig, fmt, dpi), drift


def _plot_signal(signal, fmt, dpi):
    if signal.empty:
        return None
    plot_df = signal.head(40)
    fig, ax = plt.subplots(figsize=(8.5, max(4, 0.3 * len(plot_df) + 1.8)))
    sns.barplot(data=plot_df, x="Mutual Information Signal", y="Feature", ax=ax, color=NATURE_COLORS["blue"])
    ax.set_title("Feature-Target Mutual Information Signal")
    ax.set_ylabel("")
    return _fig("Mutual Information Feature Signal", fig, fmt, dpi)


def _plot_clustered_correlation(df, target_col, fmt, dpi):
    nums = _numeric_cols(df, target_col)
    if len(nums) < 3:
        return None
    corr = df[nums].corr()
    try:
        grid = sns.clustermap(
            corr,
            cmap=sns.diverging_palette(230, 20, as_cmap=True),
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.35,
            figsize=(8, 8),
            cbar_kws={"label": "Pearson r"},
        )
        grid.fig.suptitle("Clustered Feature Correlation Map", y=1.02)
        return _fig("Clustered Correlation Heatmap", grid.fig, fmt, dpi)
    except Exception:
        return None


def _plot_pca_variance(df, target_col, fmt, dpi):
    nums = _numeric_cols(df, target_col)
    if len(nums) < 2:
        return None, pd.DataFrame()
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(df[nums]), columns=nums)
    n_comp = min(len(nums), 12)
    pca = PCA(n_components=n_comp, random_state=42).fit(StandardScaler().fit_transform(X))
    table = pd.DataFrame(
        {
            "PC": [f"PC{i+1}" for i in range(n_comp)],
            "Explained Variance Ratio": pca.explained_variance_ratio_,
            "Cumulative Explained Variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(table["PC"], table["Explained Variance Ratio"], color=NATURE_COLORS["blue"], alpha=0.78, label="Individual")
    ax.plot(table["PC"], table["Cumulative Explained Variance"], marker="o", color=NATURE_COLORS["red"], label="Cumulative")
    ax.axhline(0.9, color=NATURE_COLORS["slate"], linestyle="--", linewidth=1.3, label="90%")
    ax.set_ylim(0, 1.05)
    ax.set_title("PCA Explained Variance Spectrum")
    ax.set_ylabel("Explained variance ratio")
    ax.legend()
    return _fig("PCA Explained Variance Spectrum", fig, fmt, dpi), table


def _plot_missing_cooccurrence(df, fmt, dpi):
    miss_cols = [c for c in df.columns if df[c].isna().any()]
    if len(miss_cols) < 2:
        return None, pd.DataFrame()
    miss = df[miss_cols].isna().astype(int)
    co = miss.T.dot(miss)
    fig, ax = plt.subplots(figsize=(8, 6.8))
    sns.heatmap(co, annot=True, fmt="d", cmap="Reds", linewidths=0.4, ax=ax)
    ax.set_title("Missing-Value Co-occurrence Matrix")
    return _fig("Missing Value Cooccurrence Heatmap", fig, fmt, dpi), co


def _plot_outlier_burden(df, target_col, fmt, dpi):
    nums = _numeric_cols(df, target_col)
    if not nums:
        return None, pd.DataFrame()
    work = df[nums].replace([np.inf, -np.inf], np.nan)
    rows = []
    flags = pd.DataFrame(index=work.index)
    for col in nums:
        q1, q3 = work[col].quantile(0.25), work[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            flag = pd.Series(False, index=work.index)
        else:
            flag = (work[col] < q1 - 1.5 * iqr) | (work[col] > q3 + 1.5 * iqr)
        flags[col] = flag.fillna(False)
        rows.append({"Feature": col, "IQR Outlier Count": int(flags[col].sum()), "IQR Outlier Rate": float(flags[col].mean())})
    table = pd.DataFrame(rows).sort_values("IQR Outlier Rate", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, max(4.6, 0.26 * len(table) + 1.8)), gridspec_kw={"width_ratios": [1.1, 1]})
    sns.barplot(data=table.head(40), x="IQR Outlier Rate", y="Feature", ax=axes[0], color=NATURE_COLORS["orange"])
    axes[0].set_title("Feature-Level Outlier Burden")
    axes[0].set_ylabel("")
    burden = flags.sum(axis=1).value_counts().sort_index()
    axes[1].bar(burden.index.astype(str), burden.values, color=NATURE_COLORS["teal"])
    axes[1].set_title("Sample-Level Outlier Flag Count")
    axes[1].set_xlabel("Number of flagged features")
    axes[1].set_ylabel("Samples")
    return _fig("Outlier Burden Diagnostic", fig, fmt, dpi), table


def _plot_target_correlation_lollipop(df, target_col, fmt, dpi):
    nums = _numeric_cols(df, target_col)
    if target_col not in df.columns or not pd.api.types.is_numeric_dtype(df[target_col]) or not nums:
        return None, pd.DataFrame()
    rows = []
    for col in nums:
        rows.append({"Feature": col, "Pearson r": _safe_corr(df[col], df[target_col]), "Abs Pearson r": abs(_safe_corr(df[col], df[target_col]))})
    table = pd.DataFrame(rows).dropna().sort_values("Abs Pearson r", ascending=False)
    if table.empty:
        return None, table
    plot_df = table.head(40).sort_values("Pearson r")
    fig, ax = plt.subplots(figsize=(8.2, max(4, 0.3 * len(plot_df) + 1.8)))
    colors = [NATURE_COLORS["red"] if v < 0 else NATURE_COLORS["blue"] for v in plot_df["Pearson r"]]
    ax.hlines(plot_df["Feature"], 0, plot_df["Pearson r"], color=colors, linewidth=2)
    ax.scatter(plot_df["Pearson r"], plot_df["Feature"], color=colors, s=48, zorder=3)
    ax.axvline(0, color=NATURE_COLORS["slate"], linewidth=1)
    ax.set_title("Feature-Target Correlation Lollipop")
    ax.set_xlabel("Pearson correlation with target")
    return _fig("Feature Target Correlation Lollipop", fig, fmt, dpi), table


def _plot_model_free_importance(df, target_col, problem_type, fmt, dpi):
    if target_col not in df.columns:
        return None, pd.DataFrame()
    features = _feature_cols(df, target_col)
    work = df[features + [target_col]].dropna()
    if len(work) < 20 or not features:
        return None, pd.DataFrame()
    work = _sample_df(work, 2000)
    X = pd.get_dummies(work[features], dummy_na=True)
    y = work[target_col]
    if problem_type == "Classification":
        y_fit = pd.factorize(y)[0]
        model = RandomForestClassifier(n_estimators=180, random_state=42, n_jobs=1, class_weight="balanced")
        scoring = "accuracy"
    else:
        y_fit = pd.to_numeric(y, errors="coerce")
        mask = pd.Series(y_fit).notna()
        X, y_fit = X.loc[mask], y_fit.loc[mask]
        model = RandomForestRegressor(n_estimators=180, random_state=42, n_jobs=1)
        scoring = "r2"
    if len(X) < 20:
        return None, pd.DataFrame()
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y_fit, test_size=0.25, random_state=42)
    model.fit(X_train, y_train)
    result = permutation_importance(model, X_test, y_test, n_repeats=8, random_state=42, scoring=scoring, n_jobs=1)
    rows = []
    for feature in features:
        cols = [i for i, c in enumerate(X.columns) if c == feature or c.startswith(f"{feature}_")]
        if not cols:
            continue
        rows.append(
            {
                "Feature": feature,
                "Permutation Importance Mean": float(np.sum(result.importances_mean[cols])),
                "Permutation Importance Std": float(np.sqrt(np.sum(result.importances_std[cols] ** 2))),
            }
        )
    table = pd.DataFrame(rows).sort_values("Permutation Importance Mean", ascending=False)
    if table.empty:
        return None, table
    plot_df = table.head(30)
    fig, ax = plt.subplots(figsize=(8.5, max(4.2, 0.32 * len(plot_df) + 1.8)))
    ax.barh(plot_df["Feature"][::-1], plot_df["Permutation Importance Mean"][::-1], xerr=plot_df["Permutation Importance Std"][::-1], color=NATURE_COLORS["mint"], edgecolor="white")
    ax.set_title("Model-Free Permutation Signal Benchmark")
    ax.set_xlabel(f"Permutation score drop ({scoring})")
    return _fig("Model Free Permutation Signal Benchmark", fig, fmt, dpi), table


def _plot_tsne_projection(df, target_col, fmt, dpi):
    nums = _numeric_cols(df, target_col)
    if len(nums) < 3 or len(df) < 15:
        return None, pd.DataFrame()
    work = _sample_df(df[nums + ([target_col] if target_col in df.columns else [])].dropna(), 1200)
    if len(work) < 15:
        return None, pd.DataFrame()
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(work[nums]), columns=nums)
    perplexity = max(5, min(30, len(X) // 4))
    coords = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", learning_rate="auto").fit_transform(StandardScaler().fit_transform(X))
    table = pd.DataFrame({"TSNE1": coords[:, 0], "TSNE2": coords[:, 1]})
    if target_col in work.columns:
        table[target_col] = work[target_col].to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    hue = target_col if target_col in table.columns and table[target_col].nunique() <= 12 else None
    if hue:
        sns.scatterplot(data=table, x="TSNE1", y="TSNE2", hue=hue, ax=ax, s=38, alpha=0.72, edgecolor="white")
    else:
        color = pd.to_numeric(table.get(target_col), errors="coerce") if target_col in table.columns else NATURE_COLORS["blue"]
        sc = ax.scatter(table["TSNE1"], table["TSNE2"], c=color, cmap="viridis", s=38, alpha=0.72, edgecolor="white")
        if target_col in table.columns:
            plt.colorbar(sc, ax=ax, label=target_col)
    ax.set_title("t-SNE Manifold Projection")
    return _fig("t-SNE Manifold Projection", fig, fmt, dpi), table


def run_data_audit(df, target_col=None, problem_type="Regression", fmt="png", dpi=300):
    apply_nature_style()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quality, duplicate = build_quality_tables(df, target_col)
        leakage = build_leakage_table(df, target_col, problem_type)
        vif = build_vif_table(df, target_col)
        signal = build_target_signal_table(df, target_col, problem_type)
        plots = {}
        plot_data = {}
        plot_specs = [
            (_plot_missing_summary(quality, fmt, dpi), quality),
            (_plot_quality_flags(quality, fmt, dpi), quality),
            (_plot_leakage(leakage, fmt, dpi), leakage),
            (_plot_vif(vif, fmt, dpi), vif),
            (_plot_corr_network(df, target_col, fmt, dpi), df[_numeric_cols(df, target_col)] if _numeric_cols(df, target_col) else pd.DataFrame()),
            (_plot_clustered_correlation(df, target_col, fmt, dpi), df[_numeric_cols(df, target_col)] if _numeric_cols(df, target_col) else pd.DataFrame()),
            (_plot_pca(df, target_col, fmt, dpi), df[_numeric_cols(df, target_col) + ([target_col] if target_col in df.columns else [])] if _numeric_cols(df, target_col) else pd.DataFrame()),
            (_plot_target_lowess(df, target_col, fmt, dpi), df.copy()),
            (_plot_categorical_target(df, target_col, fmt, dpi), df.copy()),
            (_plot_signal(signal, fmt, dpi), signal),
        ]
        drift_plot, drift = _plot_train_test_drift(df, target_col, fmt, dpi)
        plot_specs.append((drift_plot, drift))
        for plot_item, table_item in [
            _plot_pca_variance(df, target_col, fmt, dpi),
            _plot_missing_cooccurrence(df, fmt, dpi),
            _plot_outlier_burden(df, target_col, fmt, dpi),
            _plot_target_correlation_lollipop(df, target_col, fmt, dpi),
            _plot_model_free_importance(df, target_col, problem_type, fmt, dpi),
            _plot_tsne_projection(df, target_col, fmt, dpi),
        ]:
            plot_specs.append((plot_item, table_item))
        for item, data in plot_specs:
            if item is None:
                continue
            name, img = item
            plots[name] = img
            plot_data[name] = data
        recommendations = []
        if int(duplicate.iloc[0]["Value"]) > 0:
            recommendations.append("存在重复行，建议在建模前确认是否为真实重复实验或数据录入重复。")
        high_missing = quality[quality["Missing Rate"] > 0.25]["Feature"].tolist()
        if high_missing:
            recommendations.append(f"高缺失字段需要处理：{', '.join(map(str, high_missing[:8]))}")
        high_vif = vif[vif["VIF"] > 10]["Feature"].tolist() if not vif.empty else []
        if high_vif:
            recommendations.append(f"存在强共线性特征，建议筛选或降维：{', '.join(map(str, high_vif[:8]))}")
        high_leak = leakage[leakage["Leakage Risk"].isin(["Critical", "High"])]["Feature"].tolist() if not leakage.empty else []
        if high_leak:
            recommendations.append(f"疑似泄漏字段需人工确认：{', '.join(map(str, high_leak[:8]))}")
        if not recommendations:
            recommendations.append("未发现严重质量风险；仍建议结合领域知识复核特征含义。")
        return {
            "tables": {
                "quality": quality.to_dict(orient="records"),
                "duplicates": duplicate.to_dict(orient="records"),
                "leakage": leakage.to_dict(orient="records"),
                "vif": vif.to_dict(orient="records"),
                "target_signal": signal.to_dict(orient="records"),
                "drift": drift.to_dict(orient="records") if isinstance(drift, pd.DataFrame) else [],
            },
            "plots": plots,
            "plot_data": plot_data,
            "recommendations": recommendations,
            "plot_help": {
                "Missingness Summary": "查看每个字段缺失比例，优先处理高缺失字段。",
                "Data Quality Flag Map": "同时标记缺失、常量列、疑似 ID 列，适合快速定位不可直接建模字段。",
                "Leakage Risk Ranking": "查找可能直接泄漏目标值或过度接近目标值的字段。",
                "VIF Multicollinearity Bar Chart": "评估数值特征之间的多重共线性，VIF 大于 10 通常需要关注。",
                "Feature Correlation Network": "展示强相关特征网络，用于发现冗余特征簇。",
                "PCA Projection Scatter Plot": "把数值特征投影到二维空间，观察类别/目标分布结构和离群样本。",
                "Feature Target Nonlinear Response Curves": "观察数值特征与数值目标之间的非线性响应关系。",
                "Categorical Target Mean CI Plot": "观察类别特征不同水平下目标均值和置信区间。",
                "Mutual Information Feature Signal": "评估每个特征对目标的非线性统计信号强度。",
                "Train Test Drift Bar Chart": "模拟训练/测试划分后检查特征分布漂移风险。",
                "Clustered Correlation Heatmap": "按相关结构自动聚类特征，用于识别冗余特征簇和变量组。",
                "PCA Explained Variance Spectrum": "展示主成分累计解释率，判断是否适合用 PCA 降维。",
                "Missing Value Cooccurrence Heatmap": "查看缺失是否成组出现，帮助判断缺失机制。",
                "Outlier Burden Diagnostic": "同时展示特征层面和样本层面的 IQR 异常负担。",
                "Feature Target Correlation Lollipop": "用棒棒糖图展示数值特征与目标的正负相关方向和强度。",
                "Model Free Permutation Signal Benchmark": "用轻量随机森林做模型无关置换信号基准，帮助识别稳定预测因子。",
                "t-SNE Manifold Projection": "非线性流形投影，用于观察样本团簇、潜在分型和离群结构。",
            },
        }
