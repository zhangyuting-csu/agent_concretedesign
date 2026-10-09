import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from core.plotting_utils import NATURE_COLORS, apply_nature_style, fig_to_base64


def _numeric_df(df):
    return df.select_dtypes(include=[np.number])


def _safe_series(df, col):
    if col not in df.columns:
        raise ValueError(f"Column not found: {col}")
    return df[col].dropna()


def get_plot_distribution(df, col, plot_type="hist", format="png", dpi=300):
    apply_nature_style()
    series = _safe_series(df, col)
    fig, ax = plt.subplots(figsize=(8, 5))

    if plot_type == "density":
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError("Density plot requires a numeric feature.")
        sns.kdeplot(series, fill=True, color=NATURE_COLORS["teal"], ax=ax)
        ax.set_ylabel("Density")
    elif plot_type == "ecdf":
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError("ECDF plot requires a numeric feature.")
        sns.ecdfplot(series, color=NATURE_COLORS["blue"], linewidth=2.2, ax=ax)
        ax.set_ylabel("Cumulative probability")
    elif pd.api.types.is_numeric_dtype(series):
        if plot_type == "hist":
            sns.histplot(series, kde=True, ax=ax, color=NATURE_COLORS["teal"], edgecolor="white", alpha=0.85)
            ax.set_ylabel("Count")
        elif plot_type == "box":
            sns.boxplot(y=series, ax=ax, color=NATURE_COLORS["mint"], width=0.4)
            ax.set_ylabel(col)
        elif plot_type == "violin":
            sns.violinplot(y=series, ax=ax, color="#c7dcef", inner="quartile")
            ax.set_ylabel(col)
        elif plot_type == "strip":
            sample = series.sample(min(len(series), 1200), random_state=42) if len(series) > 1200 else series
            sns.stripplot(y=sample, ax=ax, color=NATURE_COLORS["purple"], alpha=0.55, size=3)
            ax.set_ylabel(col)
        else:
            raise ValueError(f"Unsupported distribution plot: {plot_type}")
    else:
        counts = series.astype(str)
        if plot_type not in {"hist", "box", "violin", "strip"}:
            raise ValueError(f"{plot_type} is not supported for categorical features.")
        order = counts.value_counts().index[:20]
        sns.countplot(x=counts, order=order, ax=ax, palette="crest")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
        ax.set_ylabel("Count")

    ax.set_title(f"{col}: {plot_type.replace('_', ' ').title()}")
    ax.set_xlabel(col if plot_type not in {"box", "violin", "strip"} or not pd.api.types.is_numeric_dtype(series) else "")
    return fig_to_base64(fig, format, dpi)


def get_plot_scatter(df, col_x, col_y, format="png", dpi=300, plot_type="scatter", target_col=None):
    apply_nature_style()
    if col_x not in df.columns or col_y not in df.columns:
        raise ValueError("Selected columns are missing.")

    work_df = df[[col_x, col_y] + ([target_col] if target_col and target_col in df.columns else [])].dropna().copy()
    if work_df.empty:
        raise ValueError("No valid rows remain after removing missing values.")

    hue = target_col if target_col and target_col in work_df.columns and work_df[target_col].nunique() <= 8 else None

    if plot_type == "joint":
        g = sns.jointplot(
            data=work_df,
            x=col_x,
            y=col_y,
            kind="scatter",
            height=7,
            color=NATURE_COLORS["blue"],
            joint_kws={"alpha": 0.7, "s": 36, "edgecolor": "white", "linewidth": 0.4},
        )
        g.fig.suptitle(f"Joint Distribution: {col_x} vs {col_y}", y=1.02)
        return fig_to_base64(g.fig, format, dpi)

    fig, ax = plt.subplots(figsize=(8, 6))
    if plot_type == "hex":
        hb = ax.hexbin(work_df[col_x], work_df[col_y], gridsize=28, cmap="viridis", mincnt=1)
        fig.colorbar(hb, ax=ax, label="Counts")
    elif plot_type == "reg":
        sns.regplot(
            data=work_df,
            x=col_x,
            y=col_y,
            ax=ax,
            scatter_kws={"alpha": 0.65, "s": 38, "edgecolor": "white"},
            line_kws={"color": NATURE_COLORS["red"], "linewidth": 2.0},
            color=NATURE_COLORS["blue"],
        )
    else:
        sns.scatterplot(
            data=work_df,
            x=col_x,
            y=col_y,
            hue=hue,
            ax=ax,
            alpha=0.72,
            edgecolor="white",
            s=52,
            palette="deep",
        )
        if hue is None and pd.api.types.is_numeric_dtype(work_df[col_x]) and pd.api.types.is_numeric_dtype(work_df[col_y]):
            z = np.polyfit(work_df[col_x], work_df[col_y], 1)
            ax.plot(work_df[col_x], np.poly1d(z)(work_df[col_x]), color=NATURE_COLORS["red"], linewidth=2.0, alpha=0.8)

    ax.set_title(f"{plot_type.title()} Analysis: {col_y} vs {col_x}")
    return fig_to_base64(fig, format, dpi)


def get_plot_correlation(df, format="png", dpi=300, method="pearson"):
    apply_nature_style()
    numeric_df = _numeric_df(df)
    if numeric_df.empty:
        raise ValueError("No numeric columns available for correlation.")

    fig, ax = plt.subplots(figsize=(10, 8))
    corr = numeric_df.corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(f"Feature Correlation Heatmap ({method.title()})")
    return fig_to_base64(fig, format, dpi)


def get_plot_pairplot(df, format="png", dpi=300, target_col=None, max_cols=None):
    apply_nature_style()
    numeric_df = _numeric_df(df)
    if target_col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target_col])
    if max_cols is not None and numeric_df.shape[1] > max_cols:
        numeric_df = numeric_df.iloc[:, :max_cols]

    if numeric_df.empty:
        raise ValueError("Pair plot requires numeric features.")

    cols = numeric_df.columns.tolist()
    pair_df = df[cols].copy()
    hue = target_col if target_col in df.columns and df[target_col].nunique() <= 6 else None
    if hue:
        pair_df[hue] = df[hue]

    g = sns.pairplot(
        pair_df.dropna(),
        corner=True,
        diag_kind="kde",
        hue=hue,
        plot_kws={"alpha": 0.68, "s": 28, "edgecolor": "white", "linewidth": 0.3},
    )
    scope = f"Top {len(cols)} Features" if max_cols is not None else f"All {len(cols)} Numeric Features"
    g.fig.suptitle(f"Pairplot Analysis ({scope})", y=1.02)
    return fig_to_base64(g.fig, format, int(dpi * 0.8))


def get_plot_missingness(df, format="png", dpi=300):
    apply_nature_style()
    missing = df.isna().astype(int)
    if missing.sum().sum() == 0:
        raise ValueError("No missing values found in the dataset.")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(missing.T, cmap=["#f8fafc", "#d55e00"], cbar=False, ax=ax)
    ax.set_title("Missingness Map")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Feature")
    return fig_to_base64(fig, format, dpi)


def get_plot_target_relation(df, feature, target_col, format="png", dpi=300):
    apply_nature_style()
    if target_col not in df.columns or feature not in df.columns:
        raise ValueError("Feature or target column not found.")

    work_df = df[[feature, target_col]].dropna()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    target = work_df[target_col]
    feat = work_df[feature]

    if pd.api.types.is_numeric_dtype(feat) and pd.api.types.is_numeric_dtype(target):
        sns.regplot(
            data=work_df,
            x=feature,
            y=target_col,
            ax=ax,
            scatter_kws={"alpha": 0.65, "s": 38, "edgecolor": "white"},
            line_kws={"color": NATURE_COLORS["red"]},
            color=NATURE_COLORS["blue"],
        )
    elif not pd.api.types.is_numeric_dtype(feat) and pd.api.types.is_numeric_dtype(target):
        order = feat.astype(str).value_counts().index[:12]
        sns.boxplot(data=work_df.assign(**{feature: feat.astype(str)}), x=feature, y=target_col, order=order, ax=ax, color="#c7dcef")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    elif pd.api.types.is_numeric_dtype(feat) and not pd.api.types.is_numeric_dtype(target):
        order = target.astype(str).value_counts().index[:8]
        sns.violinplot(data=work_df.assign(**{target_col: target.astype(str)}), x=target_col, y=feature, order=order, ax=ax, color="#b7d6c6")
        ax.tick_params(axis="x", rotation=20)
    else:
        ct = pd.crosstab(feat.astype(str), target.astype(str), normalize="index")
        sns.heatmap(ct, annot=True, fmt=".2f", cmap="Blues", ax=ax)
        ax.set_ylabel(feature)
        ax.set_xlabel(target_col)

    ax.set_title(f"Feature-Target Relationship: {feature} vs {target_col}")
    return fig_to_base64(fig, format, dpi)


def get_preprocessing_plot(df_old, df_new, process_name, target_col, fmt="png", dpi=300):
    try:
        apply_nature_style()
        fig, ax = plt.subplots(figsize=(6.5, 4.5))

        if "Feature Map" in process_name:
            feats = [x["Feature"] for x in df_new][:20]
            scores = [x["Importance"] for x in df_new][:20]
            ax.barh(feats[::-1], scores[::-1], color=NATURE_COLORS["purple"], edgecolor="white")
            ax.set_xlabel("Relative importance")
            ax.set_title(f"Top {len(feats)} Predictive Features")
        else:
            num_cols = [c for c in df_old.columns if c in getattr(df_new, "columns", []) and pd.api.types.is_numeric_dtype(df_old[c]) and c != target_col]
            if len(num_cols) >= 2:
                c1, c2 = num_cols[:2]
                ax.scatter(df_old[c1], df_old[c2], alpha=0.28, label="Original", color=NATURE_COLORS["gray"], s=28)
                ax.scatter(df_new[c1], df_new[c2], alpha=0.65, label="Processed", color=NATURE_COLORS["red"], s=28)
                ax.set_xlabel(c1)
                ax.set_ylabel(c2)
                ax.set_title(f"State Shift: {process_name}")
                ax.legend()
            elif len(num_cols) == 1:
                c1 = num_cols[0]
                sns.kdeplot(df_old[c1], ax=ax, color=NATURE_COLORS["gray"], fill=True, alpha=0.25, label="Original")
                sns.kdeplot(df_new[c1], ax=ax, color=NATURE_COLORS["red"], fill=True, alpha=0.35, label="Processed")
                ax.set_title(f"Density Shift: {process_name}")
                ax.legend()
            else:
                ax.text(0.5, 0.5, "Requires >= 1 continuous numerical feature", ha="center", fontsize=10)
                ax.axis("off")

        return fig_to_base64(fig, fmt, dpi)
    except Exception:
        return None
