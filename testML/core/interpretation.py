import logging
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, partial_dependence, permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

try:
    import shap
except ImportError:
    shap = None

try:
    import dowhy
except ImportError:
    dowhy = None

try:
    import dice_ml
except ImportError:
    dice_ml = None

from core.plotting_utils import NATURE_COLORS, apply_nature_style, fig_to_base64

warnings.filterwarnings("ignore")


def _as_frame(X):
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


def _is_classifier(model):
    return hasattr(model, "predict_proba") or hasattr(model, "classes_")


def _model_output(model, X_df):
    if _is_classifier(model) and hasattr(model, "predict_proba"):
        prob = np.asarray(model.predict_proba(X_df))
        if prob.ndim == 2 and prob.shape[1] > 1:
            return prob[:, 1]
    return np.ravel(model.predict(X_df))


def _plot_shap_dependence_panel(X_sample, shap_matrix, mean_abs, fmt="png", dpi=300):
    top_features = list(X_sample.columns[np.argsort(mean_abs)[::-1][: min(6, X_sample.shape[1])]])
    if not top_features:
        return None
    rows = int(np.ceil(len(top_features) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(10, max(4.8, rows * 3.1)), squeeze=False)
    for ax, feature in zip(axes.ravel(), top_features):
        idx = X_sample.columns.get_loc(feature)
        x = pd.to_numeric(X_sample[feature], errors="coerce")
        y = shap_matrix[:, idx]
        mask = x.notna() & np.isfinite(y)
        ax.scatter(x[mask], y[mask], c=y[mask], cmap="coolwarm", s=28, alpha=0.72, edgecolors="white", linewidth=0.25)
        ax.axhline(0, color=NATURE_COLORS["slate"], linestyle="--", linewidth=1.0)
        if mask.sum() >= 8 and x[mask].nunique() >= 3:
            try:
                coef = np.polyfit(x[mask].to_numpy(dtype=float), y[mask], 2)
                grid = np.linspace(float(x[mask].min()), float(x[mask].max()), 120)
                ax.plot(grid, np.poly1d(coef)(grid), color=NATURE_COLORS["slate"], linewidth=1.6)
            except Exception:
                pass
        ax.set_xlabel(feature)
        ax.set_ylabel("SHAP value")
        ax.set_title(f"Dependence: {feature}")
    for ax in axes.ravel()[len(top_features):]:
        ax.axis("off")
    fig.suptitle("SHAP Top-Feature Dependence Panel", y=1.01)
    return fig_to_base64(fig, fmt, dpi)


def _plot_shap_cohort_heatmap(X_sample, shap_matrix, mean_abs, fmt="png", dpi=300):
    top_features = list(X_sample.columns[np.argsort(mean_abs)[::-1][: min(12, X_sample.shape[1])]])
    if not top_features:
        return None
    idx = [X_sample.columns.get_loc(f) for f in top_features]
    mat = pd.DataFrame(shap_matrix[:, idx], columns=top_features)
    mat = mat.iloc[: min(120, len(mat))].copy()
    order = np.argsort(np.abs(mat).sum(axis=1).to_numpy())[::-1]
    mat = mat.iloc[order]
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    sns.heatmap(mat, cmap="coolwarm", center=0, xticklabels=True, yticklabels=False, cbar_kws={"label": "SHAP value"}, ax=ax)
    ax.set_title("SHAP Cohort Contribution Heatmap")
    ax.set_xlabel("Top features")
    ax.set_ylabel("Samples sorted by attribution magnitude")
    return fig_to_base64(fig, fmt, dpi)


def _plot_shap_concentration_curve(mean_abs, feature_names, fmt="png", dpi=300):
    if len(mean_abs) == 0:
        return None
    order = np.argsort(mean_abs)[::-1]
    sorted_vals = np.asarray(mean_abs)[order]
    total = sorted_vals.sum()
    if total <= 0:
        return None
    cumulative = np.cumsum(sorted_vals) / total
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    x = np.arange(1, len(cumulative) + 1)
    axes[0].plot(x, cumulative, color=NATURE_COLORS["red"], linewidth=2.3, marker="o", markersize=4)
    axes[0].axhline(0.8, color=NATURE_COLORS["slate"], linestyle="--", linewidth=1.2, label="80% attribution")
    axes[0].set_xlabel("Top-k features")
    axes[0].set_ylabel("Cumulative mean |SHAP| share")
    axes[0].set_ylim(0, 1.03)
    axes[0].set_title("Attribution Concentration Curve")
    axes[0].legend()
    top = min(12, len(order))
    axes[1].barh(np.asarray(feature_names)[order[:top]][::-1], sorted_vals[:top][::-1] / total, color=NATURE_COLORS["purple"])
    axes[1].set_xlabel("Attribution share")
    axes[1].set_title("Top Attribution Share")
    return fig_to_base64(fig, fmt, dpi)


def _expected_scalar(expected_value):
    arr = np.asarray(expected_value)
    if arr.ndim == 0:
        return float(arr)
    return float(arr.ravel()[-1])


def _plot_designed_shap_beeswarm(X_sample, shap_matrix, mean_abs, fmt="png", dpi=300):
    top_idx = np.argsort(mean_abs)[::-1][: min(14, X_sample.shape[1])]
    if len(top_idx) == 0:
        return None
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(9.2, max(5.2, 0.42 * len(top_idx) + 2.0)))
    cmap = plt.get_cmap("coolwarm")
    for row_pos, feat_idx in enumerate(top_idx[::-1]):
        feature = X_sample.columns[feat_idx]
        shap_vals = np.asarray(shap_matrix[:, feat_idx], dtype=float)
        feature_vals = pd.to_numeric(X_sample[feature], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(shap_vals)
        if np.isfinite(feature_vals).sum() >= 3:
            lo, hi = np.nanpercentile(feature_vals, [2, 98])
            color_val = np.clip((feature_vals - lo) / (hi - lo + 1e-9), 0, 1)
        else:
            color_val = np.full(len(shap_vals), 0.5)
        density_rank = pd.Series(shap_vals).rank(pct=True).to_numpy()
        jitter = (rng.random(len(shap_vals)) - 0.5) * 0.62 * (0.35 + np.sin(density_rank * np.pi))
        size = 16 + 55 * np.clip(np.abs(shap_vals) / (np.nanpercentile(np.abs(shap_vals), 95) + 1e-9), 0, 1)
        ax.scatter(
            shap_vals[finite],
            np.full(finite.sum(), row_pos) + jitter[finite],
            c=color_val[finite],
            cmap=cmap,
            s=size[finite],
            alpha=0.78,
            edgecolors="white",
            linewidth=0.22,
        )
    ax.axvline(0, color=NATURE_COLORS["slate"], linestyle="--", linewidth=1.2)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels(X_sample.columns[top_idx[::-1]])
    ax.set_xlabel("SHAP value")
    ax.set_title("Designed SHAP Beeswarm Density")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Low -> high feature value")
    return fig_to_base64(fig, fmt, dpi)


def _plot_shap_interaction_constellation(X_sample, shap_matrix, mean_abs, fmt="png", dpi=300):
    top_idx = np.argsort(mean_abs)[::-1][: min(10, X_sample.shape[1])]
    if len(top_idx) < 3:
        return None
    names = X_sample.columns[top_idx].tolist()
    mat = pd.DataFrame(shap_matrix[:, top_idx], columns=names)
    corr = mat.corr().fillna(0)
    G = nx.Graph()
    max_imp = float(np.max(mean_abs[top_idx]) + 1e-9)
    for feature, idx in zip(names, top_idx):
        G.add_node(feature, size=mean_abs[idx] / max_imp)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            weight = float(corr.loc[a, b])
            if abs(weight) >= 0.18:
                G.add_edge(a, b, weight=weight)
    pos = nx.circular_layout(G)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.05, 0.95]})
    axes[0].axis("off")
    for u, v, data in G.edges(data=True):
        w = data["weight"]
        color = NATURE_COLORS["red"] if w > 0 else NATURE_COLORS["blue"]
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=0.6 + 5.0 * abs(w), alpha=0.55, edge_color=color, ax=axes[0])
    node_sizes = [650 + 2600 * G.nodes[n]["size"] for n in G.nodes]
    node_colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, len(G.nodes)))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, edgecolors="white", linewidths=1.2, ax=axes[0])
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=axes[0])
    axes[0].set_title("SHAP Interaction Constellation")
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", linewidths=0.35, cbar=False, ax=axes[1])
    axes[1].set_title("Attribution Co-Movement")
    return fig_to_base64(fig, fmt, dpi)


def _plot_local_explanation_storyboard(X_sample, shap_matrix, expected_value, sample_idx, fmt="png", dpi=300):
    sample_idx = int(np.clip(sample_idx, 0, len(X_sample) - 1))
    values = np.asarray(shap_matrix[sample_idx], dtype=float)
    order = np.argsort(np.abs(values))[::-1][: min(12, len(values))]
    features = X_sample.columns[order].to_numpy()
    contrib = values[order]
    base = _expected_scalar(expected_value)
    cumulative = base + np.cumsum(contrib)
    colors = [NATURE_COLORS["red"] if v >= 0 else NATURE_COLORS["blue"] for v in contrib]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), gridspec_kw={"width_ratios": [1.2, 0.8]})
    y = np.arange(len(features))
    lefts = np.r_[base, cumulative[:-1]]
    axes[0].barh(y, contrib, left=lefts, color=colors, alpha=0.88, edgecolor="white", linewidth=0.6)
    axes[0].plot(np.r_[base, cumulative], np.r_[-0.6, y], color=NATURE_COLORS["slate"], linewidth=1.5, alpha=0.75)
    axes[0].axvline(base, color=NATURE_COLORS["slate"], linestyle="--", linewidth=1.1, label=f"Base={base:.3g}")
    axes[0].axvline(cumulative[-1], color="black", linewidth=1.5, label=f"Final={cumulative[-1]:.3g}")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(features)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Prediction path")
    axes[0].set_title(f"Local Explanation Storyboard: sample {sample_idx}")
    axes[0].legend(fontsize=8)

    feature_values = X_sample.iloc[sample_idx][features]
    display = pd.DataFrame({"Feature": features, "Value": feature_values.astype(str), "SHAP": contrib})
    display["Direction"] = np.where(display["SHAP"] >= 0, "push up", "push down")
    axes[1].axis("off")
    table = axes[1].table(
        cellText=display.round(4).values,
        colLabels=display.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    axes[1].set_title("Feature Evidence Cards")
    return fig_to_base64(fig, fmt, dpi)


def _plot_attribution_terrain(X_sample, shap_matrix, model, fmt="png", dpi=300):
    if len(X_sample) < 12 or shap_matrix.shape[1] < 2:
        return None
    coords = PCA(n_components=2, random_state=42).fit_transform(shap_matrix)
    pred = _model_output(model, X_sample)
    strength = np.abs(shap_matrix).sum(axis=1)
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    try:
        contour = ax.tricontourf(coords[:, 0], coords[:, 1], pred, levels=18, cmap="viridis", alpha=0.72)
        fig.colorbar(contour, ax=ax, label="Prediction terrain")
    except Exception:
        pass
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=strength, cmap="magma_r", s=32 + 75 * strength / (np.nanpercentile(strength, 95) + 1e-9), alpha=0.82, edgecolors="white", linewidth=0.32)
    fig.colorbar(sc, ax=ax, label="Total |SHAP|")
    ax.set_xlabel("Explanation PC1")
    ax.set_ylabel("Explanation PC2")
    ax.set_title("Attribution Terrain Map")
    return fig_to_base64(fig, fmt, dpi)


def _plot_explanation_fingerprint(X_sample, shap_matrix, mean_abs, sample_idx, fmt="png", dpi=300):
    top_idx = np.argsort(mean_abs)[::-1][: min(12, X_sample.shape[1])]
    if len(top_idx) < 3:
        return None
    sample_idx = int(np.clip(sample_idx, 0, len(X_sample) - 1))
    strength = np.abs(shap_matrix[:, top_idx])
    strength = strength / (strength.sum(axis=1, keepdims=True) + 1e-9)
    total = np.abs(shap_matrix).sum(axis=1)
    selected = [sample_idx]
    selected += list(np.argsort(total)[-4:])
    selected += list(np.argsort(total)[:4])
    selected = list(dict.fromkeys([idx for idx in selected if 0 <= idx < len(X_sample)]))[:9]
    labels = [f"S{idx}" + ("*" if idx == sample_idx else "") for idx in selected]
    theta = np.linspace(0, 2 * np.pi, len(top_idx), endpoint=False)
    theta = np.r_[theta, theta[0]]
    fig, ax = plt.subplots(figsize=(7.4, 7.0), subplot_kw={"projection": "polar"})
    colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, len(selected)))
    for color, idx, label in zip(colors, selected, labels):
        vals = np.r_[strength[idx], strength[idx, 0]]
        ax.plot(theta, vals, color=color, linewidth=1.8, label=label)
        ax.fill(theta, vals, color=color, alpha=0.055)
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(X_sample.columns[top_idx], fontsize=8)
    ax.set_title("Local Explanation Fingerprints")
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.18, 1.13), frameon=False)
    return fig_to_base64(fig, fmt, dpi)


def _plot_designed_xai_suite(X_sample, shap_matrix, mean_abs, model, expected_value, sample_idx, fmt="png", dpi=300):
    plots = {}
    candidates = {
        "Designed SHAP Beeswarm Density": _plot_designed_shap_beeswarm(X_sample, shap_matrix, mean_abs, fmt, dpi),
        "SHAP Interaction Constellation": _plot_shap_interaction_constellation(X_sample, shap_matrix, mean_abs, fmt, dpi),
        "Local Explanation Storyboard": _plot_local_explanation_storyboard(X_sample, shap_matrix, expected_value, sample_idx, fmt, dpi),
        "Attribution Terrain Map": _plot_attribution_terrain(X_sample, shap_matrix, model, fmt, dpi),
        "Local Explanation Fingerprints": _plot_explanation_fingerprint(X_sample, shap_matrix, mean_abs, sample_idx, fmt, dpi),
    }
    for name, image in candidates.items():
        if image:
            plots[name] = image
    return plots


def _feature_percentiles(X_sample, row):
    percentiles = {}
    for col in X_sample.columns:
        series = pd.to_numeric(X_sample[col], errors="coerce")
        try:
            value = float(row[col])
            percentiles[col] = float((series <= value).mean())
        except Exception:
            percentiles[col] = np.nan
    return percentiles


def _plot_prediction_reason_wheel(X_sample, shap_matrix, expected_value, sample_idx, fmt="png", dpi=300):
    sample_idx = int(np.clip(sample_idx, 0, len(X_sample) - 1))
    vals = np.asarray(shap_matrix[sample_idx], dtype=float)
    names = np.asarray(X_sample.columns)
    order = np.argsort(np.abs(vals))[::-1]
    top = order[: min(8, len(order))]
    other = order[min(8, len(order)):]
    wheel_vals = np.abs(vals[top]).tolist()
    wheel_labels = names[top].tolist()
    wheel_colors = [NATURE_COLORS["red"] if vals[i] >= 0 else NATURE_COLORS["blue"] for i in top]
    if len(other):
        wheel_vals.append(float(np.abs(vals[other]).sum()))
        wheel_labels.append("Other")
        wheel_colors.append(NATURE_COLORS["gray"])
    base = _expected_scalar(expected_value)
    final = base + float(vals.sum())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), gridspec_kw={"width_ratios": [0.92, 1.08]})
    wedges, _ = axes[0].pie(
        wheel_vals,
        startangle=90,
        colors=wheel_colors,
        wedgeprops={"width": 0.38, "edgecolor": "white", "linewidth": 1.2},
    )
    axes[0].text(0, 0.1, f"{final:.3g}", ha="center", va="center", fontsize=22, fontweight="bold", color=NATURE_COLORS["slate"])
    axes[0].text(0, -0.12, "prediction", ha="center", va="center", fontsize=9, color=NATURE_COLORS["slate"])
    axes[0].set_title("Prediction Reason Wheel")
    axes[0].legend(wedges, wheel_labels, loc="center left", bbox_to_anchor=(0.98, 0.5), fontsize=8, frameon=False)

    signed = pd.DataFrame({"Feature": names[top], "Contribution": vals[top]})
    signed = signed.sort_values("Contribution")
    axes[1].barh(
        signed["Feature"],
        signed["Contribution"],
        color=[NATURE_COLORS["red"] if v > 0 else NATURE_COLORS["blue"] for v in signed["Contribution"]],
    )
    axes[1].axvline(0, color=NATURE_COLORS["slate"], linestyle="--", linewidth=1)
    axes[1].axvline(final - base, color="black", linewidth=1.2, alpha=0.65, label=f"net shift={final-base:.3g}")
    axes[1].set_xlabel("SHAP contribution")
    axes[1].set_title(f"Base {base:.3g} -> Final {final:.3g}")
    axes[1].legend(fontsize=8)
    return fig_to_base64(fig, fmt, dpi)


def _plot_plain_language_factor_cards(X_sample, shap_matrix, sample_idx, fmt="png", dpi=300):
    sample_idx = int(np.clip(sample_idx, 0, len(X_sample) - 1))
    row = X_sample.iloc[sample_idx]
    vals = np.asarray(shap_matrix[sample_idx], dtype=float)
    names = np.asarray(X_sample.columns)
    pct = _feature_percentiles(X_sample, row)
    order = np.argsort(np.abs(vals))[::-1][: min(10, len(vals))]
    cards = []
    for idx in order:
        direction = "raises" if vals[idx] >= 0 else "lowers"
        p = pct.get(names[idx], np.nan)
        if np.isnan(p):
            position = "unknown population position"
        elif p >= 0.75:
            position = "high value"
        elif p <= 0.25:
            position = "low value"
        else:
            position = "mid-range value"
        cards.append({
            "Feature": names[idx],
            "Value": row[names[idx]],
            "Percentile": p,
            "Effect": vals[idx],
            "Message": f"{position} {direction} prediction",
        })
    card_df = pd.DataFrame(cards)
    fig, axes = plt.subplots(1, 2, figsize=(12, max(5.2, 0.35 * len(card_df) + 2)))
    axes[0].axis("off")
    display = card_df[["Feature", "Value", "Percentile", "Effect", "Message"]].copy()
    display["Percentile"] = display["Percentile"].map(lambda x: "" if pd.isna(x) else f"{x:.0%}")
    display["Effect"] = display["Effect"].map(lambda x: f"{x:+.3g}")
    table = axes[0].table(cellText=display.values, colLabels=display.columns, loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    axes[0].set_title("Plain-Language Factor Cards")

    y = np.arange(len(card_df))
    axes[1].barh(y, card_df["Effect"], color=[NATURE_COLORS["red"] if v >= 0 else NATURE_COLORS["blue"] for v in card_df["Effect"]])
    axes[1].scatter(card_df["Effect"], y, s=80 + 220 * card_df["Percentile"].fillna(0.5), color="white", edgecolors=NATURE_COLORS["slate"], linewidth=1.0, zorder=3)
    axes[1].axvline(0, color=NATURE_COLORS["slate"], linestyle="--")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(card_df["Feature"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Contribution; marker size follows feature percentile")
    axes[1].set_title("Effect Direction and Population Position")
    return fig_to_base64(fig, fmt, dpi)


def _plot_what_if_response_ladder(model, X_sample, shap_matrix, sample_idx, fmt="png", dpi=300):
    sample_idx = int(np.clip(sample_idx, 0, len(X_sample) - 1))
    row = X_sample.iloc[[sample_idx]].copy()
    vals = np.asarray(shap_matrix[sample_idx], dtype=float)
    top_features = [f for f in X_sample.columns[np.argsort(np.abs(vals))[::-1]] if pd.api.types.is_numeric_dtype(X_sample[f])][: min(6, X_sample.shape[1])]
    if not top_features:
        return None
    base_pred = float(_model_output(model, row)[0])
    records = []
    for feature in top_features:
        series = pd.to_numeric(X_sample[feature], errors="coerce").dropna()
        if series.nunique() < 3:
            continue
        for label, q in [("low P10", 0.10), ("median P50", 0.50), ("high P90", 0.90)]:
            candidate = row.copy()
            candidate[feature] = float(series.quantile(q))
            pred = float(_model_output(model, candidate)[0])
            records.append({"Feature": feature, "Intervention": label, "Prediction": pred, "Delta": pred - base_pred})
    df = pd.DataFrame(records)
    if df.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, max(5.2, 0.55 * len(top_features) + 2)))
    sns.barplot(data=df, y="Feature", x="Delta", hue="Intervention", palette="vlag", ax=axes[0])
    axes[0].axvline(0, color=NATURE_COLORS["slate"], linestyle="--")
    axes[0].set_title("What-If Response Ladder")
    axes[0].set_xlabel("Prediction change from current sample")
    pivot = df.pivot(index="Feature", columns="Intervention", values="Prediction")
    sns.heatmap(pivot, annot=True, fmt=".3g", cmap="mako", linewidths=0.4, ax=axes[1])
    axes[1].set_title(f"Intervention Predictions (current={base_pred:.3g})")
    return fig_to_base64(fig, fmt, dpi)


def _plot_similar_case_contrast(model, X_sample, shap_matrix, sample_idx, fmt="png", dpi=300):
    X_num = X_sample.select_dtypes(include=[np.number]).copy()
    if X_num.shape[1] < 2 or len(X_num) < 8:
        return None
    sample_idx = int(np.clip(sample_idx, 0, len(X_sample) - 1))
    filled = X_num.fillna(X_num.median())
    scale = filled.std().replace(0, 1.0)
    Z = (filled - filled.mean()) / scale
    nn = NearestNeighbors(n_neighbors=min(8, len(Z))).fit(Z)
    _, neighbors = nn.kneighbors(Z.iloc[[sample_idx]])
    neighbor_idx = [idx for idx in neighbors[0] if idx != sample_idx][:5]
    selected = [sample_idx] + neighbor_idx
    preds = _model_output(model, X_sample.iloc[selected])
    top_idx = np.argsort(np.abs(shap_matrix[sample_idx]))[::-1][: min(8, shap_matrix.shape[1])]
    top_features = X_sample.columns[top_idx]
    contrib = pd.DataFrame(shap_matrix[selected][:, top_idx], index=[f"sample {i}" for i in selected], columns=top_features)
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4.8, 0.45 * len(selected) + 2)))
    sns.heatmap(contrib, cmap="coolwarm", center=0, annot=True, fmt=".2g", linewidths=0.35, ax=axes[0])
    axes[0].set_title("Similar-Case Explanation Contrast")
    bars = axes[1].barh([f"sample {i}" for i in selected], preds, color=[NATURE_COLORS["red"]] + [NATURE_COLORS["teal"]] * (len(selected) - 1))
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Prediction")
    axes[1].set_title("Prediction Among Nearest Similar Cases")
    for bar, pred in zip(bars, preds):
        axes[1].text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {pred:.3g}", va="center", fontsize=8)
    return fig_to_base64(fig, fmt, dpi)


def _plot_mechanism_quadrant_map(X_sample, shap_matrix, mean_abs, fmt="png", dpi=300):
    top_idx = np.argsort(mean_abs)[::-1][: min(12, X_sample.shape[1])]
    rows = []
    for idx in top_idx:
        feature = X_sample.columns[idx]
        x = pd.to_numeric(X_sample[feature], errors="coerce")
        s = np.asarray(shap_matrix[:, idx], dtype=float)
        mask = x.notna() & np.isfinite(s)
        if mask.sum() < 6:
            continue
        x_rank = x[mask].rank(pct=True).to_numpy(dtype=float)
        high_mask = x_rank >= 0.75
        low_mask = x_rank <= 0.25
        rows.append({
            "Feature": feature,
            "High value effect": float(np.mean(s[mask][high_mask])) if high_mask.any() else 0.0,
            "Low value effect": float(np.mean(s[mask][low_mask])) if low_mask.any() else 0.0,
            "Mean |SHAP|": float(mean_abs[idx]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.2, 6.3))
    sc = ax.scatter(
        df["Low value effect"],
        df["High value effect"],
        s=380 * df["Mean |SHAP|"] / (df["Mean |SHAP|"].max() + 1e-9) + 80,
        c=df["Mean |SHAP|"],
        cmap="viridis",
        edgecolors="white",
        linewidth=0.8,
        alpha=0.88,
    )
    ax.axhline(0, color=NATURE_COLORS["slate"], linestyle="--")
    ax.axvline(0, color=NATURE_COLORS["slate"], linestyle="--")
    for _, row in df.iterrows():
        ax.text(row["Low value effect"], row["High value effect"], f" {row['Feature']}", fontsize=8, va="center")
    ax.set_xlabel("Effect when feature is low")
    ax.set_ylabel("Effect when feature is high")
    ax.set_title("Mechanism Quadrant Map")
    fig.colorbar(sc, ax=ax, label="Mean |SHAP|")
    return fig_to_base64(fig, fmt, dpi)


def _plot_intuitive_xai_suite(X_sample, shap_matrix, mean_abs, model, expected_value, sample_idx, fmt="png", dpi=300):
    plots = {}
    candidates = {
        "Prediction Reason Wheel": _plot_prediction_reason_wheel(X_sample, shap_matrix, expected_value, sample_idx, fmt, dpi),
        "Plain-Language Factor Cards": _plot_plain_language_factor_cards(X_sample, shap_matrix, sample_idx, fmt, dpi),
        "What-If Response Ladder": _plot_what_if_response_ladder(model, X_sample, shap_matrix, sample_idx, fmt, dpi),
        "Similar-Case Explanation Contrast": _plot_similar_case_contrast(model, X_sample, shap_matrix, sample_idx, fmt, dpi),
        "Mechanism Quadrant Map": _plot_mechanism_quadrant_map(X_sample, shap_matrix, mean_abs, fmt, dpi),
    }
    for name, image in candidates.items():
        if image:
            plots[name] = image
    return plots


def _plot_ice_heterogeneity(model, X_train, feature, fmt="png", dpi=300):
    X_train = _as_frame(X_train).copy()
    if feature not in X_train.columns:
        return None
    vals = pd.to_numeric(X_train[feature], errors="coerce")
    valid = vals.notna()
    X_valid = X_train.loc[valid].copy()
    vals = vals.loc[valid]
    if len(X_valid) < 12 or vals.nunique() < 4:
        return None
    sample = X_valid.sample(min(90, len(X_valid)), random_state=42)
    grid = np.quantile(vals, np.linspace(0.02, 0.98, 30))
    grid = np.unique(grid)
    if len(grid) < 4:
        return None
    curves = []
    for _, row in sample.iterrows():
        X_rep = pd.DataFrame([row.to_dict()] * len(grid))
        X_rep[feature] = grid
        curves.append(_model_output(model, X_rep))
    curves = np.asarray(curves, dtype=float)
    centered = curves - curves[:, [0]]
    p10, p50, p90 = np.quantile(centered, [0.1, 0.5, 0.9], axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for curve in centered[:45]:
        axes[0].plot(grid, curve, color=NATURE_COLORS["gray"], alpha=0.25, linewidth=0.9)
    axes[0].plot(grid, p50, color=NATURE_COLORS["red"], linewidth=2.3, label="Median ICE")
    axes[0].fill_between(grid, p10, p90, color=NATURE_COLORS["red"], alpha=0.13, label="10-90% band")
    axes[0].axhline(0, color=NATURE_COLORS["slate"], linestyle="--", linewidth=1)
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel("Centered prediction change")
    axes[0].set_title("ICE Heterogeneity Envelope")
    axes[0].legend()
    slopes = np.gradient(centered, grid, axis=1)
    slope_summary = np.nanmedian(np.abs(slopes), axis=0)
    axes[1].plot(grid, slope_summary, color=NATURE_COLORS["purple"], linewidth=2.2)
    axes[1].fill_between(grid, np.nanquantile(np.abs(slopes), 0.1, axis=0), np.nanquantile(np.abs(slopes), 0.9, axis=0), color=NATURE_COLORS["purple"], alpha=0.14)
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel("|local slope|")
    axes[1].set_title("Local Sensitivity Spectrum")
    return fig_to_base64(fig, fmt, dpi)


def _extract_shap_matrix(shap_values):
    if hasattr(shap_values, "values"):
        values = shap_values.values
        if values.ndim == 3:
            return values[:, :, 1]
        return values
    if isinstance(shap_values, list):
        return shap_values[1] if len(shap_values) > 1 else shap_values[0]
    values = np.asarray(shap_values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def _safe_numeric_df(df, target_col):
    work_df = df.copy()
    for col in work_df.columns:
        if col == target_col:
            continue
        if not pd.api.types.is_numeric_dtype(work_df[col]):
            codes, uniques = pd.factorize(work_df[col], sort=True)
            work_df[col] = pd.Series(codes, index=work_df.index).replace(-1, np.nan)
    if not pd.api.types.is_numeric_dtype(work_df[target_col]):
        codes, uniques = pd.factorize(work_df[target_col], sort=True)
        work_df[target_col] = pd.Series(codes, index=work_df.index).replace(-1, np.nan)
    work_df = work_df.replace([np.inf, -np.inf], np.nan).dropna()
    return work_df


def _bootstrap_ci(values, alpha=0.05):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None, None
    return float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))


def _ale_curve(model, X_train, feature, bins=20, bootstrap_rounds=18):
    X_train = _as_frame(X_train).copy()
    if feature not in X_train.columns:
        raise ValueError(f"Feature {feature} not found for ALE.")
    x = pd.to_numeric(X_train[feature], errors="coerce")
    valid_mask = x.notna()
    X_train = X_train.loc[valid_mask].reset_index(drop=True)
    x = x.loc[valid_mask].reset_index(drop=True)
    if len(X_train) < 12:
        raise ValueError("Not enough valid rows for ALE.")

    quantiles = np.unique(np.quantile(x, np.linspace(0, 1, min(max(6, bins + 1), len(np.unique(x))))))
    if len(quantiles) < 3:
        spread = float(np.std(x) or 1.0)
        quantiles = np.linspace(float(x.min()), float(x.max()) + spread * 1e-3, 6)

    def _single_curve(X_ref):
        effects = []
        centers = []
        counts = []
        for left, right in zip(quantiles[:-1], quantiles[1:]):
            if right <= left:
                continue
            mask = (x >= left) & (x < right if right < quantiles[-1] else x <= right)
            if not mask.any():
                continue
            lower_df = X_ref.loc[mask].copy()
            upper_df = X_ref.loc[mask].copy()
            lower_df[feature] = left
            upper_df[feature] = right
            diff = _model_output(model, upper_df) - _model_output(model, lower_df)
            effects.append(float(np.mean(diff)))
            centers.append((left + right) / 2)
            counts.append(int(mask.sum()))
        if not effects:
            raise ValueError("Failed to compute ALE increments.")
        ale = np.cumsum(effects)
        weights = np.asarray(counts, dtype=float)
        ale = ale - np.average(ale, weights=weights)
        return np.asarray(centers, dtype=float), np.asarray(ale, dtype=float)

    centers, ale = _single_curve(X_train)
    boot_curves = []
    rng = np.random.default_rng(42)
    for _ in range(bootstrap_rounds):
        idx = rng.integers(0, len(X_train), len(X_train))
        X_boot = X_train.iloc[idx].reset_index(drop=True)
        try:
            _, boot_ale = _single_curve(X_boot)
            if len(boot_ale) == len(ale):
                boot_curves.append(boot_ale)
        except Exception:
            continue
    if boot_curves:
        boot_arr = np.asarray(boot_curves, dtype=float)
        lower = np.quantile(boot_arr, 0.025, axis=0)
        upper = np.quantile(boot_arr, 0.975, axis=0)
    else:
        lower = ale
        upper = ale
    return centers, ale, lower, upper


def _official_dice_counterfactuals(model, X_train, x0, desired_value=None, total_cfs=4):
    if dice_ml is None:
        raise ImportError("dice-ml is not installed.")
    X_train = _as_frame(X_train).copy()
    x0 = _as_frame(x0).copy()
    outcome_name = "__dice_target__"
    train_df = X_train.copy()
    if _is_classifier(model):
        train_df[outcome_name] = model.predict(X_train)
        desired_class = "opposite"
        if desired_value is not None:
            desired_class = int(desired_value)
    else:
        preds = np.ravel(model.predict(X_train))
        train_df[outcome_name] = preds
        center = float(desired_value) if desired_value is not None else float(np.ravel(model.predict(x0))[0] + np.std(preds) * 0.75)
        span = max(float(np.std(preds) * 0.2), 1e-6)
        desired_class = None
        desired_range = [center - span, center + span]

    dice_data = dice_ml.Data(
        dataframe=train_df,
        continuous_features=list(X_train.columns),
        outcome_name=outcome_name,
    )
    dice_model = dice_ml.Model(
        model=model,
        backend="sklearn",
        model_type="classifier" if _is_classifier(model) else "regressor",
    )
    explainer = dice_ml.Dice(dice_data, dice_model, method="random")
    kwargs = {"query_instances": x0, "total_CFs": total_cfs, "features_to_vary": "all"}
    if _is_classifier(model):
        kwargs["desired_class"] = desired_class
    else:
        kwargs["desired_range"] = desired_range
    dice_exp = explainer.generate_counterfactuals(**kwargs)
    cf_df = dice_exp.cf_examples_list[0].final_cfs_df
    if cf_df is None or cf_df.empty:
        raise ValueError("Official DiCE returned no counterfactuals.")
    return cf_df.reset_index(drop=True), "Official DiCE (dice-ml)"


def _fallback_counterfactuals(model, X_train, x0, desired_value=None, total_cfs=4):
    X_train = _as_frame(X_train)
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("Counterfactual search requires numeric features.")
    original_pred = float(np.ravel(model.predict(x0))[0])
    target_value = float(desired_value) if desired_value is not None else original_pred + np.std(np.ravel(model.predict(X_train))) * 0.75
    rng = np.random.default_rng(42)
    scales = X_train[numeric_cols].std().replace(0, 1.0)
    candidates = []
    for _ in range(300):
        candidate = x0.copy()
        n_mut = int(rng.integers(1, min(4, len(numeric_cols)) + 1))
        chosen = rng.choice(numeric_cols, size=n_mut, replace=False)
        for col in chosen:
            candidate[col] = float(candidate[col].iloc[0] + rng.normal(0, scales[col] * 0.8))
            candidate[col] = np.clip(candidate[col], X_train[col].min(), X_train[col].max())
        pred = float(np.ravel(model.predict(candidate))[0])
        delta = np.abs(candidate[numeric_cols].iloc[0].to_numpy(dtype=float) - x0[numeric_cols].iloc[0].to_numpy(dtype=float))
        score = abs(pred - target_value) + 0.15 * np.sum(delta / (scales.to_numpy(dtype=float) + 1e-9))
        candidates.append((score, pred, candidate.iloc[0].copy()))
    candidates.sort(key=lambda item: item[0])
    diverse = []
    for _, pred, row in candidates:
        if len(diverse) >= total_cfs:
            break
        if not diverse:
            diverse.append((pred, row))
            continue
        distances = [
            np.linalg.norm(
                (row[numeric_cols].to_numpy(dtype=float) - other[numeric_cols].to_numpy(dtype=float))
                / (scales.to_numpy(dtype=float) + 1e-9)
            )
            for _, other in diverse
        ]
        if min(distances) > 0.8:
            diverse.append((pred, row))
    if not diverse:
        raise ValueError("No diverse counterfactual candidates found.")
    cf_df = pd.DataFrame([row for _, row in diverse]).reset_index(drop=True)
    cf_df["Prediction"] = [pred for pred, _ in diverse]
    return cf_df, "DiCE-style diverse counterfactual search"


def _normalize_ci(point, low, high):
    point = float(point)
    if low is None or high is None or pd.isna(low) or pd.isna(high):
        return point, point
    low = float(low)
    high = float(high)
    if low > high:
        low, high = high, low
    low = min(low, point)
    high = max(high, point)
    return low, high


def _adjusted_effect_single(df, treatment, target, n_boot=60):
    cols = [treatment] + [c for c in df.columns if c not in {treatment, target}]
    X = df[cols].copy()
    y = df[target].astype(float).values
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    point = float(model.coef_[0])

    boot = []
    n = len(df)
    rng = np.random.default_rng(42)
    for _ in range(min(n_boot, max(20, n // 5))):
        idx = rng.integers(0, n, n)
        Xb = X.iloc[idx]
        yb = y[idx]
        try:
            mb = Ridge(alpha=1.0).fit(Xb, yb)
            boot.append(float(mb.coef_[0]))
        except Exception:
            continue
    ci_low, ci_high = _bootstrap_ci(boot)
    return point, ci_low, ci_high


def _partial_linear_dml_single(df, treatment, target, n_splits=3):
    features = [c for c in df.columns if c not in {treatment, target}]
    if not features:
        return None

    X = df[features]
    t = df[treatment].astype(float).values
    y = df[target].astype(float).values
    n_splits = max(2, min(n_splits, len(df) // 20 if len(df) >= 40 else 2))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_resid = np.zeros(len(df))
    t_resid = np.zeros(len(df))

    for train_idx, test_idx in kf.split(X):
        rf_y = RandomForestRegressor(n_estimators=120, random_state=42, min_samples_leaf=3)
        rf_t = RandomForestRegressor(n_estimators=120, random_state=43, min_samples_leaf=3)
        rf_y.fit(X.iloc[train_idx], y[train_idx])
        rf_t.fit(X.iloc[train_idx], t[train_idx])
        y_resid[test_idx] = y[test_idx] - rf_y.predict(X.iloc[test_idx])
        t_resid[test_idx] = t[test_idx] - rf_t.predict(X.iloc[test_idx])

    denom = float(np.dot(t_resid, t_resid))
    if abs(denom) < 1e-9:
        return None
    theta = float(np.dot(t_resid, y_resid) / denom)
    resid = y_resid - theta * t_resid
    se = float(np.sqrt(np.mean(resid**2) / (denom + 1e-9)))
    return theta, theta - 1.96 * se, theta + 1.96 * se


def _naive_effect_single(df, treatment, target):
    x = df[treatment].astype(float).values
    y = df[target].astype(float).values
    if np.std(x) < 1e-12:
        return 0.0
    return float(np.cov(x, y, ddof=0)[0, 1] / (np.var(x) + 1e-9))


def _dowhy_effect_single(df, treatment, target):
    if dowhy is None:
        return None
    common_causes = [c for c in df.columns if c not in {treatment, target}]
    model = dowhy.CausalModel(
        data=df,
        treatment=str(treatment),
        outcome=str(target),
        common_causes=[str(c) for c in common_causes],
        logging_level=logging.CRITICAL,
    )
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
    return {
        "value": float(estimate.value),
        "estimand": str(identified_estimand),
        "method": "DoWhy backdoor.linear_regression",
    }


def _build_causal_graph(ranking_df, target_col, fmt="png", dpi=300):
    apply_nature_style()
    ranking_df = ranking_df.copy().head(24)
    G = nx.DiGraph()
    G.add_node(target_col)
    for _, row in ranking_df.iterrows():
        G.add_node(row["Feature"])
        G.add_edge(row["Feature"], target_col, weight=row["ATE"])

    fig, ax = plt.subplots(figsize=(11, max(6.0, 0.42 * len(ranking_df) + 2.5)))
    ordered = ranking_df.sort_values("ATE", ascending=False).reset_index(drop=True)
    ypos = np.linspace(0.92, 0.08, len(ordered))
    max_abs = max(ordered["ATE"].abs().max(), 1e-9)

    ax.scatter([0.82], [0.5], s=3600, color=NATURE_COLORS["purple"], alpha=0.95, zorder=4)
    ax.text(0.82, 0.5, target_col, ha="center", va="center", color="white", fontsize=11, fontweight="bold", zorder=5)

    for y, (_, row) in zip(ypos, ordered.iterrows()):
        effect = float(row["ATE"])
        color = NATURE_COLORS["red"] if effect > 0 else NATURE_COLORS["blue"]
        size = 450 + 1900 * abs(effect) / max_abs
        ax.scatter([0.18], [y], s=size, color=NATURE_COLORS["teal"], alpha=0.88, edgecolors="white", linewidths=1.0, zorder=4)
        ax.text(0.03, y, row["Feature"], ha="left", va="center", fontsize=9.5, fontweight="bold", color=NATURE_COLORS["slate"])
        ax.annotate(
            "",
            xy=(0.75, 0.5),
            xytext=(0.22, y),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=1.4 + 4.0 * abs(effect) / max_abs,
                color=color,
                alpha=0.8,
                shrinkA=10,
                shrinkB=18,
                connectionstyle="arc3,rad=0.0",
            ),
            zorder=3,
        )
        ax.text(0.49, (y + 0.5) / 2, f"{effect:.3f}", fontsize=8.5, color=color, fontweight="bold", ha="center", va="center")

    ax.text(0.18, 0.98, "Input Features", ha="center", va="center", fontsize=10, fontweight="bold", color=NATURE_COLORS["teal"])
    ax.text(0.82, 0.98, "Target", ha="center", va="center", fontsize=10, fontweight="bold", color=NATURE_COLORS["purple"])
    ax.set_title("Global Causal Influence Map")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig_to_base64(fig, fmt, dpi)


def _build_causal_ranking_plot(ranking_df, fmt="png", dpi=300):
    apply_nature_style()
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.45 * len(ranking_df) + 2)))
    colors = [NATURE_COLORS["red"] if x > 0 else NATURE_COLORS["blue"] for x in ranking_df["ATE"]]
    ax.barh(ranking_df["Feature"][::-1], ranking_df["ATE"][::-1], color=colors[::-1])
    ax.axvline(0, color=NATURE_COLORS["slate"], linewidth=1.3)
    ax.set_xlabel("Estimated treatment effect")
    ax.set_title("ATE Ranking")
    return fig_to_base64(fig, fmt, dpi)


def _build_causal_forest_plot(ranking_df, fmt="png", dpi=300):
    apply_nature_style()
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.45 * len(ranking_df) + 2)))
    ordered = ranking_df.copy()
    ci_pairs = ordered.apply(lambda row: _normalize_ci(row["ATE"], row["CI Low"], row["CI High"]), axis=1)
    ordered["CI Low"] = [pair[0] for pair in ci_pairs]
    ordered["CI High"] = [pair[1] for pair in ci_pairs]
    ordered = ordered.iloc[::-1]
    y = np.arange(len(ordered))
    left_err = np.maximum(0.0, ordered["ATE"].to_numpy(dtype=float) - ordered["CI Low"].to_numpy(dtype=float))
    right_err = np.maximum(0.0, ordered["CI High"].to_numpy(dtype=float) - ordered["ATE"].to_numpy(dtype=float))
    ax.errorbar(
        ordered["ATE"],
        y,
        xerr=[left_err, right_err],
        fmt="o",
        color=NATURE_COLORS["purple"],
        ecolor=NATURE_COLORS["slate"],
        elinewidth=1.4,
        capsize=3,
    )
    ax.axvline(0, color=NATURE_COLORS["slate"], linestyle="--", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["Feature"])
    ax.set_xlabel("Effect size with 95% CI")
    ax.set_title("Causal Effect Forest Plot")
    return fig_to_base64(fig, fmt, dpi)


def _build_dose_response_plot(df, treatment, target, adjusted_effect, fmt="png", dpi=300):
    apply_nature_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plot_df = df[[treatment, target]].copy().sort_values(treatment)
    sample_df = plot_df.sample(min(len(plot_df), 800), random_state=42) if len(plot_df) > 800 else plot_df
    ax.scatter(sample_df[treatment], sample_df[target], s=28, alpha=0.32, color=NATURE_COLORS["gray"], edgecolors="none")

    bins = min(12, max(5, plot_df[treatment].nunique()))
    plot_df["bin"] = pd.qcut(plot_df[treatment], q=bins, duplicates="drop")
    grouped = plot_df.groupby("bin", observed=False).agg(
        treatment_mean=(treatment, "mean"),
        outcome_mean=(target, "mean"),
        outcome_std=(target, "std"),
        n=(target, "size"),
    )
    ci = 1.96 * grouped["outcome_std"].fillna(0) / np.sqrt(grouped["n"].clip(lower=1))
    ax.plot(grouped["treatment_mean"], grouped["outcome_mean"], color=NATURE_COLORS["red"], linewidth=2.2, marker="o")
    ax.fill_between(
        grouped["treatment_mean"],
        grouped["outcome_mean"] - ci,
        grouped["outcome_mean"] + ci,
        color=NATURE_COLORS["red"],
        alpha=0.16,
    )

    line_x = np.linspace(plot_df[treatment].min(), plot_df[treatment].max(), 100)
    line_y = grouped["outcome_mean"].mean() + adjusted_effect * (line_x - plot_df[treatment].mean())
    ax.plot(line_x, line_y, linestyle="--", color=NATURE_COLORS["blue"], linewidth=1.8, label="Adjusted linear effect")
    ax.set_xlabel(treatment)
    ax.set_ylabel(target)
    ax.set_title(f"Dose-Response View: {treatment} -> {target}")
    ax.legend()
    return fig_to_base64(fig, fmt, dpi)


def _build_estimator_heatmap(ranking_df, fmt="png", dpi=300):
    apply_nature_style()
    cols = [c for c in ["Naive", "Adjusted", "DML", "DoWhy", "ATE"] if c in ranking_df.columns]
    heat_df = ranking_df.set_index("Feature")[cols].fillna(0.0).head(15)
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.42 * len(heat_df) + 2)))
    im = ax.imshow(heat_df.values, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(heat_df.columns)))
    ax.set_xticklabels(heat_df.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(heat_df.index)))
    ax.set_yticklabels(heat_df.index)
    ax.set_title("Estimator Comparison Heatmap")
    fig.colorbar(im, ax=ax, shrink=0.85)
    return fig_to_base64(fig, fmt, dpi)


def _estimate_causal_bundle(df, target_col, treatment_col=None):
    numeric_df = _safe_numeric_df(df, target_col)
    if target_col not in numeric_df.columns or len(numeric_df) < 20:
        raise ValueError("Not enough clean rows for causal estimation.")

    candidate_treatments = [c for c in numeric_df.columns if c != target_col]
    if not candidate_treatments:
        raise ValueError("No valid treatment features available.")

    active_treatments = [treatment_col] if treatment_col else candidate_treatments[:20]
    active_treatments = [t for t in active_treatments if t in candidate_treatments]
    rows = []
    estimand_text = "Fallback estimand: E[Y | do(T=t+1)] - E[Y | do(T=t)], approximated with adjusted regression and orthogonalized residualization."
    methods_used = []
    if dowhy is not None:
        methods_used.append("DoWhy backdoor regression")
    methods_used.extend(["Adjusted Ridge", "Partial Linear DML"])

    for treat in active_treatments:
        local_df = numeric_df[[target_col] + [c for c in candidate_treatments if c == treat or c != target_col]].copy()
        local_df = local_df.dropna()
        if len(local_df) < 20 or local_df[treat].nunique() < 3:
            continue

        naive = _naive_effect_single(local_df, treat, target_col)
        adjusted, ci_low, ci_high = _adjusted_effect_single(local_df, treat, target_col)
        dml_out = _partial_linear_dml_single(local_df, treat, target_col)
        dml = dml_out[0] if dml_out is not None else np.nan
        dml_low = dml_out[1] if dml_out is not None else np.nan
        dml_high = dml_out[2] if dml_out is not None else np.nan

        dowhy_out = None
        if dowhy is not None:
            try:
                dowhy_out = _dowhy_effect_single(local_df, treat, target_col)
                estimand_text = dowhy_out["estimand"]
            except Exception:
                dowhy_out = None

        combined_candidates = [adjusted, dml]
        if dowhy_out is not None:
            combined_candidates.append(dowhy_out["value"])
        combined = float(np.nanmean([x for x in combined_candidates if pd.notna(x)]))
        ci_low_final, ci_high_final = _normalize_ci(
            combined,
            ci_low if ci_low is not None else (dml_low if pd.notna(dml_low) else combined),
            ci_high if ci_high is not None else (dml_high if pd.notna(dml_high) else combined),
        )
        row = {
            "Feature": treat,
            "ATE": combined,
            "Naive": float(naive),
            "Adjusted": float(adjusted),
            "DML": float(dml) if pd.notna(dml) else np.nan,
            "DoWhy": float(dowhy_out["value"]) if dowhy_out is not None else np.nan,
            "CI Low": float(ci_low_final),
            "CI High": float(ci_high_final),
            "N": int(len(local_df)),
        }
        rows.append(row)

    if not rows:
        raise ValueError("Causal estimation returned no usable treatment effects.")

    ranking_df = pd.DataFrame(rows)
    ranking_df["AbsATE"] = ranking_df["ATE"].abs()
    ranking_df = ranking_df.sort_values("AbsATE", ascending=False).reset_index(drop=True)
    return numeric_df, ranking_df, estimand_text, methods_used


def _plot_extended_shap_interpretation(X_sample, shap_matrix, mean_abs, model, fmt="png", dpi=300):
    plots = {}
    X_sample = _as_frame(X_sample).reset_index(drop=True)
    shap_df = pd.DataFrame(shap_matrix, columns=X_sample.columns)
    abs_df = shap_df.abs()
    top_features = list(np.asarray(X_sample.columns)[np.argsort(mean_abs)[::-1][: min(12, X_sample.shape[1])]])
    if not top_features:
        return plots

    pos_mean = shap_df[top_features].clip(lower=0).mean()
    neg_mean = shap_df[top_features].clip(upper=0).mean()
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.35 * len(top_features) + 1.6)))
    y = np.arange(len(top_features))
    ax.barh(y, pos_mean.values, color=NATURE_COLORS["red"], label="Positive")
    ax.barh(y, neg_mean.values, color=NATURE_COLORS["blue"], label="Negative")
    ax.set_yticks(y)
    ax.set_yticklabels(top_features)
    ax.axvline(0, color=NATURE_COLORS["slate"], linestyle="--")
    ax.set_xlabel("Mean signed SHAP contribution")
    ax.set_title("SHAP Directional Importance")
    ax.legend()
    plots["SHAP Directional Importance"] = fig_to_base64(fig, fmt, dpi)

    dominance = (np.abs(pos_mean) - np.abs(neg_mean)).sort_values()
    fig, ax = plt.subplots(figsize=(8.2, max(4.5, 0.35 * len(dominance) + 1.6)))
    ax.barh(dominance.index, dominance.values, color=[NATURE_COLORS["red"] if v > 0 else NATURE_COLORS["blue"] for v in dominance.values])
    ax.axvline(0, color=NATURE_COLORS["slate"], linestyle="--")
    ax.set_xlabel("Positive strength - negative strength")
    ax.set_title("SHAP Positive-Negative Dominance")
    plots["SHAP Positive-Negative Dominance"] = fig_to_base64(fig, fmt, dpi)

    var_df = pd.DataFrame({"Feature": top_features, "SHAP variance": shap_df[top_features].var().values}).sort_values("SHAP variance")
    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.35 * len(var_df) + 1.6)))
    ax.barh(var_df["Feature"], var_df["SHAP variance"], color=NATURE_COLORS["purple"])
    ax.set_xlabel("Variance of SHAP values")
    ax.set_title("SHAP Contribution Variance")
    plots["SHAP Contribution Variance"] = fig_to_base64(fig, fmt, dpi)

    share = abs_df[top_features].div(abs_df[top_features].sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    entropy = -(share * np.log(share + 1e-12)).sum(axis=1) / np.log(max(2, len(top_features)))
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    sns.histplot(entropy, bins=22, kde=True, color=NATURE_COLORS["teal"], ax=ax)
    ax.set_xlabel("Normalized attribution entropy")
    ax.set_title("SHAP Attribution Entropy Distribution")
    plots["SHAP Attribution Entropy Distribution"] = fig_to_base64(fig, fmt, dpi)

    driver = abs_df[top_features].idxmax(axis=1)
    counts = driver.value_counts().reindex(top_features).fillna(0).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.35 * len(counts) + 1.6)))
    ax.barh(counts.index, counts.values, color=NATURE_COLORS["orange"])
    ax.set_xlabel("Dominated sample count")
    ax.set_title("SHAP Top Driver Frequency")
    plots["SHAP Top Driver Frequency"] = fig_to_base64(fig, fmt, dpi)

    try:
        pred = _model_output(model, X_sample)
        nq = min(4, pd.Series(pred).nunique())
        if nq >= 2:
            cohorts = pd.qcut(pd.Series(pred), q=nq, labels=[f"Q{i+1}" for i in range(nq)], duplicates="drop")
            cohort_mean = shap_df[top_features].groupby(cohorts, observed=False).mean()
            fig, ax = plt.subplots(figsize=(9, max(4.6, 0.55 * len(cohort_mean) + 2.2)))
            sns.heatmap(cohort_mean, cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "Mean SHAP"})
            ax.set_xlabel("Feature")
            ax.set_ylabel("Prediction cohort")
            ax.set_title("SHAP Prediction-Cohort Profile")
            plots["SHAP Prediction-Cohort Profile"] = fig_to_base64(fig, fmt, dpi)
    except Exception:
        pass

    if len(top_features) >= 2:
        corr = shap_df[top_features].corr()
        fig, ax = plt.subplots(figsize=(8.5, 7))
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
        ax.set_title("SHAP Attribution Correlation Network")
        plots["SHAP Attribution Correlation Network"] = fig_to_base64(fig, fmt, dpi)

    strength = abs_df[top_features].sum(axis=1).sort_values(ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(np.arange(1, len(strength) + 1), strength.cumsum() / (strength.sum() + 1e-9), color=NATURE_COLORS["red"], linewidth=2)
    ax.axhline(0.8, color=NATURE_COLORS["slate"], linestyle="--")
    ax.set_xlabel("Samples sorted by total |SHAP|")
    ax.set_ylabel("Cumulative attribution strength")
    ax.set_title("SHAP Sample Influence Concentration")
    plots["SHAP Sample Influence Concentration"] = fig_to_base64(fig, fmt, dpi)

    sample_idx = int(np.argmax(abs_df[top_features].sum(axis=1).to_numpy()))
    local = shap_df.loc[sample_idx, top_features].sort_values()
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.34 * len(local) + 1.6)))
    ax.barh(local.index, local.values, color=[NATURE_COLORS["red"] if v > 0 else NATURE_COLORS["blue"] for v in local.values])
    ax.axvline(0, color=NATURE_COLORS["slate"], linestyle="--")
    ax.set_xlabel("Local SHAP contribution")
    ax.set_title("Highest-Influence Sample Decomposition")
    plots["Highest-Influence Sample Decomposition"] = fig_to_base64(fig, fmt, dpi)

    try:
        rows = []
        for feature in top_features[: min(10, len(top_features))]:
            vals = pd.to_numeric(X_sample[feature], errors="coerce")
            if vals.nunique(dropna=True) < 3:
                continue
            bins = pd.qcut(vals, q=min(6, vals.nunique()), labels=False, duplicates="drop")
            tmp = pd.DataFrame({"bin": bins, "shap": shap_df[feature]}).dropna()
            prof = tmp.groupby("bin", observed=False)["shap"].mean()
            for b, val in prof.items():
                rows.append({"Feature": feature, "Value bin": f"B{int(b)+1}", "Mean SHAP": val})
        mat = pd.DataFrame(rows).pivot(index="Feature", columns="Value bin", values="Mean SHAP")
        if not mat.empty:
            fig, ax = plt.subplots(figsize=(8.8, max(4.8, 0.38 * len(mat) + 2)))
            sns.heatmap(mat, cmap="coolwarm", center=0, annot=True, fmt=".2g", ax=ax)
            ax.set_title("SHAP Value-Bin Effect Matrix")
            plots["SHAP Value-Bin Effect Matrix"] = fig_to_base64(fig, fmt, dpi)
    except Exception:
        pass

    return plots


def _plot_extended_response_interpretation(model, X_train, feature, feature2=None, fmt="png", dpi=300):
    plots = {}
    X_train = _as_frame(X_train).copy()
    feature = str(feature)
    if feature not in X_train.columns:
        return plots
    x = pd.to_numeric(X_train[feature], errors="coerce")
    valid = x.notna()
    X_valid = X_train.loc[valid].copy()
    x = x.loc[valid]
    if len(X_valid) < 12 or x.nunique() < 4:
        return plots
    grid = np.unique(np.quantile(x, np.linspace(0.02, 0.98, 35)))
    template = X_valid.iloc[[0]].copy()
    for col in X_valid.columns:
        if pd.api.types.is_numeric_dtype(X_valid[col]):
            template[col] = X_valid[col].median()
        else:
            template[col] = X_valid[col].mode().iloc[0] if not X_valid[col].mode().empty else X_valid[col].iloc[0]
    response = []
    for value in grid:
        row = template.copy()
        row[feature] = value
        response.append(float(_model_output(model, row)[0]))
    response = np.asarray(response)
    derivative = np.gradient(response, grid)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(grid, derivative, color=NATURE_COLORS["purple"], linewidth=2.2)
    ax.axhline(0, color=NATURE_COLORS["slate"], linestyle="--")
    ax.set_xlabel(feature)
    ax.set_ylabel("Local derivative")
    ax.set_title("Marginal Response Derivative")
    plots["Marginal Response Derivative"] = fig_to_base64(fig, fmt, dpi)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot(grid, response, color=NATURE_COLORS["blue"], linewidth=2.2)
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel("Prediction")
    axes[0].set_title("Reference-Point Response Curve")
    signs = np.sign(derivative)
    axes[1].bar(["Positive", "Flat", "Negative"], [np.mean(signs > 0), np.mean(signs == 0), np.mean(signs < 0)], color=[NATURE_COLORS["red"], NATURE_COLORS["gray"], NATURE_COLORS["blue"]])
    axes[1].set_ylabel("Fraction of grid")
    axes[1].set_title("Response Monotonicity Balance")
    plots["Response Monotonicity Diagnostics"] = fig_to_base64(fig, fmt, dpi)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    anchors = X_valid.select_dtypes(include=[np.number]).quantile(np.linspace(0.1, 0.9, 5))
    for q in anchors.index:
        row = template.copy()
        for col in anchors.columns:
            row[col] = anchors.loc[q, col]
        preds = []
        for value in grid:
            local = row.copy()
            local[feature] = value
            preds.append(float(_model_output(model, local)[0]))
        ax.plot(grid, preds, linewidth=1.7, label=f"Anchor Q{int(q*100)}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Prediction")
    ax.set_title("Quantile-Anchored Response Curves")
    ax.legend(fontsize=8)
    plots["Quantile-Anchored Response Curves"] = fig_to_base64(fig, fmt, dpi)

    density, edges = np.histogram(x, bins=min(18, max(6, int(np.sqrt(len(x))))), density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    interp_density = np.interp(grid, centers, density, left=0, right=0)
    risk = 1 - interp_density / (interp_density.max() + 1e-9)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(grid, risk, color=NATURE_COLORS["red"], linewidth=2.2)
    ax.fill_between(grid, 0, risk, color=NATURE_COLORS["red"], alpha=0.12)
    ax.set_xlabel(feature)
    ax.set_ylabel("Low-density extrapolation risk")
    ax.set_title("Response Extrapolation Risk")
    plots["Response Extrapolation Risk"] = fig_to_base64(fig, fmt, dpi)

    levels = np.quantile(response, [0.1, 0.25, 0.5, 0.75, 0.9])
    required = [grid[int(np.argmin(np.abs(response - level)))] for level in levels]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(levels, required, marker="o", color=NATURE_COLORS["teal"], linewidth=2.2)
    ax.set_xlabel("Target prediction level")
    ax.set_ylabel(f"Required {feature}")
    ax.set_title("Counterfactual Threshold Path")
    plots["Counterfactual Threshold Path"] = fig_to_base64(fig, fmt, dpi)

    if feature2 and str(feature2) != "None" and str(feature2) in X_train.columns and str(feature2) != feature:
        f2 = str(feature2)
        y = pd.to_numeric(X_train[f2], errors="coerce")
        if y.notna().sum() >= 12 and y.nunique() >= 4:
            y_grid = np.unique(np.quantile(y.dropna(), np.linspace(0.05, 0.95, 22)))
            z = np.zeros((len(y_grid), len(grid)))
            for i, yv in enumerate(y_grid):
                batch = pd.concat([template.copy()] * len(grid), ignore_index=True)
                batch[feature] = grid
                batch[f2] = yv
                z[i, :] = _model_output(model, batch)
            fig, ax = plt.subplots(figsize=(8, 6))
            contour = ax.contourf(grid, y_grid, z, levels=18, cmap="viridis")
            ax.set_xlabel(feature)
            ax.set_ylabel(f2)
            ax.set_title("Pairwise Response Interaction Ridge")
            fig.colorbar(contour, ax=ax, label="Prediction")
            plots["Pairwise Response Interaction Ridge"] = fig_to_base64(fig, fmt, dpi)

            fig, ax = plt.subplots(figsize=(8, 5.6))
            ax.plot(grid, np.std(z, axis=0), color=NATURE_COLORS["orange"], linewidth=2.2)
            ax.set_xlabel(feature)
            ax.set_ylabel(f"Prediction SD across {f2}")
            ax.set_title("Conditional Interaction Strength Curve")
            plots["Conditional Interaction Strength Curve"] = fig_to_base64(fig, fmt, dpi)

    return plots


def _plot_extended_permutation_interpretation(result, imp_df, X_test, model, fmt="png", dpi=300):
    plots = {}
    X_test = _as_frame(X_test)
    importances = pd.DataFrame(result.importances.T, columns=X_test.columns)
    top_features = imp_df["Feature"].tolist()[: min(12, len(imp_df))]
    if not top_features:
        return plots

    dist = importances[top_features].melt(var_name="Feature", value_name="Importance")
    fig, ax = plt.subplots(figsize=(9, max(5, 0.38 * len(top_features) + 2)))
    sns.boxplot(data=dist, y="Feature", x="Importance", order=top_features, ax=ax, color="#c7dcef")
    ax.axvline(0, color=NATURE_COLORS["slate"], linestyle="--")
    ax.set_title("Permutation Repeat Distribution")
    plots["Permutation Repeat Distribution"] = fig_to_base64(fig, fmt, dpi)

    snr = imp_df.copy()
    snr["Signal-to-noise"] = snr["Importance Mean"] / (snr["Importance Std"] + 1e-9)
    plot_df = snr.head(15).sort_values("Signal-to-noise")
    fig, ax = plt.subplots(figsize=(8, max(4.8, 0.36 * len(plot_df) + 1.6)))
    ax.barh(plot_df["Feature"], plot_df["Signal-to-noise"], color=NATURE_COLORS["teal"])
    ax.axvline(0, color=NATURE_COLORS["slate"], linestyle="--")
    ax.set_xlabel("Mean importance / std")
    ax.set_title("Permutation Importance Signal-to-Noise")
    plots["Permutation Importance Signal-to-Noise"] = fig_to_base64(fig, fmt, dpi)

    ranks = importances.rank(axis=1, ascending=False)
    rank_df = ranks[top_features].agg(["mean", "std"]).T.reset_index().rename(columns={"index": "Feature", "mean": "Mean rank", "std": "Rank std"})
    fig, ax = plt.subplots(figsize=(8, max(4.8, 0.36 * len(rank_df) + 1.6)))
    ax.errorbar(rank_df["Mean rank"], np.arange(len(rank_df)), xerr=rank_df["Rank std"], fmt="o", color=NATURE_COLORS["purple"], capsize=3)
    ax.set_yticks(np.arange(len(rank_df)))
    ax.set_yticklabels(rank_df["Feature"])
    ax.invert_xaxis()
    ax.set_xlabel("Permutation rank (lower is better)")
    ax.set_title("Permutation Rank Stability")
    plots["Permutation Rank Stability"] = fig_to_base64(fig, fmt, dpi)

    numeric = X_test[top_features].apply(pd.to_numeric, errors="coerce")
    if numeric.shape[1] >= 2:
        corr = numeric.corr().abs()
        imp_share = imp_df.set_index("Feature").loc[top_features, "Importance Mean"].clip(lower=0)
        redundancy = corr.mul(imp_share / (imp_share.max() + 1e-9), axis=0)
        fig, ax = plt.subplots(figsize=(8.2, 6.8))
        sns.heatmap(redundancy, cmap="rocket_r", ax=ax, cbar_kws={"label": "Correlation-weighted importance"})
        ax.set_title("Permutation Redundancy Proxy Map")
        plots["Permutation Redundancy Proxy Map"] = fig_to_base64(fig, fmt, dpi)

    try:
        baseline_pred = _model_output(model, X_test)
        baseline_var = float(np.var(baseline_pred))
        rows = []
        for k in range(1, min(12, len(top_features)) + 1):
            keep = top_features[:k]
            X_mask = X_test.copy()
            for col in X_mask.columns:
                if col not in keep:
                    X_mask[col] = X_mask[col].sample(frac=1, random_state=40 + k).to_numpy()
            pred = _model_output(model, X_mask)
            rows.append({"Top-k": k, "Prediction variance retained": float(np.var(pred) / (baseline_var + 1e-9))})
        suff = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        ax.plot(suff["Top-k"], suff["Prediction variance retained"], marker="o", color=NATURE_COLORS["red"], linewidth=2)
        ax.axhline(0.8, color=NATURE_COLORS["slate"], linestyle="--")
        ax.set_xlabel("Top-k permutation features retained")
        ax.set_ylabel("Prediction variance retained")
        ax.set_title("Top-k Feature Sufficiency Proxy")
        plots["Top-k Feature Sufficiency Proxy"] = fig_to_base64(fig, fmt, dpi)
    except Exception:
        pass

    return plots


def get_shap_plots(model, X_train, sample_idx=0, fmt="png", dpi=300):
    if shap is None:
        raise ImportError("SHAP is not installed. Please install shap to generate explanation plots.")
    apply_nature_style()
    X_train = _as_frame(X_train)
    sample_size = min(200, len(X_train))
    X_sample = X_train.sample(sample_size, random_state=42)
    plots = {}

    try:
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
        shap_matrix = _extract_shap_matrix(shap_values)
        expected_value = explainer.expected_value
    except Exception:
        explainer = shap.KernelExplainer(model.predict, shap.kmeans(X_sample, min(10, len(X_sample))))
        raw_values = explainer.shap_values(X_sample)
        shap_matrix = _extract_shap_matrix(raw_values)
        shap_values = raw_values
        expected_value = explainer.expected_value

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_matrix, X_sample, show=False)
    plt.title("SHAP Global Feature Impact")
    plt.tight_layout()
    plots["Global Summary"] = fig_to_base64(plt.gcf(), fmt, dpi)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_matrix, X_sample, plot_type="bar", show=False)
    plt.title("SHAP Absolute Importance")
    plt.tight_layout()
    plots["Global Bar Importance"] = fig_to_base64(plt.gcf(), fmt, dpi)

    mean_abs = np.abs(shap_matrix).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(X_sample.columns[top_idx][::-1], mean_abs[top_idx][::-1], color=NATURE_COLORS["purple"])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top Feature Attribution Ranking")
    plots["Attribution Ranking"] = fig_to_base64(fig, fmt, dpi)

    dependence_panel = _plot_shap_dependence_panel(X_sample, shap_matrix, mean_abs, fmt, dpi)
    if dependence_panel:
        plots["SHAP Dependence Panel"] = dependence_panel
    cohort_heatmap = _plot_shap_cohort_heatmap(X_sample, shap_matrix, mean_abs, fmt, dpi)
    if cohort_heatmap:
        plots["SHAP Cohort Heatmap"] = cohort_heatmap
    concentration_curve = _plot_shap_concentration_curve(mean_abs, X_sample.columns.to_numpy(), fmt, dpi)
    if concentration_curve:
        plots["SHAP Attribution Concentration"] = concentration_curve

    sample_idx = min(sample_idx, len(X_sample) - 1)
    try:
        plots.update(_plot_designed_xai_suite(X_sample, shap_matrix, mean_abs, model, expected_value, sample_idx, fmt, dpi))
    except Exception:
        pass
    try:
        plots.update(_plot_intuitive_xai_suite(X_sample, shap_matrix, mean_abs, model, expected_value, sample_idx, fmt, dpi))
    except Exception:
        pass
    try:
        shap_obj = shap_values[sample_idx] if hasattr(shap_values, "__getitem__") and not isinstance(shap_values, list) else None
        if shap_obj is not None and hasattr(shap.plots, "waterfall"):
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_obj, show=False)
            plt.title(f"SHAP Waterfall (Sample {sample_idx})")
            plt.tight_layout()
            plots["Local Waterfall Map"] = fig_to_base64(plt.gcf(), fmt, dpi)
    except Exception:
        pass

    top_feature = X_sample.columns[int(top_idx[0])] if len(top_idx) else X_sample.columns[0]
    try:
        plt.figure(figsize=(8, 5.5))
        shap.dependence_plot(top_feature, shap_matrix, X_sample, show=False)
        plt.title(f"SHAP Dependence: {top_feature}")
        plt.tight_layout()
        plots["Top-Feature Dependence"] = fig_to_base64(plt.gcf(), fmt, dpi)
    except Exception:
        pass

    try:
        base_val = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) and np.asarray(expected_value).ndim > 0 else expected_value
        decision_slice = shap_matrix[: min(40, len(X_sample))]
        feature_slice = X_sample.iloc[: min(40, len(X_sample))]
        plt.figure(figsize=(10, 6))
        shap.decision_plot(base_val, decision_slice, feature_slice, show=False)
        plt.title("SHAP Decision Plot")
        plt.tight_layout()
        plots["Decision Plot"] = fig_to_base64(plt.gcf(), fmt, dpi)
    except Exception:
        pass

    try:
        top_features = X_sample.columns[np.argsort(mean_abs)[::-1][: min(10, len(X_sample.columns))]]
        interaction_proxy = pd.DataFrame(shap_matrix[:, [X_sample.columns.get_loc(col) for col in top_features]], columns=top_features).corr()
        fig, ax = plt.subplots(figsize=(7.8, 6.2))
        sns.heatmap(interaction_proxy, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
        ax.set_title("SHAP Interaction Proxy Heatmap")
        plots["SHAP Interaction Proxy Heatmap"] = fig_to_base64(fig, fmt, dpi)
    except Exception:
        pass

    try:
        plots.update(_plot_extended_shap_interpretation(X_sample, shap_matrix, mean_abs, model, fmt, dpi))
    except Exception:
        pass

    return plots


def get_pdp_plot(model, X_train, feature, feature2=None, feature3=None, fmt="png", dpi=300):
    apply_nature_style()
    plots = {}
    X_train = _as_frame(X_train)

    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        PartialDependenceDisplay.from_estimator(model, X_train, features=[str(feature)], kind="both", ax=ax, subsample=80)
        plt.title(f"PDP + ICE: {feature}")
        plt.tight_layout()
        plots["2D PDP & ICE"] = fig_to_base64(fig, fmt, dpi)
    except Exception as e:
        plots["2D Error"] = f"Failed 2D PDP: {str(e)}"

    try:
        ale_x, ale_y, ale_low, ale_high = _ale_curve(model, X_train, str(feature))
        fig, ax = plt.subplots(figsize=(8, 5.4))
        ax.plot(ale_x, ale_y, color=NATURE_COLORS["teal"], linewidth=2.3, label="ALE")
        ax.fill_between(ale_x, ale_low, ale_high, color=NATURE_COLORS["teal"], alpha=0.18, label="95% bootstrap band")
        ax.axhline(0, color=NATURE_COLORS["slate"], linestyle="--", linewidth=1.1)
        ax.set_xlabel(str(feature))
        ax.set_ylabel("Accumulated local effect")
        ax.set_title(f"ALE Main Effect: {feature}")
        ax.legend()
        plots["ALE Main Effect"] = fig_to_base64(fig, fmt, dpi)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        feature_values = pd.to_numeric(X_train[str(feature)], errors="coerce")
        ax.hist(feature_values.dropna(), bins=min(20, max(8, int(np.sqrt(len(feature_values.dropna()))))), color=NATURE_COLORS["gray"], alpha=0.35, density=True)
        ax2 = ax.twinx()
        ax2.plot(ale_x, ale_y, color=NATURE_COLORS["red"], linewidth=2.1)
        ax2.fill_between(ale_x, ale_low, ale_high, color=NATURE_COLORS["red"], alpha=0.12)
        ax.set_xlabel(str(feature))
        ax.set_ylabel("Feature density")
        ax2.set_ylabel("ALE")
        ax.set_title(f"ALE with Feature Density: {feature}")
        plots["ALE Density Overlay"] = fig_to_base64(fig, fmt, dpi)
    except Exception as e:
        plots["ALE Error"] = f"Failed ALE: {str(e)}"

    try:
        ice_heterogeneity = _plot_ice_heterogeneity(model, X_train, str(feature), fmt, dpi)
        if ice_heterogeneity:
            plots["ICE Heterogeneity and Sensitivity"] = ice_heterogeneity
    except Exception:
        pass

    try:
        plots.update(_plot_extended_response_interpretation(model, X_train, str(feature), feature2, fmt, dpi))
    except Exception:
        pass

    if feature2 and str(feature2) != "None" and str(feature) != str(feature2):
        try:
            pdp_res = partial_dependence(model, X_train, features=[str(feature), str(feature2)], grid_resolution=35)
            if hasattr(pdp_res, "average"):
                pdp_vals = pdp_res.average[0] if pdp_res.average.ndim == 3 else pdp_res.average
                grid_axes = pdp_res["grid_values"] if "grid_values" in pdp_res else pdp_res["values"]
            else:
                pdp_vals, grid_axes = pdp_res[0], pdp_res[1]
                if np.asarray(pdp_vals).ndim == 3:
                    pdp_vals = pdp_vals[0]

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            X_g, Y_g = np.meshgrid(grid_axes[0], grid_axes[1])
            surf = ax.plot_surface(X_g, Y_g, np.asarray(pdp_vals).T, cmap="viridis", edgecolor="none", alpha=0.92)
            ax.set_xlabel(str(feature))
            ax.set_ylabel(str(feature2))
            ax.set_zlabel("Partial dependence")
            ax.set_title(f"3D Surface PDP: {feature} vs {feature2}")
            fig.colorbar(surf, ax=ax, shrink=0.55, aspect=10)
            plots["3D Surface PDP"] = fig_to_base64(fig, fmt, dpi)

            fig, ax = plt.subplots(figsize=(8, 6))
            contour = ax.contourf(X_g, Y_g, np.asarray(pdp_vals).T, levels=18, cmap="viridis")
            ax.set_xlabel(str(feature))
            ax.set_ylabel(str(feature2))
            ax.set_title(f"PDP Contour Map: {feature} vs {feature2}")
            fig.colorbar(contour, ax=ax, label="Partial dependence")
            plots["PDP Contour Map"] = fig_to_base64(fig, fmt, dpi)
        except Exception as e:
            plots["3D Error"] = f"Failed 3D Surface: {str(e)}"

    if feature3 and str(feature3) != "None" and len({str(feature), str(feature2), str(feature3)}) == 3:
        try:
            sample_df = X_train.sample(min(500, len(X_train)), random_state=42).copy()
            y_pred = model.predict_proba(sample_df)[:, 1] if hasattr(model, "predict_proba") else model.predict(sample_df)

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            sc = ax.scatter(
                sample_df[str(feature)],
                sample_df[str(feature2)],
                sample_df[str(feature3)],
                c=y_pred,
                cmap="coolwarm",
                s=42,
                alpha=0.84,
                edgecolor="white",
                linewidth=0.4,
            )
            ax.set_xlabel(str(feature))
            ax.set_ylabel(str(feature2))
            ax.set_zlabel(str(feature3))
            ax.set_title(f"3-Feature Model Response: {feature}, {feature2}, {feature3}")
            cb = plt.colorbar(sc, ax=ax, shrink=0.55, aspect=10)
            cb.set_label("Prediction magnitude")
            plots["3-Feature Voyager (3D)"] = fig_to_base64(fig, fmt, dpi)
        except Exception as e:
            plots["Voyager Error"] = f"Failed Voyager: {str(e)}"

    return plots


def get_permutation_importance_plots(model, X_test, y_test, fmt="png", dpi=300):
    apply_nature_style()
    X_test = _as_frame(X_test)
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=42,
        n_jobs=1,
    )
    imp_df = pd.DataFrame(
        {
            "Feature": X_test.columns,
            "Importance Mean": result.importances_mean,
            "Importance Std": result.importances_std,
        }
    ).sort_values("Importance Mean", ascending=False)
    top_df = imp_df.head(15)
    plots = {}

    fig, ax = plt.subplots(figsize=(8.5, max(4.8, 0.42 * len(top_df) + 2)))
    ax.barh(top_df["Feature"][::-1], top_df["Importance Mean"][::-1], color=NATURE_COLORS["orange"])
    ax.set_xlabel("Mean permutation importance")
    ax.set_title("Permutation Importance Ranking")
    plots["Permutation Importance Ranking"] = fig_to_base64(fig, fmt, dpi)

    fig, ax = plt.subplots(figsize=(9, max(5.2, 0.45 * len(top_df) + 2)))
    y = np.arange(len(top_df))
    ordered = top_df.iloc[::-1]
    ax.errorbar(
        ordered["Importance Mean"],
        y,
        xerr=ordered["Importance Std"],
        fmt="o",
        color=NATURE_COLORS["teal"],
        ecolor=NATURE_COLORS["slate"],
        capsize=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["Feature"])
    ax.axvline(0, color=NATURE_COLORS["slate"], linestyle="--", linewidth=1.1)
    ax.set_xlabel("Importance mean +/- std")
    ax.set_title("Permutation Importance Stability")
    plots["Permutation Importance Stability"] = fig_to_base64(fig, fmt, dpi)

    cumulative = top_df["Importance Mean"].clip(lower=0).cumsum()
    total = cumulative.iloc[-1] if len(cumulative) else 1.0
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(range(1, len(top_df) + 1), cumulative / max(total, 1e-9), marker="o", color=NATURE_COLORS["purple"], linewidth=2)
    ax.set_xlabel("Top-ranked features included")
    ax.set_ylabel("Cumulative contribution share")
    ax.set_title("Permutation Cumulative Drop Curve")
    ax.set_ylim(0, 1.05)
    plots["Permutation Cumulative Drop Curve"] = fig_to_base64(fig, fmt, dpi)

    try:
        plots.update(_plot_extended_permutation_interpretation(result, imp_df, X_test, model, fmt, dpi))
    except Exception:
        pass

    return plots


def get_advanced_xai_atlas(model, X_train, y_train=None, fmt="png", dpi=300):
    if shap is None:
        raise ImportError("SHAP is required for the Advanced XAI Atlas.")
    apply_nature_style()
    plots = {}
    X_train = _as_frame(X_train).copy()
    sample_size = min(220, len(X_train))
    X_sample = X_train.sample(sample_size, random_state=42).reset_index(drop=True)

    try:
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
        shap_matrix = _extract_shap_matrix(shap_values)
    except Exception:
        explainer = shap.KernelExplainer(model.predict, shap.kmeans(X_sample, min(10, len(X_sample))))
        shap_matrix = _extract_shap_matrix(explainer.shap_values(X_sample))

    feature_names = np.asarray(X_sample.columns)
    mean_abs = np.abs(shap_matrix).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][: min(10, len(feature_names))]
    top_features = feature_names[top_idx]
    shap_top = pd.DataFrame(shap_matrix[:, top_idx], columns=top_features)
    pred = _model_output(model, X_sample)
    total_strength = np.abs(shap_top).sum(axis=1).to_numpy()

    # 1. Explanation manifold.
    coords = PCA(n_components=2, random_state=42).fit_transform(shap_matrix)
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=pred, cmap="viridis", s=42, alpha=0.82, edgecolors="white", linewidth=0.35)
    fig.colorbar(sc, ax=ax, label="Prediction")
    ax.set_xlabel("Explanation PC1")
    ax.set_ylabel("Explanation PC2")
    ax.set_title("XAI Explanation Manifold")
    plots["XAI Explanation Manifold"] = fig_to_base64(fig, fmt, dpi)

    # 2. Explanation manifold by top driver.
    drivers = shap_top.abs().idxmax(axis=1)
    driver_codes, driver_labels = pd.factorize(drivers)
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    cmap = plt.get_cmap("tab10")
    for code, label in enumerate(driver_labels):
        mask = driver_codes == code
        ax.scatter(coords[mask, 0], coords[mask, 1], s=38, alpha=0.78, color=cmap(code % 10), label=str(label), edgecolors="white", linewidth=0.25)
    ax.set_xlabel("Explanation PC1")
    ax.set_ylabel("Explanation PC2")
    ax.set_title("XAI Manifold by Dominant Feature")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    plots["XAI Manifold by Dominant Feature"] = fig_to_base64(fig, fmt, dpi)

    # 3. SHAP river sorted by prediction.
    order = np.argsort(pred)
    positive_parts = shap_top.clip(lower=0).iloc[order]
    negative_parts = -shap_top.clip(upper=0).iloc[order]
    x = np.arange(len(order))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    axes[0].stackplot(x, positive_parts.T, labels=top_features, alpha=0.86)
    axes[0].set_ylabel("Positive SHAP")
    axes[0].set_title("SHAP Contribution River")
    axes[1].stackplot(x, negative_parts.T, labels=top_features, alpha=0.86)
    axes[1].set_ylabel("Negative |SHAP|")
    axes[1].set_xlabel("Samples sorted by prediction")
    axes[0].legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.34), frameon=False)
    plots["SHAP Contribution River"] = fig_to_base64(fig, fmt, dpi)

    # 4. Polar attribution rose.
    rose_vals = mean_abs[top_idx] / (mean_abs[top_idx].sum() + 1e-9)
    theta = np.linspace(0, 2 * np.pi, len(top_features), endpoint=False)
    fig = plt.figure(figsize=(6.6, 6.4))
    ax = fig.add_subplot(111, projection="polar")
    bars = ax.bar(theta, rose_vals, width=2 * np.pi / len(top_features) * 0.86, color=plt.get_cmap("turbo")(np.linspace(0.05, 0.9, len(top_features))), alpha=0.88)
    ax.set_xticks(theta)
    ax.set_xticklabels(top_features, fontsize=8)
    ax.set_title("SHAP Polar Attribution Rose", pad=24)
    plots["SHAP Polar Attribution Rose"] = fig_to_base64(fig, fmt, dpi)

    # 5. Explanation similarity heatmap.
    subset = min(80, len(shap_matrix))
    mat = shap_matrix[:subset]
    norm = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    sim = (mat / norm) @ (mat / norm).T
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    sns.heatmap(sim, cmap="mako", vmin=-1, vmax=1, ax=ax, xticklabels=False, yticklabels=False, cbar_kws={"label": "Cosine similarity"})
    ax.set_title("Local Explanation Similarity Map")
    plots["Local Explanation Similarity Map"] = fig_to_base64(fig, fmt, dpi)

    # 6. Explanation clustering heatmap.
    n_clusters = min(5, max(2, len(X_sample) // 30))
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(shap_top)
    clustered = shap_top.copy()
    clustered["cluster"] = labels
    clustered["strength"] = total_strength
    clustered = clustered.sort_values(["cluster", "strength"], ascending=[True, False]).drop(columns=["cluster", "strength"])
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    sns.heatmap(clustered.iloc[:120], cmap="coolwarm", center=0, ax=ax, xticklabels=True, yticklabels=False, cbar_kws={"label": "SHAP"})
    ax.set_title("Explanation Cluster Heatmap")
    plots["Explanation Cluster Heatmap"] = fig_to_base64(fig, fmt, dpi)

    # 7. Cluster signature radar.
    cluster_mean = pd.DataFrame(np.abs(shap_top), columns=top_features).groupby(labels).mean()
    angles = np.linspace(0, 2 * np.pi, len(top_features), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(7, 6.6))
    ax = fig.add_subplot(111, polar=True)
    for cluster_id, row in cluster_mean.iterrows():
        vals = (row / (row.max() + 1e-9)).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.8, label=f"C{cluster_id}")
        ax.fill(angles, vals, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top_features, fontsize=8)
    ax.set_title("Explanation Cluster Signature Radar")
    ax.legend(fontsize=8, frameon=False)
    plots["Explanation Cluster Signature Radar"] = fig_to_base64(fig, fmt, dpi)

    # 8. Prototype and criticism map.
    center = coords.mean(axis=0)
    dist = np.linalg.norm(coords - center, axis=1)
    prototypes = np.argsort(dist)[: min(8, len(dist))]
    criticisms = np.argsort(dist)[-min(8, len(dist)):]
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    ax.scatter(coords[:, 0], coords[:, 1], c=total_strength, cmap="rocket_r", s=32, alpha=0.65)
    ax.scatter(coords[prototypes, 0], coords[prototypes, 1], marker="o", s=120, facecolors="none", edgecolors=NATURE_COLORS["teal"], linewidth=2, label="Prototypes")
    ax.scatter(coords[criticisms, 0], coords[criticisms, 1], marker="X", s=110, color=NATURE_COLORS["red"], label="Criticisms")
    ax.set_xlabel("Explanation PC1")
    ax.set_ylabel("Explanation PC2")
    ax.set_title("Prototype-Criticism Explanation Map")
    ax.legend()
    plots["Prototype-Criticism Explanation Map"] = fig_to_base64(fig, fmt, dpi)

    # 9. Bootstrap attribution stability.
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(80):
        idx = rng.integers(0, len(shap_top), len(shap_top))
        boot.append(np.abs(shap_top.iloc[idx]).mean(axis=0).to_numpy())
    boot = np.asarray(boot)
    lower, middle, upper = np.quantile(boot, [0.025, 0.5, 0.975], axis=0)
    stable = pd.DataFrame({"Feature": top_features, "median": middle, "low": lower, "high": upper}).sort_values("median")
    fig, ax = plt.subplots(figsize=(8.2, max(4.6, 0.38 * len(stable) + 1.6)))
    y = np.arange(len(stable))
    ax.errorbar(stable["median"], y, xerr=[stable["median"] - stable["low"], stable["high"] - stable["median"]], fmt="o", color=NATURE_COLORS["blue"], ecolor=NATURE_COLORS["slate"], capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(stable["Feature"])
    ax.set_xlabel("Bootstrap mean |SHAP|")
    ax.set_title("Bootstrap Attribution Stability")
    plots["Bootstrap Attribution Stability"] = fig_to_base64(fig, fmt, dpi)

    # 10. Surrogate rule tree.
    try:
        target = pred
        if _is_classifier(model) and hasattr(model, "classes_"):
            tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=max(3, len(X_sample) // 30), random_state=42)
        else:
            tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=max(3, len(X_sample) // 30), random_state=42)
        X_numeric = X_sample.apply(pd.to_numeric, errors="coerce").fillna(0)
        tree.fit(X_numeric, target)
        fig, ax = plt.subplots(figsize=(13, 6.8))
        plot_tree(tree, feature_names=list(X_numeric.columns), filled=True, rounded=True, impurity=False, fontsize=7, ax=ax)
        ax.set_title("Surrogate Rule Tree for Model Predictions")
        plots["Surrogate Rule Tree"] = fig_to_base64(fig, fmt, dpi)
    except Exception:
        pass

    # 11. Local waterfall gallery for representative samples.
    representative = list(prototypes[:2]) + list(criticisms[:2])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for ax, idx in zip(axes.ravel(), representative):
        vals = shap_top.iloc[idx].sort_values()
        ax.barh(vals.index, vals.values, color=[NATURE_COLORS["red"] if v > 0 else NATURE_COLORS["blue"] for v in vals.values])
        ax.axvline(0, color=NATURE_COLORS["slate"], linestyle="--")
        ax.set_title(f"Sample {idx} | pred={pred[idx]:.3g}", fontsize=10)
    fig.suptitle("Representative Local Explanation Gallery", y=1.01)
    plots["Representative Local Explanation Gallery"] = fig_to_base64(fig, fmt, dpi)

    # 12. Explanation strength vs prediction.
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.scatter(pred, total_strength, c=entropy if "entropy" in locals() else total_strength, cmap="viridis", s=42, alpha=0.78, edgecolors="white", linewidth=0.3)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Total |SHAP|")
    ax.set_title("Explanation Strength vs Prediction")
    plots["Explanation Strength vs Prediction"] = fig_to_base64(fig, fmt, dpi)

    return plots


def get_causal_effect(df, target_col, treatment_col=None, fmt="png", dpi=300):
    logging.getLogger("dowhy").setLevel(logging.CRITICAL)
    try:
        numeric_df, ranking_df, estimand_text, methods_used = _estimate_causal_bundle(df, target_col, treatment_col)

        focus_row = ranking_df.iloc[0] if treatment_col is None else ranking_df[ranking_df["Feature"] == treatment_col].iloc[0]
        ate = float(focus_row["ATE"])
        ci_low, ci_high = _normalize_ci(ate, focus_row["CI Low"], focus_row["CI High"])

        results = {
            "ate": ate,
            "ate_ci": [ci_low, ci_high],
            "ate_map": {row["Feature"]: float(row["ATE"]) for _, row in ranking_df.iterrows()},
            "estimand": estimand_text,
            "method": "Consensus estimate from adjusted regression + orthogonalized DML" + (" + DoWhy" if dowhy is not None else ""),
            "methods_used": methods_used,
            "message": f"Causal analysis completed on {len(numeric_df)} clean samples across {len(ranking_df)} treatment candidates.",
            "summary_table": ranking_df[["Feature", "ATE", "CI Low", "CI High", "Naive", "Adjusted", "DML", "DoWhy", "N"]]
            .round(4)
            .replace({np.nan: None})
            .to_dict(orient="records"),
            "graph": _build_causal_graph(ranking_df, target_col, fmt, dpi),
            "ranking": _build_causal_ranking_plot(ranking_df, fmt, dpi),
            "forest": _build_causal_forest_plot(ranking_df.head(20), fmt, dpi),
            "dose_response": _build_dose_response_plot(numeric_df, focus_row["Feature"], target_col, focus_row["Adjusted"], fmt, dpi),
            "estimator_heatmap": _build_estimator_heatmap(ranking_df, fmt, dpi),
        }
        return results
    except Exception as e:
        raise Exception(f"Causal Batch Inference Failed: {str(e)}")


def get_counterfactual_plots(model, X_train, sample_idx=0, desired_value=None, fmt="png", dpi=300):
    apply_nature_style()
    X_train = _as_frame(X_train)
    sample_idx = min(max(int(sample_idx), 0), len(X_train) - 1)
    x0 = X_train.iloc[[sample_idx]].copy()
    original_pred = float(np.ravel(model.predict(x0))[0])
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("Counterfactual search requires numeric features.")
    target_value = float(desired_value) if desired_value is not None else original_pred + np.std(np.ravel(model.predict(X_train))) * 0.75

    try:
        cf_df, method_name = _official_dice_counterfactuals(model, X_train, x0, desired_value, total_cfs=4)
    except Exception:
        cf_df, method_name = _fallback_counterfactuals(model, X_train, x0, desired_value, total_cfs=4)

    if "Prediction" not in cf_df.columns:
        cf_df["Prediction"] = np.ravel(model.predict(cf_df[X_train.columns]))
    cf_df.insert(0, "Sample", [f"CF{i+1}" for i in range(len(cf_df))])

    delta_df = pd.DataFrame(
        {
            "Feature": numeric_cols,
            "Original": x0[numeric_cols].iloc[0].to_numpy(dtype=float),
            "Counterfactual": cf_df[numeric_cols].iloc[0].to_numpy(dtype=float),
        }
    )
    delta_df["Delta"] = delta_df["Counterfactual"] - delta_df["Original"]
    delta_df = delta_df.reindex(delta_df["Delta"].abs().sort_values(ascending=False).index).head(15)

    plots = {}
    fig, ax = plt.subplots(figsize=(8.2, max(4.6, 0.34 * len(delta_df) + 2.1)))
    colors = [NATURE_COLORS["red"] if x > 0 else NATURE_COLORS["blue"] for x in delta_df["Delta"]]
    ax.barh(delta_df["Feature"][::-1], delta_df["Delta"][::-1], color=colors[::-1])
    ax.axvline(0, color=NATURE_COLORS["slate"], linewidth=1.2)
    ax.set_xlabel("Feature change")
    ax.set_title("DiCE-Style Counterfactual Feature Shifts")
    plots["Counterfactual Feature Shifts"] = fig_to_base64(fig, fmt, dpi)

    compare_df = pd.DataFrame({"Original": x0[numeric_cols].iloc[0], "Counterfactual": cf_df[numeric_cols].iloc[0]})
    fig, ax = plt.subplots(figsize=(7.6, max(4.4, 0.28 * len(compare_df.head(15)) + 2.2)))
    sns.heatmap(compare_df.head(15), cmap="coolwarm", center=0, annot=True, fmt=".3f", ax=ax)
    ax.set_title("Counterfactual Comparison Heatmap")
    plots["Counterfactual Comparison Heatmap"] = fig_to_base64(fig, fmt, dpi)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    preds = [original_pred] + cf_df["Prediction"].tolist()
    labels = ["Original"] + cf_df["Sample"].tolist()
    ax.bar(labels, preds, color=[NATURE_COLORS["slate"]] + [NATURE_COLORS["teal"]] * len(cf_df))
    ax.axhline(target_value, color=NATURE_COLORS["red"], linestyle="--", linewidth=1.5, label=f"Desired = {target_value:.3f}")
    ax.set_ylabel("Prediction")
    ax.set_title("Counterfactual Prediction Shift")
    ax.legend()
    plots["Counterfactual Prediction Shift"] = fig_to_base64(fig, fmt, dpi)

    return {
        "original_prediction": round(original_pred, 5),
        "desired_prediction": round(target_value, 5),
        "method": method_name,
        "counterfactuals": cf_df.round(5).replace({np.nan: None}).to_dict(orient="records"),
        "plots": plots,
    }
