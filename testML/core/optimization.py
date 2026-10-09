import ast
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from core.plotting_utils import NATURE_COLORS, apply_nature_style, fig_to_base64


ALGORITHM_REGISTRY = {
    "NSGA-II": "nsga2",
    "R-NSGA-II": "nsga2",
    "MOEA/D": "moead",
    "MOPSO": "mopso",
    "SMPSO": "mopso",
    "Differential Evolution MO": "de",
    "Adaptive Weighted Sum": "weighted",
    "Epsilon-Constraint Sweep": "epsilon",
    "Random Pareto Search": "random",
    "Simulated Annealing MO": "anneal",
}


COMPARE_METRIC_NAMES = [
    "Hypervolume",
    "Spacing",
    "Spread",
    "Feasible Ratio",
    "Nondominated Count",
    "Mean Ideal Distance",
    "Mean Crowding",
]


ALLOWED_AST_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.IfExp,
    ast.Subscript,
    ast.Attribute,
    ast.List,
    ast.Tuple,
}


SAFE_FUNCTIONS = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "log": np.log,
    "log1p": np.log1p,
    "exp": np.exp,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "clip": np.clip,
    "where": np.where,
    "mean": np.mean,
    "std": np.std,
}


def _validate_expression(expr):
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_AST_NODES:
            raise ValueError(f"Unsupported expression element: {type(node).__name__}")
    return compile(tree, "<expr>", "eval")


def _safe_eval(expr, local_vars):
    compiled = _validate_expression(expr)
    return eval(compiled, {"__builtins__": {}, "np": np, "math": math, **SAFE_FUNCTIONS}, local_vars)


def get_optimization_schema(df, target_col, trained_models):
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if target_col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target_col])
    variables = []
    for col in numeric_df.columns:
        col_series = numeric_df[col].dropna()
        if col_series.empty:
            continue
        lo = float(col_series.min())
        hi = float(col_series.max())
        if lo == hi:
            hi = lo + 1.0
        variables.append({"name": col, "lower": lo, "upper": hi})
    return {
        "variables": variables,
        "algorithms": list(ALGORITHM_REGISTRY.keys()),
        "compare_default": ["NSGA-II", "MOEA/D", "MOPSO", "Adaptive Weighted Sum"],
        "formats": ["png", "svg", "pdf", "emf"],
        "models": list(trained_models.keys()),
        "regression_models": [],
        "objective_template": [
            {"name": "ModelPrediction", "expression": "prediction", "goal": "max"},
            {"name": "EnergyPenalty", "expression": "abs(x1) + abs(x2)", "goal": "min"},
        ],
        "constraint_template": ["prediction - 0.2 >= 0", "x1 + x2 <= 1.5"],
        "surrogate_template": [
            {"name": "Strength", "target_col": "CompressiveStrength", "model_name": "Random Forest", "alias": "strength_pred"},
            {"name": "Flowability", "target_col": "Flowability", "model_name": "Extra Trees", "alias": "flow_pred"},
        ],
        "formula_template": [
            {"name": "Carbon", "expression": "0.9 * cement + 0.08 * slag + 0.005 * coarse_agg", "goal": "min"},
            {"name": "Cost", "expression": "0.5 * cement + 0.18 * slag + 0.03 * water + 0.02 * sand", "goal": "min"},
        ],
    }


def _initialize_population(bounds, size, rng):
    lower = np.array([b["lower"] for b in bounds], dtype=float)
    upper = np.array([b["upper"] for b in bounds], dtype=float)
    pop = rng.uniform(lower, upper, size=(size, len(bounds)))
    return np.clip(pop, lower, upper)


def _candidate_frame(population, bounds):
    return pd.DataFrame(population, columns=[b["name"] for b in bounds])


def _transform_objective(value, spec):
    goal = (spec.get("goal") or "min").lower()
    if goal == "min":
        return value
    if goal == "max":
        return -value
    if goal == "target":
        target = float(spec.get("target", 0))
        return abs(value - target)
    if goal == "range":
        low = float(spec.get("lower", -np.inf))
        high = float(spec.get("upper", np.inf))
        if low <= value <= high:
            return 0.0
        return min(abs(value - low), abs(value - high))
    return value


def _evaluate_population(population, bounds, objectives, constraints, prediction_fn):
    frame = _candidate_frame(population, bounds)
    local_vars = {col: frame[col].to_numpy(dtype=float) for col in frame.columns}
    prediction_context = prediction_fn(frame)
    if isinstance(prediction_context, dict):
        prediction = np.asarray(prediction_context.get("prediction", np.zeros(len(frame))), dtype=float)
        for key, value in prediction_context.items():
            arr = np.asarray(value, dtype=float)
            if arr.ndim == 0:
                arr = np.full(len(frame), float(arr))
            local_vars[key] = arr
    else:
        prediction = np.asarray(prediction_context, dtype=float)
    local_vars["prediction"] = prediction
    local_vars["model_prediction"] = prediction

    raw_objectives = []
    transformed = []
    objective_names = []
    for spec in objectives:
        vals = np.asarray(_safe_eval(spec["expression"], local_vars), dtype=float)
        if vals.ndim == 0:
            vals = np.full(len(frame), float(vals))
        raw_objectives.append(vals)
        transformed.append(np.vectorize(lambda x: _transform_objective(float(x), spec))(vals))
        objective_names.append(spec["name"])

    raw_objectives = np.vstack(raw_objectives).T
    transformed = np.vstack(transformed).T

    violations = np.zeros(len(frame), dtype=float)
    constraint_records = []
    for expr in constraints:
        vals = _safe_eval(expr, local_vars)
        if isinstance(vals, (bool, np.bool_)):
            vals = np.full(len(frame), bool(vals))
        vals = np.asarray(vals)
        if vals.dtype == bool:
            violation = (~vals).astype(float)
        else:
            violation = np.maximum(-vals if ">=" in expr or ">" in expr else vals, 0.0)
        violations += violation
        constraint_records.append(violation)

    return {
        "frame": frame,
        "prediction": prediction,
        "raw_objectives": raw_objectives,
        "transformed": transformed,
        "constraint_violation": violations,
        "objective_names": objective_names,
    }


def _dominates(a_obj, a_violation, b_obj, b_violation):
    if a_violation < b_violation:
        return True
    if a_violation > b_violation:
        return False
    return np.all(a_obj <= b_obj) and np.any(a_obj < b_obj)


def _fast_non_dominated_sort(objs, violations):
    n = len(objs)
    dominates = [[] for _ in range(n)]
    domination_count = np.zeros(n, dtype=int)
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(objs[p], violations[p], objs[q], violations[q]):
                dominates[p].append(q)
            elif _dominates(objs[q], violations[q], objs[p], violations[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominates[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1
    return fronts


def _crowding_distance(front, objs):
    if len(front) == 0:
        return {}
    distance = {idx: 0.0 for idx in front}
    front_objs = objs[front]
    for m in range(front_objs.shape[1]):
        order = np.argsort(front_objs[:, m])
        distance[front[order[0]]] = np.inf
        distance[front[order[-1]]] = np.inf
        low = front_objs[order[0], m]
        high = front_objs[order[-1], m]
        if high - low < 1e-12:
            continue
        for pos in range(1, len(front) - 1):
            left = front_objs[order[pos - 1], m]
            right = front_objs[order[pos + 1], m]
            distance[front[order[pos]]] += (right - left) / (high - low)
    return distance


def _select_by_fronts(population, evals, size):
    fronts = _fast_non_dominated_sort(evals["transformed"], evals["constraint_violation"])
    chosen = []
    for front in fronts:
        if len(chosen) + len(front) <= size:
            chosen.extend(front)
        else:
            distance = _crowding_distance(front, evals["transformed"])
            ranked = sorted(front, key=lambda idx: distance.get(idx, 0.0), reverse=True)
            chosen.extend(ranked[: size - len(chosen)])
            break
    chosen = np.array(chosen, dtype=int)
    return population[chosen], {
        "frame": evals["frame"].iloc[chosen].reset_index(drop=True),
        "prediction": evals["prediction"][chosen],
        "raw_objectives": evals["raw_objectives"][chosen],
        "transformed": evals["transformed"][chosen],
        "constraint_violation": evals["constraint_violation"][chosen],
        "objective_names": evals["objective_names"],
    }


def _blend_crossover(parents, bounds, rng):
    lower = np.array([b["lower"] for b in bounds], dtype=float)
    upper = np.array([b["upper"] for b in bounds], dtype=float)
    idx = rng.permutation(len(parents))
    shuffled = parents[idx]
    alpha = rng.uniform(0.2, 0.8, size=(len(parents), 1))
    children = alpha * parents + (1 - alpha) * shuffled
    mutation = rng.normal(0, (upper - lower) * 0.05, size=children.shape)
    mask = rng.random(children.shape) < 0.35
    children = children + mutation * mask
    return np.clip(children, lower, upper)


def _de_step(pop, bounds, rng):
    lower = np.array([b["lower"] for b in bounds], dtype=float)
    upper = np.array([b["upper"] for b in bounds], dtype=float)
    trial = pop.copy()
    for i in range(len(pop)):
        choices = [idx for idx in range(len(pop)) if idx != i]
        a, b, c = rng.choice(choices, size=3, replace=False)
        mutant = pop[a] + 0.7 * (pop[b] - pop[c])
        cross = rng.random(pop.shape[1]) < 0.7
        if not cross.any():
            cross[rng.integers(0, pop.shape[1])] = True
        trial[i, cross] = mutant[cross]
    return np.clip(trial, lower, upper)


def _mopso_step(pop, archive, velocity, bounds, rng):
    lower = np.array([b["lower"] for b in bounds], dtype=float)
    upper = np.array([b["upper"] for b in bounds], dtype=float)
    leaders = archive[rng.integers(0, len(archive), size=len(pop))]
    inertia = 0.65
    cognitive = 0.9
    social = 0.9
    velocity = (
        inertia * velocity
        + cognitive * rng.random(pop.shape) * (archive[: len(pop)] - pop if len(archive) >= len(pop) else leaders - pop)
        + social * rng.random(pop.shape) * (leaders - pop)
    )
    pop = np.clip(pop + velocity, lower, upper)
    return pop, velocity


def _weighted_step(pop, evals, bounds, rng):
    lower = np.array([b["lower"] for b in bounds], dtype=float)
    upper = np.array([b["upper"] for b in bounds], dtype=float)
    weights = rng.dirichlet(np.ones(evals["transformed"].shape[1]), size=len(pop))
    scores = (evals["transformed"] * weights).sum(axis=1) + 5 * evals["constraint_violation"]
    elite = pop[np.argsort(scores)[: max(2, len(pop) // 3)]]
    children = _blend_crossover(elite[rng.integers(0, len(elite), size=len(pop))], bounds, rng)
    return np.clip(children, lower, upper)


def _epsilon_constraint_step(pop, evals, bounds, rng):
    lower = np.array([b["lower"] for b in bounds], dtype=float)
    upper = np.array([b["upper"] for b in bounds], dtype=float)
    primary = evals["transformed"][:, 0]
    if evals["transformed"].shape[1] > 1:
        epsilon = np.quantile(evals["transformed"][:, 1], 0.5)
        feasible = evals["transformed"][:, 1] <= epsilon
    else:
        feasible = np.ones(len(pop), dtype=bool)
    scores = primary + 10 * (~feasible).astype(float) + 5 * evals["constraint_violation"]
    elite = pop[np.argsort(scores)[: max(2, len(pop) // 3)]]
    return _blend_crossover(elite[rng.integers(0, len(elite), size=len(pop))], bounds, rng)


def _anneal_step(pop, evals, bounds, temperature, rng):
    lower = np.array([b["lower"] for b in bounds], dtype=float)
    upper = np.array([b["upper"] for b in bounds], dtype=float)
    scores = evals["transformed"].sum(axis=1) + 10 * evals["constraint_violation"]
    proposal = pop + rng.normal(0, (upper - lower) * max(0.02, temperature), size=pop.shape)
    proposal = np.clip(proposal, lower, upper)
    return proposal, scores


def _approx_hypervolume(points):
    if len(points) == 0:
        return 0.0
    pts = np.asarray(points, dtype=float)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    denom = np.where(maxs - mins < 1e-9, 1.0, maxs - mins)
    norm = (pts - mins) / denom
    ref = np.ones(norm.shape[1]) * 1.1
    samples = np.random.default_rng(42).uniform(0, ref, size=(3000, norm.shape[1]))
    dominated = np.any(np.all(norm[:, None, :] <= samples[None, :, :], axis=2), axis=0)
    return float(dominated.mean() * np.prod(ref))


def _spacing_metric(points):
    if len(points) < 2:
        return 0.0
    pts = np.asarray(points, dtype=float)
    dists = []
    for i in range(len(pts)):
        others = np.delete(pts, i, axis=0)
        d = np.min(np.linalg.norm(others - pts[i], axis=1))
        dists.append(d)
    return float(np.std(dists))


def _spread_metric(points):
    if len(points) < 2:
        return 0.0
    pts = np.asarray(points, dtype=float)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    return float(np.linalg.norm(maxs - mins))


def _ideal_distance_metric(points):
    if len(points) == 0:
        return 0.0
    pts = np.asarray(points, dtype=float)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    denom = np.where(maxs - mins < 1e-9, 1.0, maxs - mins)
    norm = (pts - mins) / denom
    return float(np.mean(np.linalg.norm(norm, axis=1)))


def _mean_crowding_metric(points):
    if len(points) < 3:
        return 0.0
    front = list(range(len(points)))
    crowd = _crowding_distance(front, np.asarray(points, dtype=float))
    finite = [val for val in crowd.values() if np.isfinite(val)]
    if not finite:
        return 0.0
    return float(np.mean(finite))


def _normalize_decision_matrix(values):
    arr = np.asarray(values, dtype=float)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    denom = np.where(maxs - mins < 1e-9, 1.0, maxs - mins)
    return (arr - mins) / denom, mins, maxs


def _decision_cost_matrix(raw_values, objectives):
    cols = []
    for idx, spec in enumerate(objectives):
        vals = np.asarray(raw_values[:, idx], dtype=float)
        cols.append(np.vectorize(lambda x: _transform_objective(float(x), spec))(vals))
    return np.vstack(cols).T


def _rank_solutions(pareto_frame, raw_objectives, objectives):
    if len(pareto_frame) == 0:
        return pareto_frame
    cost_matrix = _decision_cost_matrix(raw_objectives, objectives)
    norm_cost, _, _ = _normalize_decision_matrix(cost_matrix)
    n_obj = norm_cost.shape[1]
    weights = np.ones(n_obj, dtype=float) / max(n_obj, 1)

    weighted = norm_cost * weights
    ideal_best = weighted.min(axis=0)
    ideal_worst = weighted.max(axis=0)
    d_best = np.linalg.norm(weighted - ideal_best, axis=1)
    d_worst = np.linalg.norm(weighted - ideal_worst, axis=1)
    topsis = d_worst / (d_best + d_worst + 1e-12)

    weighted_sum = 1.0 - np.sum(weighted, axis=1)

    f_star = norm_cost.min(axis=0)
    f_minus = norm_cost.max(axis=0)
    denom = np.where(f_minus - f_star < 1e-9, 1.0, f_minus - f_star)
    s = np.sum(weights * (norm_cost - f_star) / denom, axis=1)
    r = np.max(weights * (norm_cost - f_star) / denom, axis=1)
    s_star, s_minus = s.min(), s.max()
    r_star, r_minus = r.min(), r.max()
    v = 0.5
    q = v * (s - s_star) / (s_minus - s_star + 1e-12) + (1 - v) * (r - r_star) / (r_minus - r_star + 1e-12)

    ranked = pareto_frame.copy()
    ranked["TOPSIS Score"] = topsis
    ranked["TOPSIS Rank"] = pd.Series((-topsis).argsort().argsort() + 1, index=ranked.index)
    ranked["Weighted Sum Score"] = weighted_sum
    ranked["Weighted Sum Rank"] = pd.Series((-weighted_sum).argsort().argsort() + 1, index=ranked.index)
    ranked["VIKOR Q"] = q
    ranked["VIKOR Rank"] = pd.Series(q.argsort().argsort() + 1, index=ranked.index)
    ranked["Compromise Rank"] = ranked[["TOPSIS Rank", "Weighted Sum Rank", "VIKOR Rank"]].mean(axis=1).rank(method="dense").astype(int)
    ranked["Consensus Score"] = 1.0 / (ranked["Compromise Rank"] + 1e-9)
    ranked = ranked.sort_values(["Compromise Rank", "TOPSIS Rank", "VIKOR Rank"]).reset_index(drop=True)
    return ranked


def _summarize_result_metrics(history_df, pareto_eval):
    transformed = pareto_eval["transformed"]
    feasible_ratio = float(history_df["feasible_ratio"].iloc[-1]) if not history_df.empty else 0.0
    metrics = {
        "Hypervolume": float(history_df["hypervolume"].iloc[-1]) if not history_df.empty else 0.0,
        "Feasible Ratio": feasible_ratio,
        "Nondominated Count": int(history_df["nondominated"].iloc[-1]) if not history_df.empty else 0,
        "Spacing": float(history_df["spacing"].iloc[-1]) if not history_df.empty else 0.0,
        "Spread": _spread_metric(transformed),
        "Mean Ideal Distance": _ideal_distance_metric(transformed),
        "Mean Crowding": _mean_crowding_metric(transformed),
    }
    return metrics


def _nondominated_archive(pop, evals):
    fronts = _fast_non_dominated_sort(evals["transformed"], evals["constraint_violation"])
    if not fronts:
        return pop[:0], evals["frame"].iloc[:0]
    front = np.array(fronts[0], dtype=int)
    return pop[front], {
        "frame": evals["frame"].iloc[front].reset_index(drop=True),
        "prediction": evals["prediction"][front],
        "raw_objectives": evals["raw_objectives"][front],
        "transformed": evals["transformed"][front],
        "constraint_violation": evals["constraint_violation"][front],
        "objective_names": evals["objective_names"],
    }


def _build_pareto_plot(pareto_evals, fmt="png", dpi=300):
    apply_nature_style()
    raw = pareto_evals["raw_objectives"]
    names = pareto_evals["objective_names"]
    fig = plt.figure(figsize=(8, 6))
    if raw.shape[1] >= 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(raw[:, 0], raw[:, 1], raw[:, 2], c=pareto_evals["prediction"], cmap="viridis", s=45, alpha=0.85)
        ax.set_zlabel(names[2])
    else:
        ax = fig.add_subplot(111)
        yvals = raw[:, 1] if raw.shape[1] > 1 else pareto_evals["prediction"]
        sc = ax.scatter(raw[:, 0], yvals, c=pareto_evals["prediction"], cmap="viridis", s=48, alpha=0.85)
        fig.colorbar(sc, ax=ax, label="Model prediction")
        if raw.shape[1] > 1:
            ax.set_ylabel(names[1])
        else:
            ax.set_ylabel("Prediction")
    ax.set_xlabel(names[0])
    ax.set_title("Pareto Front")
    return fig_to_base64(fig, fmt, dpi)


def _build_parallel_coordinates(pareto_evals, fmt="png", dpi=300):
    apply_nature_style()
    raw = pareto_evals["raw_objectives"]
    names = pareto_evals["objective_names"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if len(raw) == 0:
        return fig_to_base64(fig, fmt, dpi)
    mins = raw.min(axis=0)
    maxs = raw.max(axis=0)
    denom = np.where(maxs - mins < 1e-9, 1.0, maxs - mins)
    norm = (raw - mins) / denom
    for row in norm:
        ax.plot(range(len(names)), row, alpha=0.35, color=NATURE_COLORS["blue"])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylim(0, 1)
    ax.set_title("Pareto Objective Parallel Coordinates")
    return fig_to_base64(fig, fmt, dpi)


def _build_metric_history(history, fmt="png", dpi=300):
    apply_nature_style()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(history["generation"], history["hypervolume"], label="Hypervolume", color=NATURE_COLORS["red"], linewidth=2)
    ax.plot(history["generation"], history["feasible_ratio"], label="Feasible ratio", color=NATURE_COLORS["teal"], linewidth=2)
    ax.plot(history["generation"], history["nondominated"], label="Nondominated count", color=NATURE_COLORS["purple"], linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_title("Optimization Progress")
    ax.legend()
    return fig_to_base64(fig, fmt, dpi)


def _build_objective_heatmap(pareto_evals, fmt="png", dpi=300):
    apply_nature_style()
    raw = pd.DataFrame(pareto_evals["raw_objectives"], columns=pareto_evals["objective_names"])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    corr = raw.corr()
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("Objective Correlation Heatmap")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return fig_to_base64(fig, fmt, dpi)


def _build_spacing_history(history, fmt="png", dpi=300):
    apply_nature_style()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(history["generation"], history["spacing"], color=NATURE_COLORS["purple"], linewidth=2, marker="o")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Spacing")
    ax.set_title("Pareto Spacing History")
    return fig_to_base64(fig, fmt, dpi)


def _build_solution_correlation(pareto_frame, objective_names, fmt="png", dpi=300):
    apply_nature_style()
    cols = [c for c in pareto_frame.columns if c in objective_names or c == "prediction"]
    if not cols:
        cols = pareto_frame.select_dtypes(include=[np.number]).columns.tolist()[:8]
    corr = pareto_frame[cols].corr()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("Solution Correlation Heatmap")
    fig.colorbar(im, ax=ax, shrink=0.82)
    return fig_to_base64(fig, fmt, dpi)


def _build_objective_distribution(pareto_frame, objective_names, fmt="png", dpi=300):
    apply_nature_style()
    plot_cols = [c for c in objective_names if c in pareto_frame.columns][:4]
    fig, axes = plt.subplots(len(plot_cols), 1, figsize=(8, max(4.5, 2.2 * len(plot_cols))), squeeze=False)
    for ax, col in zip(axes.ravel(), plot_cols):
        sns.histplot(pareto_frame[col], kde=True, ax=ax, color=NATURE_COLORS["orange"])
        ax.set_title(f"Distribution: {col}")
    fig.tight_layout()
    return fig_to_base64(fig, fmt, dpi)


def _build_tradeoff_matrix(pareto_evals, fmt="png", dpi=300):
    apply_nature_style()
    raw = pd.DataFrame(pareto_evals["raw_objectives"], columns=pareto_evals["objective_names"])
    if raw.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "At least two objectives required", ha="center", va="center")
        ax.axis("off")
        return fig_to_base64(fig, fmt, dpi)
    sample = raw.head(min(len(raw), 120))
    g = sns.pairplot(sample, corner=True, diag_kind="kde", plot_kws={"s": 24, "alpha": 0.65, "color": NATURE_COLORS["blue"]})
    g.fig.suptitle("Objective Trade-off Matrix", y=1.02)
    return fig_to_base64(g.fig, fmt, dpi)


def _build_solution_rankings(ranked_frame, fmt="png", dpi=300):
    apply_nature_style()
    top = ranked_frame.head(min(15, len(ranked_frame))).copy()
    fig, ax = plt.subplots(figsize=(9, max(4.6, 0.35 * len(top) + 2.3)))
    sns.barplot(data=top, x="TOPSIS Score", y=top.index.astype(str), orient="h", color=NATURE_COLORS["teal"], ax=ax)
    ax.set_xlabel("TOPSIS score")
    ax.set_ylabel("Solution order")
    ax.set_title("TOPSIS Screening of Pareto Solutions")
    return fig_to_base64(fig, fmt, dpi)


def _build_ranking_agreement(ranked_frame, fmt="png", dpi=300):
    apply_nature_style()
    top = ranked_frame.head(min(20, len(ranked_frame))).copy()
    rank_df = top[["TOPSIS Rank", "Weighted Sum Rank", "VIKOR Rank", "Compromise Rank"]]
    fig, ax = plt.subplots(figsize=(7.6, max(4.4, 0.32 * len(rank_df) + 2.2)))
    sns.heatmap(rank_df, annot=True, fmt=".0f", cmap="YlGnBu_r", cbar=False, ax=ax)
    ax.set_title("Ranking Method Agreement")
    return fig_to_base64(fig, fmt, dpi)


def _build_solution_score_heatmap(ranked_frame, objective_names, fmt="png", dpi=300):
    apply_nature_style()
    cols = [c for c in objective_names if c in ranked_frame.columns]
    extra = [c for c in ["TOPSIS Score", "Weighted Sum Score", "VIKOR Q"] if c in ranked_frame.columns]
    sample = ranked_frame[cols + extra].head(min(20, len(ranked_frame)))
    fig, ax = plt.subplots(figsize=(8.8, max(4.6, 0.28 * len(sample) + 2.1)))
    sns.heatmap(sample, cmap="crest", annot=True, fmt=".3f", ax=ax)
    ax.set_title("Solution Score Heatmap")
    return fig_to_base64(fig, fmt, dpi)


def _build_top_solution_profile(ranked_frame, objective_names, objectives, fmt="png", dpi=300):
    apply_nature_style()
    if ranked_frame.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No Pareto solution available", ha="center", va="center")
        ax.axis("off")
        return fig_to_base64(fig, fmt, dpi)
    top = ranked_frame.iloc[0]
    values = [float(top[name]) for name in objective_names if name in ranked_frame.columns]
    labels = [name for name in objective_names if name in ranked_frame.columns]
    norms, _, _ = _normalize_decision_matrix(np.array(values, dtype=float).reshape(1, -1))
    goals = [spec.get("goal", "min") for spec in objectives[: len(labels)]]
    adjusted = [1.0 - norms[0, i] if goals[i] == "min" else norms[0, i] for i in range(len(labels))]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(labels, adjusted, color=_distinct_palette(len(labels)))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Normalized desirability")
    ax.set_title("Best Solution Objective Profile")
    ax.tick_params(axis="x", rotation=20)
    for bar, raw in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{raw:.3f}", ha="center", va="bottom", fontsize=8)
    return fig_to_base64(fig, fmt, dpi)


def build_shrinkage_curve_plot(curve_df, time_col="age", value_col="predicted_shrinkage", fmt="png", dpi=300):
    apply_nature_style()
    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    if curve_df is None or curve_df.empty:
        ax.text(0.5, 0.5, "No shrinkage curve data available", ha="center", va="center")
        ax.axis("off")
        return fig_to_base64(fig, fmt, dpi)

    plot_df = curve_df.copy()
    plot_df[time_col] = pd.to_numeric(plot_df[time_col], errors="coerce")
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[time_col, value_col])
    solution_col = "solution_id" if "solution_id" in plot_df.columns else None
    label_col = "solution_label" if "solution_label" in plot_df.columns else solution_col
    palette = _distinct_palette(max(plot_df[solution_col].nunique() if solution_col else 1, 1))

    if solution_col:
        for color_idx, (solution_id, group) in enumerate(plot_df.groupby(solution_col, sort=False)):
            group = group.sort_values(time_col)
            label = str(group[label_col].iloc[0]) if label_col and label_col in group.columns else f"Solution {solution_id}"
            ax.plot(group[time_col], group[value_col], marker="o", markersize=3.5, linewidth=2.2, color=palette[color_idx % len(palette)], label=label)
            if {"lower_shrinkage", "upper_shrinkage"}.issubset(group.columns):
                lower = pd.to_numeric(group["lower_shrinkage"], errors="coerce")
                upper = pd.to_numeric(group["upper_shrinkage"], errors="coerce")
                if lower.notna().any() and upper.notna().any():
                    ax.fill_between(group[time_col], lower, upper, color=palette[color_idx % len(palette)], alpha=0.16, linewidth=0)
    else:
        plot_df = plot_df.sort_values(time_col)
        ax.plot(plot_df[time_col], plot_df[value_col], marker="o", linewidth=2.2, color=NATURE_COLORS["blue"], label="Prediction")
        if {"lower_shrinkage", "upper_shrinkage"}.issubset(plot_df.columns):
            ax.fill_between(plot_df[time_col], plot_df["lower_shrinkage"], plot_df["upper_shrinkage"], color=NATURE_COLORS["blue"], alpha=0.16, linewidth=0)

    ax.set_xlabel(time_col)
    ax.set_ylabel("Predicted shrinkage")
    ax.set_title("Optimized Mix Predicted Shrinkage Curves")
    ax.legend(loc="best", fontsize=8)
    return fig_to_base64(fig, fmt, dpi)


def _distinct_palette(count):
    cmap = plt.get_cmap("viridis")
    return [cmap(val) for val in np.linspace(0.1, 0.9, max(count, 1))]


def _build_algorithm_comparison_bars(compare_df, metric_name, fmt="png", dpi=300):
    apply_nature_style()
    fig, ax = plt.subplots(figsize=(8.2, max(4.5, 0.45 * len(compare_df) + 2)))
    sns.barplot(data=compare_df, x=metric_name, y="Algorithm", hue="Algorithm", dodge=False, palette="crest", legend=False, ax=ax)
    ax.set_title(f"Algorithm Comparison: {metric_name}")
    return fig_to_base64(fig, fmt, dpi)


def _build_algorithm_metric_heatmap(compare_df, fmt="png", dpi=300):
    apply_nature_style()
    metric_cols = [c for c in COMPARE_METRIC_NAMES if c in compare_df.columns]
    heatmap_df = compare_df.set_index("Algorithm")[metric_cols]
    fig, ax = plt.subplots(figsize=(9, max(4.6, 0.4 * len(compare_df) + 2.2)))
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="mako", ax=ax)
    ax.set_title("Algorithm Performance Matrix")
    return fig_to_base64(fig, fmt, dpi)


def _build_algorithm_ranking(compare_df, fmt="png", dpi=300):
    apply_nature_style()
    metric_cols = [c for c in COMPARE_METRIC_NAMES if c in compare_df.columns]
    rank_df = compare_df.copy()
    for col in metric_cols:
        ascending = col in {"Spacing", "Mean Ideal Distance"}
        rank_df[f"{col} Rank"] = rank_df[col].rank(ascending=ascending, method="dense")
    rank_cols = [f"{c} Rank" for c in metric_cols]
    rank_df["Consensus Rank"] = rank_df[rank_cols].mean(axis=1).rank(method="dense")
    plot_df = rank_df.set_index("Algorithm")[rank_cols + ["Consensus Rank"]]
    fig, ax = plt.subplots(figsize=(10, max(4.8, 0.42 * len(compare_df) + 2.6)))
    sns.heatmap(plot_df, annot=True, fmt=".1f", cmap="YlOrRd_r", cbar=False, ax=ax)
    ax.set_title("Algorithm Ranking Agreement")
    return fig_to_base64(fig, fmt, dpi)


def _build_history_compare(compare_histories, metric_name, fmt="png", dpi=300):
    apply_nature_style()
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for algo, history_df in compare_histories.items():
        if history_df.empty or metric_name not in history_df.columns:
            continue
        ax.plot(history_df["generation"], history_df[metric_name], linewidth=2, label=algo)
    ax.set_xlabel("Generation")
    ax.set_title(f"Algorithm Progress: {metric_name.replace('_', ' ').title()}")
    ax.legend()
    return fig_to_base64(fig, fmt, dpi)


def _rank_algorithm_records(records):
    if not records:
        return pd.DataFrame()
    compare_df = pd.DataFrame(records)
    metric_cols = [c for c in COMPARE_METRIC_NAMES if c in compare_df.columns]
    if not metric_cols:
        compare_df["Consensus Rank"] = 1
        return compare_df
    rank_df = compare_df.copy()
    for col in metric_cols:
        ascending = col in {"Spacing", "Mean Ideal Distance"}
        rank_df[f"{col} Rank"] = rank_df[col].rank(ascending=ascending, method="dense")
    rank_cols = [f"{c} Rank" for c in metric_cols]
    rank_df["Consensus Score"] = rank_df[rank_cols].mean(axis=1)
    rank_df["Consensus Rank"] = rank_df["Consensus Score"].rank(ascending=True, method="dense").astype(int)
    return rank_df.sort_values(["Consensus Rank", "Consensus Score", "Algorithm"]).reset_index(drop=True)


def _run_single_optimization(schema, objectives, constraints, algorithm_name, prediction_fn, pop_size=64, generations=30, random_state=42):
    rng = np.random.default_rng(random_state)
    mode = ALGORITHM_REGISTRY.get(algorithm_name, "nsga2")
    population = _initialize_population(schema, pop_size, rng)
    velocity = np.zeros_like(population)
    history = {"generation": [], "hypervolume": [], "feasible_ratio": [], "nondominated": [], "spacing": []}

    archive_pop = population.copy()
    archive_evals = _evaluate_population(archive_pop, schema, objectives, constraints, prediction_fn)

    for generation in range(1, generations + 1):
        evals = _evaluate_population(population, schema, objectives, constraints, prediction_fn)
        combined_pop = np.vstack([archive_pop, population])
        combined_eval = _evaluate_population(combined_pop, schema, objectives, constraints, prediction_fn)
        archive_pop, archive_evals = _nondominated_archive(combined_pop, combined_eval)
        population, evals = _select_by_fronts(population, evals, min(pop_size, len(population)))

        history["generation"].append(generation)
        history["hypervolume"].append(_approx_hypervolume(archive_evals["transformed"]))
        history["feasible_ratio"].append(float(np.mean(evals["constraint_violation"] <= 1e-9)))
        history["nondominated"].append(int(len(archive_evals["frame"])))
        history["spacing"].append(_spacing_metric(archive_evals["transformed"]))

        if mode == "nsga2":
            offspring = _blend_crossover(population[rng.integers(0, len(population), size=len(population))], schema, rng)
            population = np.vstack([population, offspring])
            population, _ = _select_by_fronts(population, _evaluate_population(population, schema, objectives, constraints, prediction_fn), pop_size)
        elif mode == "de":
            trial = _de_step(population, schema, rng)
            population = np.vstack([population, trial])
            population, _ = _select_by_fronts(population, _evaluate_population(population, schema, objectives, constraints, prediction_fn), pop_size)
        elif mode == "mopso":
            leaders = archive_pop if len(archive_pop) else population
            population, velocity = _mopso_step(population, leaders, velocity, schema, rng)
        elif mode == "moead":
            trial = _weighted_step(population, evals, schema, rng)
            population = np.vstack([population, trial])
            population, _ = _select_by_fronts(population, _evaluate_population(population, schema, objectives, constraints, prediction_fn), pop_size)
        elif mode == "weighted":
            population = _weighted_step(population, evals, schema, rng)
        elif mode == "epsilon":
            population = _epsilon_constraint_step(population, evals, schema, rng)
        elif mode == "anneal":
            temperature = max(0.02, 1 - generation / max(generations, 1))
            proposal, current_scores = _anneal_step(population, evals, schema, temperature, rng)
            proposal_eval = _evaluate_population(proposal, schema, objectives, constraints, prediction_fn)
            proposal_scores = proposal_eval["transformed"].sum(axis=1) + 10 * proposal_eval["constraint_violation"]
            accept = proposal_scores < current_scores + rng.random(len(current_scores)) * temperature
            population[accept] = proposal[accept]
        else:
            population = _initialize_population(schema, pop_size, rng)

    final_eval = _evaluate_population(archive_pop, schema, objectives, constraints, prediction_fn)
    pareto_pop, pareto_eval = _nondominated_archive(archive_pop, final_eval)
    pareto_frame = pareto_eval["frame"].copy()
    pareto_frame["prediction"] = pareto_eval["prediction"]
    for idx, obj_name in enumerate(pareto_eval["objective_names"]):
        pareto_frame[obj_name] = pareto_eval["raw_objectives"][:, idx]
    pareto_frame["constraint_violation"] = pareto_eval["constraint_violation"]
    pareto_frame = pareto_frame.sort_values("constraint_violation").reset_index(drop=True)
    history_df = pd.DataFrame(history)
    metrics = _summarize_result_metrics(history_df, pareto_eval)
    ranked_frame = _rank_solutions(pareto_frame, pareto_eval["raw_objectives"], objectives)
    return {
        "metrics": metrics,
        "pareto_eval": pareto_eval,
        "pareto_frame": pareto_frame,
        "ranked_frame": ranked_frame,
        "history_df": history_df,
    }


def run_multiobjective_optimization(
    schema,
    objectives,
    constraints,
    algorithm_name,
    prediction_fn,
    pop_size=64,
    generations=30,
    random_state=42,
    fmt="png",
    dpi=300,
    compare_algorithms=None,
):
    requested_algorithm = algorithm_name
    selected_compare = compare_algorithms or []
    ordered_algorithms = []
    for algo in [algorithm_name] + selected_compare:
        if algo and algo not in ordered_algorithms:
            ordered_algorithms.append(algo)

    runs = {}
    comparison_records = []
    comparison_histories = {}
    for idx, algo in enumerate(ordered_algorithms):
        run = _run_single_optimization(schema, objectives, constraints, algo, prediction_fn, pop_size, generations, random_state + 17 * idx)
        runs[algo] = run
        record = {"Algorithm": algo}
        for key, val in run["metrics"].items():
            record[key] = round(val, 4) if isinstance(val, float) else val
        comparison_records.append(record)
        comparison_histories[algo] = run["history_df"]

    ranking_df = _rank_algorithm_records(comparison_records)
    best_algorithm = ranking_df.iloc[0]["Algorithm"] if not ranking_df.empty else algorithm_name
    selected_run = runs.get(best_algorithm) or runs[algorithm_name]

    pareto_eval = selected_run["pareto_eval"]
    pareto_frame = selected_run["pareto_frame"]
    ranked_frame = selected_run["ranked_frame"]
    history_df = selected_run["history_df"]
    metrics = {k: round(v, 4) if isinstance(v, float) else v for k, v in selected_run["metrics"].items()}

    plots = {
        "Pareto Front": _build_pareto_plot(pareto_eval, fmt, dpi),
        "Parallel Coordinates": _build_parallel_coordinates(pareto_eval, fmt, dpi),
        "Progress History": _build_metric_history(history_df, fmt, dpi),
        "Objective Correlation": _build_objective_heatmap(pareto_eval, fmt, dpi),
        "Spacing History": _build_spacing_history(history_df, fmt, dpi),
        "Solution Correlation": _build_solution_correlation(pareto_frame, pareto_eval["objective_names"], fmt, dpi),
        "Objective Distribution": _build_objective_distribution(pareto_frame, pareto_eval["objective_names"], fmt, dpi),
        "Trade-off Matrix": _build_tradeoff_matrix(pareto_eval, fmt, dpi),
        "TOPSIS Screening": _build_solution_rankings(ranked_frame, fmt, dpi),
        "Ranking Agreement": _build_ranking_agreement(ranked_frame, fmt, dpi),
        "Solution Score Heatmap": _build_solution_score_heatmap(ranked_frame, pareto_eval["objective_names"], fmt, dpi),
        "Best Solution Profile": _build_top_solution_profile(ranked_frame, pareto_eval["objective_names"], objectives, fmt, dpi),
    }

    comparison = pd.DataFrame(comparison_records) if selected_compare else pd.DataFrame()
    comparison_plots = {}
    if not comparison.empty:
        comparison_plots = {
            "Algorithm Hypervolume": _build_algorithm_comparison_bars(comparison, "Hypervolume", fmt, dpi),
            "Algorithm Feasible Ratio": _build_algorithm_comparison_bars(comparison, "Feasible Ratio", fmt, dpi),
            "Algorithm Nondominated Count": _build_algorithm_comparison_bars(comparison, "Nondominated Count", fmt, dpi),
            "Algorithm Performance Matrix": _build_algorithm_metric_heatmap(comparison, fmt, dpi),
            "Algorithm Ranking Agreement": _build_algorithm_ranking(comparison, fmt, dpi),
            "Algorithm Hypervolume History": _build_history_compare(comparison_histories, "hypervolume", fmt, dpi),
            "Algorithm Feasible Ratio History": _build_history_compare(comparison_histories, "feasible_ratio", fmt, dpi),
            "Algorithm Spacing History": _build_history_compare(comparison_histories, "spacing", fmt, dpi),
        }

    return {
        "metrics": metrics,
        "solutions": ranked_frame.head(80).round(5).replace({np.nan: None}).to_dict(orient="records"),
        "plots": plots,
        "comparison_metrics": comparison.round(5).replace({np.nan: None}).to_dict(orient="records") if not comparison.empty else [],
        "comparison_plots": comparison_plots,
        "objective_names": pareto_eval["objective_names"],
        "history": history_df.round(5).to_dict(orient="records"),
        "best_solution": ranked_frame.head(1).round(5).replace({np.nan: None}).to_dict(orient="records")[0] if not ranked_frame.empty else None,
        "requested_algorithm": requested_algorithm,
        "selected_algorithm": best_algorithm,
        "best_algorithm": best_algorithm,
        "algorithm_selection_metrics": ranking_df.round(5).replace({np.nan: None}).to_dict(orient="records") if not ranking_df.empty else [],
    }
