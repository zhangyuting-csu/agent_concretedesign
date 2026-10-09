import os
import sys
import io
import re
import base64
import zipfile
import json
from itertools import combinations
from urllib.parse import unquote

LOCAL_PY_PACKAGES = os.path.join(os.path.dirname(__file__), ".python_packages")
if os.path.isdir(LOCAL_PY_PACKAGES) and LOCAL_PY_PACKAGES not in sys.path:
    sys.path.insert(0, LOCAL_PY_PACKAGES)

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
import numpy as np


app = FastAPI(title="NatureML Pro Data Science Platform", version="2.0")


def _slugify_filename(value):
    text = re.sub(r'[\\/:*?"<>|]+', "_", str(value or "plot")).strip()
    text = re.sub(r"\s+", "_", text)
    return text or "plot"


def _decode_data_url(data_url):
    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        raise ValueError("Invalid data URL payload.")
    header, encoded = data_url.split(",", 1)
    if ";base64" not in header:
        raise ValueError("Only base64 data URLs are supported.")
    mime = header[5:].split(";", 1)[0]
    ext_map = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/webp": "webp",
        "image/svg+xml": "svg",
        "image/emf": "emf",
        "application/pdf": "pdf",
    }
    ext = ext_map.get(mime, "bin")
    return mime, ext, base64.b64decode(unquote(encoded))


def _safe_sheet_name(value, used=None):
    used = used if used is not None else set()
    name = re.sub(r"[\[\]:*?/\\]+", "_", str(value or "Sheet")).strip()[:31] or "Sheet"
    base = name
    suffix = 2
    while name in used:
        tail = f"_{suffix}"
        name = f"{base[:31 - len(tail)]}{tail}"
        suffix += 1
    used.add(name)
    return name


def _plot_type_label(plot_name, context=None, payload=None):
    text = " ".join(str(x or "") for x in [plot_name, context, (payload or {}).get("plot_type")]).lower()
    rules = [
        ("radar", "Radar / 雷达图"),
        ("radial", "Radial / 雷达径向图"),
        ("taylor", "Taylor Diagram / 泰勒图"),
        ("heatmap", "Heatmap / 热力图"),
        ("corr", "Correlation Heatmap / 相关性热力图"),
        ("confusion", "Confusion Matrix / 混淆矩阵"),
        ("roc", "ROC Curve / ROC 曲线"),
        ("precision-recall", "Precision-Recall Curve / PR 曲线"),
        ("calibration", "Calibration Curve / 校准曲线"),
        ("scatter", "Scatter Plot / 散点图"),
        ("reg", "Regression Scatter / 回归散点图"),
        ("hex", "Hexbin Density / 六边形密度图"),
        ("joint", "Joint Distribution / 联合分布图"),
        ("hist", "Histogram / 直方图"),
        ("density", "Density Curve / 密度曲线"),
        ("ecdf", "ECDF Curve / 经验累计分布图"),
        ("box", "Box Plot / 箱线图"),
        ("violin", "Violin Plot / 小提琴图"),
        ("strip", "Strip Plot / 条带散点图"),
        ("pair", "Pairplot Matrix / 成对关系矩阵图"),
        ("missing", "Missingness Map / 缺失值图"),
        ("shap", "SHAP Explanation Plot / SHAP 解释图"),
        ("pdp", "PDP/ALE Response Plot / PDP 或 ALE 响应图"),
        ("permutation", "Permutation Importance Plot / 置换重要性图"),
        ("counterfactual", "Counterfactual Plot / 反事实解释图"),
        ("pareto", "Pareto Front / 帕累托前沿图"),
        ("leaderboard", "Leaderboard Bar Chart / 排行榜条形图"),
        ("matrix", "Matrix Heatmap / 矩阵热力图"),
        ("curve", "Curve Plot / 曲线图"),
        ("bar", "Bar Chart / 条形图"),
    ]
    for key, label in rules:
        if key in text:
            return label
    return f"{plot_name or 'Plot'} / 数据图"


def _to_frame(data, name="value"):
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data.reset_index(drop=False) if data.index.name is not None else data.reset_index(drop=True)
    if isinstance(data, pd.Series):
        return data.rename(name).reset_index()
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            return pd.DataFrame({name: data})
        return pd.DataFrame(data)
    if isinstance(data, list):
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)
    if isinstance(data, dict):
        try:
            return pd.DataFrame(data)
        except Exception:
            return pd.DataFrame([data])
    return pd.DataFrame({name: [data]})


def _jsonable(value):
    if isinstance(value, str):
        if value.startswith("data:"):
            return f"{value.split(',', 1)[0]},<base64 omitted>"
        if len(value) > 30000:
            return f"{value[:30000]}...<truncated>"
        return value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return json.dumps([_jsonable(v) for v in value], ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps({str(k): _jsonable(v) for k, v in value.items()}, ensure_ascii=False)
    return str(value)


def _strip_data_urls(value):
    if isinstance(value, dict):
        return {k: ("<base64 image omitted>" if k == "data_url" else _strip_data_urls(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_data_urls(v) for v in value]
    if isinstance(value, str) and value.startswith("data:"):
        return "<base64 image omitted>"
    return value


def _write_workbook(filename_base, metadata, sheets):
    output = io.BytesIO()
    used = set()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        meta_rows = [{"Field": key, "Value": _jsonable(value)} for key, value in metadata.items()]
        pd.DataFrame(meta_rows).to_excel(writer, sheet_name=_safe_sheet_name("Plot_Metadata", used), index=False)
        for sheet_name, data in sheets:
            frame = _to_frame(data)
            if frame.empty:
                frame = pd.DataFrame({"message": ["No tabular data available for this sheet."]})
            frame.to_excel(writer, sheet_name=_safe_sheet_name(sheet_name, used), index=False)
    encoded = base64.b64encode(output.getvalue()).decode("utf-8")
    return {
        "status": "success",
        "filename": f"{_slugify_filename(filename_base)}.xlsx",
        "content_base64": encoded,
    }


def _series_with_prediction(model, X, y=None, prefix="test"):
    frame = _to_frame(X).copy()
    if y is not None:
        frame[f"{prefix}_true"] = np.asarray(y)
    if model is not None and X is not None:
        try:
            pred = model.predict(X)
            frame[f"{prefix}_prediction"] = np.asarray(pred)
        except Exception as exc:
            frame[f"{prefix}_prediction_error"] = str(exc)
        try:
            if hasattr(model, "predict_proba"):
                proba = np.asarray(model.predict_proba(X))
                for idx in range(proba.shape[1]):
                    frame[f"{prefix}_probability_class_{idx}"] = proba[:, idx]
        except Exception:
            pass
    return frame


def _safe_qcut_labels(values, q=8, prefix="Q"):
    values = pd.Series(values).reset_index(drop=True)
    unique = int(values.nunique(dropna=True))
    if unique <= 1:
        return pd.Series([f"{prefix}1"] * len(values))
    bins = min(q, unique)
    try:
        codes = pd.qcut(values, q=bins, labels=False, duplicates="drop")
    except Exception:
        codes = pd.cut(values, bins=bins, labels=False, duplicates="drop", include_lowest=True)
    return pd.Series(codes).fillna(0).astype(int).map(lambda x: f"{prefix}{int(x) + 1}")


def _ecdf_frame(values, value_name="value"):
    arr = np.sort(np.asarray(values, dtype=float))
    if len(arr) == 0:
        return pd.DataFrame(columns=[value_name, "ecdf"])
    return pd.DataFrame({value_name: arr, "ecdf": np.arange(1, len(arr) + 1) / len(arr)})


def _model_plot_reconstruction_sheets(plot_name, model, X_test, y_test, X_train=None, y_train=None):
    sheets = []
    if model is None or X_test is None or y_test is None:
        return sheets
    try:
        pred = np.asarray(model.predict(X_test), dtype=float)
        y_true = np.asarray(y_test, dtype=float)
    except Exception:
        return sheets
    if len(y_true) == 0 or len(pred) != len(y_true):
        return sheets

    name = str(plot_name or "").lower()
    residual = y_true - pred
    signed_error = pred - y_true
    abs_error = np.abs(residual)
    n = len(y_true)
    point = pd.DataFrame(
        {
            "sample_index": np.arange(n),
            "observed": y_true,
            "predicted": pred,
            "residual_observed_minus_predicted": residual,
            "signed_error_predicted_minus_observed": signed_error,
            "absolute_error": abs_error,
            "relative_absolute_error": abs_error / (np.abs(y_true) + 1e-9),
            "observed_percentile": pd.Series(y_true).rank(pct=True).to_numpy(),
            "predicted_percentile": pd.Series(pred).rank(pct=True).to_numpy(),
        }
    )
    sheets.append(("Point_Level_Plot_Data", point))

    def add(sheet_name, frame):
        if frame is not None:
            sheets.append((sheet_name, frame))

    if "lag" in name:
        add("Plot_Reconstruction_Data", pd.DataFrame({"residual_t": residual[:-1], "residual_t_plus_1": residual[1:]}))
    elif "rolling rmse" in name:
        window = max(4, n // 10)
        add("Plot_Reconstruction_Data", pd.DataFrame({"sample_order": np.arange(n), "rolling_rmse": pd.Series(residual).rolling(window, min_periods=2).apply(lambda x: float(np.sqrt(np.mean(x**2))))}))
    elif "quantile trend" in name:
        window = max(6, n // 8)
        roll = pd.Series(residual).rolling(window, min_periods=4)
        add("Plot_Reconstruction_Data", pd.DataFrame({"sample_order": np.arange(n), "residual_p10": roll.quantile(0.1), "residual_p50": roll.quantile(0.5), "residual_p90": roll.quantile(0.9)}))
    elif "volatility" in name:
        window = max(4, n // 10)
        add("Plot_Reconstruction_Data", pd.DataFrame({"sample_order": np.arange(n), "rolling_residual_std": pd.Series(residual).rolling(window, min_periods=2).std()}))
    elif "pareto" in name or "contribution" in name or "waterfall" in name or "tail concentration" in name:
        order = np.argsort(abs_error)[::-1]
        sorted_df = point.iloc[order].reset_index(drop=True)
        sorted_df["rank_by_absolute_error"] = np.arange(1, len(sorted_df) + 1)
        sorted_df["cumulative_error_share"] = sorted_df["absolute_error"].cumsum() / (sorted_df["absolute_error"].sum() + 1e-9)
        sorted_df["is_top_10_percent_error"] = sorted_df["rank_by_absolute_error"] <= max(1, int(np.ceil(0.1 * n)))
        add("Plot_Reconstruction_Data", sorted_df)
    elif "decile" in name or "segment" in name or "ladder" in name or "calibration residual" in name or "mape by target" in name:
        if "target" in name or "observed" in name or "mape" in name:
            segment = _safe_qcut_labels(y_true, 10, "T")
            segment_source = y_true
        else:
            segment = _safe_qcut_labels(pred, 10, "P")
            segment_source = pred
        frame = point.copy()
        frame["segment"] = segment
        frame["segment_source_value"] = segment_source
        grouped = frame.groupby("segment", observed=False).agg(
            n=("sample_index", "size"),
            observed_mean=("observed", "mean"),
            predicted_mean=("predicted", "mean"),
            residual_mean=("residual_observed_minus_predicted", "mean"),
            signed_error_median=("signed_error_predicted_minus_observed", "median"),
            absolute_error_mean=("absolute_error", "mean"),
            absolute_error_median=("absolute_error", "median"),
            mape_median=("relative_absolute_error", lambda x: float(np.median(x) * 100)),
        ).reset_index()
        add("Plot_Reconstruction_Data", grouped)
        add("Segment_Point_Assignment", frame[["sample_index", "segment", "observed", "predicted", "residual_observed_minus_predicted", "absolute_error"]])
    elif "qq" in name or "normality" in name:
        q = np.linspace(0.01, 0.99, n)
        rng = np.random.default_rng(42)
        add("Plot_Reconstruction_Data", pd.DataFrame({
            "probability": q,
            "theoretical_normal_quantile": np.quantile(rng.normal(size=8000), q),
            "standardized_residual_quantile": np.quantile((residual - residual.mean()) / (residual.std() + 1e-9), q),
            "tail_flag": np.where((q < 0.1) | (q > 0.9), "tail", "center"),
        }))
    elif "ecdf" in name:
        pos = _ecdf_frame(residual[residual > 0], "positive_residual")
        neg = _ecdf_frame(-residual[residual < 0], "negative_residual_magnitude")
        add("Positive_Residual_ECDF", pos)
        add("Negative_Residual_ECDF", neg)
    elif "symmetry" in name:
        q = np.linspace(0.05, 0.95, 19)
        add("Plot_Reconstruction_Data", pd.DataFrame({"probability": q, "negative_residual_quantile": np.quantile(np.maximum(-residual, 0), q), "positive_residual_quantile": np.quantile(np.maximum(residual, 0), q)}))
    elif "periodogram" in name or "spectrum" in name:
        centered = residual - residual.mean()
        add("Plot_Reconstruction_Data", pd.DataFrame({"frequency": np.fft.rfftfreq(len(centered)), "power": np.abs(np.fft.rfft(centered)) ** 2}))
    elif "tolerance" in name:
        thresholds = np.linspace(0, np.quantile(abs_error, 0.98), 60)
        add("Plot_Reconstruction_Data", pd.DataFrame({"absolute_error_threshold": thresholds, "coverage": [np.mean(abs_error <= t) for t in thresholds], "utility": [np.mean(abs_error <= t) - 0.02 * t for t in thresholds]}))
    elif "probabilistic prediction interval" in name or "bootstrap uncertainty" in name or "uncertainty band" in name:
        estimators = getattr(model, "estimators_", None)
        interval_frame = None
        if estimators is not None:
            try:
                members = [est for est in np.ravel(estimators) if est is not None and hasattr(est, "predict")]
                if len(members) >= 5:
                    X_member = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test
                    member_preds = np.vstack([np.asarray(est.predict(X_member), dtype=float) for est in members])
                    low = np.nanquantile(member_preds, 0.05, axis=0)
                    high = np.nanquantile(member_preds, 0.95, axis=0)
                    mean_pred = np.nanmean(member_preds, axis=0)
                    interval_frame = point.copy()
                    interval_frame["ensemble_mean_prediction"] = mean_pred
                    interval_frame["interval_lower_5pct"] = low
                    interval_frame["interval_upper_95pct"] = high
                    interval_frame["interval_width"] = high - low
                    interval_frame["covered_by_interval"] = (y_true >= low) & (y_true <= high)
                    interval_frame["ensemble_member_count"] = len(members)
                    add("Plot_Reconstruction_Data", interval_frame)
                    if len(np.unique(interval_frame["interval_width"])) >= 3:
                        interval_frame["width_stratum"] = pd.qcut(interval_frame["interval_width"], q=min(6, len(np.unique(interval_frame["interval_width"]))), duplicates="drop")
                        add("Interval_Width_Summary", interval_frame.groupby("width_stratum", observed=False).agg(
                            n=("sample_index", "size"),
                            coverage=("covered_by_interval", "mean"),
                            mean_interval_width=("interval_width", "mean"),
                            mean_absolute_error=("absolute_error", "mean"),
                        ).reset_index())
                    member_df = pd.DataFrame(member_preds.T, columns=[f"member_{i+1}_prediction" for i in range(member_preds.shape[0])])
                    member_df.insert(0, "sample_index", np.arange(n))
                    add("Ensemble_Member_Predictions", member_df)
            except Exception:
                interval_frame = None
        if interval_frame is None:
            add("Plot_Reconstruction_Data", point)
    elif "conformal" in name:
        if X_train is not None and y_train is not None:
            try:
                train_pred = np.asarray(model.predict(X_train), dtype=float)
                y_train_arr = np.asarray(y_train, dtype=float)
                q = float(np.quantile(np.abs(y_train_arr - train_pred), 0.9))
                conformal = point.copy()
                conformal["conformal_lower_90pct"] = pred - q
                conformal["conformal_upper_90pct"] = pred + q
                conformal["conformal_interval_width"] = 2 * q
                conformal["covered_by_conformal_interval"] = (y_true >= conformal["conformal_lower_90pct"]) & (y_true <= conformal["conformal_upper_90pct"])
                conformal["train_absolute_residual_quantile_90"] = q
                add("Plot_Reconstruction_Data", conformal)
            except Exception:
                add("Plot_Reconstruction_Data", point)
        else:
            add("Plot_Reconstruction_Data", point)
    elif "robust loss" in name:
        deltas = np.linspace(0.1, np.quantile(abs_error, 0.95) + 1e-9, 40)
        add("Plot_Reconstruction_Data", pd.DataFrame({"huber_delta": deltas, "mean_huber_loss": [np.mean(np.where(abs_error <= d, 0.5 * abs_error**2, d * (abs_error - 0.5 * d))) for d in deltas]}))
    elif "worst-case" in name:
        cutoffs = np.quantile(abs_error, np.linspace(0.55, 0.99, 20))
        add("Plot_Reconstruction_Data", pd.DataFrame({"worst_sample_share_percent": np.linspace(45, 1, len(cutoffs)), "absolute_error_cutoff": cutoffs, "mean_error_in_worst_tail": [abs_error[abs_error >= c].mean() for c in cutoffs]}))
    elif "severity" in name:
        qs = [0, np.quantile(abs_error, 0.5), np.quantile(abs_error, 0.8), np.quantile(abs_error, 0.95), abs_error.max() + 1e-9]
        bins = np.unique(qs)
        labels = ["Low", "Moderate", "High", "Extreme"][: max(1, len(bins) - 1)]
        sev = pd.cut(abs_error, bins=bins, labels=labels, include_lowest=True)
        add("Plot_Reconstruction_Data", pd.DataFrame({"severity": pd.Series(sev).astype(str), "sample_index": np.arange(n), "absolute_error": abs_error}).groupby("severity", observed=False).agg(n=("sample_index", "size"), mean_absolute_error=("absolute_error", "mean")).reset_index())
    elif "sign transition" in name:
        signs = np.where(residual >= 0, "Positive/Under-predicted", "Negative/Over-predicted")
        add("Plot_Reconstruction_Data", pd.DataFrame({"current_residual_sign": signs[:-1], "next_residual_sign": signs[1:]}).value_counts().reset_index(name="count"))
    elif "percentile calibration" in name or "rank concordance" in name:
        add("Plot_Reconstruction_Data", point[["sample_index", "observed_percentile", "predicted_percentile", "observed", "predicted", "absolute_error"]])

    if isinstance(X_test, pd.DataFrame):
        numeric = X_test.select_dtypes(include=[np.number]).copy().reset_index(drop=True)
        if not numeric.empty:
            cols = numeric.columns.tolist()[:12]
            z = (numeric[cols] - numeric[cols].mean()) / (numeric[cols].std().replace(0, 1.0) + 1e-9)
            if "feature-error correlation" in name:
                rows = []
                for col in cols:
                    s = pd.to_numeric(numeric[col], errors="coerce")
                    mask = s.notna()
                    if mask.sum() > 5 and s[mask].nunique() > 2:
                        rows.append({"feature": col, "correlation_with_absolute_error": float(np.corrcoef(s[mask], abs_error[mask])[0, 1])})
                add("Plot_Reconstruction_Data", pd.DataFrame(rows))
            elif "hard-sample fingerprint" in name:
                worst = np.argsort(abs_error)[-min(20, n):]
                heat = z.iloc[worst].copy()
                heat.insert(0, "sample_index", worst)
                heat.insert(1, "absolute_error", abs_error[worst])
                add("Plot_Reconstruction_Data", heat)
            elif "easy-vs-hard" in name:
                hard = abs_error >= np.quantile(abs_error, 0.8)
                easy = abs_error <= np.quantile(abs_error, 0.2)
                add("Plot_Reconstruction_Data", pd.DataFrame({"feature": cols, "hard_minus_easy_mean_zscore": (z.loc[hard, cols].mean() - z.loc[easy, cols].mean()).to_numpy()}))
            elif "distance-to-center" in name:
                dist = np.sqrt(np.nanmean(np.square(z.to_numpy()), axis=1))
                seg = _safe_qcut_labels(dist, 8, "D")
                add("Plot_Reconstruction_Data", pd.DataFrame({"segment": seg, "distance_to_center": dist, "absolute_error": abs_error}).groupby("segment", observed=False).agg(mean_distance=("distance_to_center", "mean"), mean_absolute_error=("absolute_error", "mean"), n=("absolute_error", "size")).reset_index())
            elif "two-feature error surface" in name or "feature-binned residual heatmap" in name:
                if len(cols) >= 2:
                    f1, f2 = cols[0], cols[1]
                    grid = pd.DataFrame({"feature_1_bin": _safe_qcut_labels(numeric[f1], 8, "X"), "feature_2_bin": _safe_qcut_labels(numeric[f2], 8, "Y"), "absolute_error": abs_error, "residual": residual})
                    add("Plot_Reconstruction_Data", grid.groupby(["feature_1_bin", "feature_2_bin"], observed=False).agg(mean_absolute_error=("absolute_error", "mean"), mean_residual=("residual", "mean"), n=("absolute_error", "size")).reset_index())
            elif "feature outlier" in name:
                rows = []
                for col in cols:
                    s = numeric[col]
                    out = np.abs((s - s.mean()) / (s.std() + 1e-9)) > 2
                    if out.sum() and (~out).sum():
                        rows.append({"feature": col, "outlier_count": int(out.sum()), "non_outlier_count": int((~out).sum()), "outlier_mean_error": float(abs_error[out].mean()), "non_outlier_mean_error": float(abs_error[~out].mean()), "error_risk_ratio": float(abs_error[out].mean() / (abs_error[~out].mean() + 1e-9))})
                add("Plot_Reconstruction_Data", pd.DataFrame(rows))
            elif "conditional coverage" in name:
                tol = np.quantile(abs_error, 0.75)
                rows = []
                for col in cols:
                    seg = _safe_qcut_labels(numeric[col], 4, "S")
                    rates = pd.DataFrame({"segment": seg, "covered": abs_error <= tol}).groupby("segment", observed=False)["covered"].mean()
                    rows.append({"feature": col, "tolerance_absolute_error": tol, "min_coverage": rates.min(), "max_coverage": rates.max(), "coverage_spread": rates.max() - rates.min()})
                add("Plot_Reconstruction_Data", pd.DataFrame(rows))
            elif "input noise" in name:
                levels = [0.01, 0.03, 0.05, 0.08, 0.12]
                rows = []
                rng = np.random.default_rng(42)
                scale = numeric.std().replace(0, 1.0)
                for level in levels:
                    noisy = X_test.copy()
                    for col in numeric.columns:
                        noisy[col] = noisy[col] + rng.normal(0, scale[col] * level, len(noisy))
                    noisy_pred = np.asarray(model.predict(noisy), dtype=float)
                    rows.append({"feature_noise_level_percent_of_std": level * 100, "mean_prediction_perturbation": float(np.mean(np.abs(noisy_pred - pred)))})
                add("Plot_Reconstruction_Data", pd.DataFrame(rows))

    if len(sheets) == 1:
        sheets.append(("Plot_Reconstruction_Data", point))
    return sheets

# Store global state (For local app/single user)
SESSION_STATE = {
    'df': None,
    'original_df': None,
    'target_col': None,
    'problem_type': 'Regression',
    'selected_features': [],
    'best_model': None,
    'best_model_name': None,
    'X_train': None,
    'X_test': None,
    'y_train': None,
    'y_test': None,
    'preprocess_artifacts': {},
    'feature_columns': [],
    'base_feature_columns': [],
    'optimization_result': None,
    'optimization_export_package': None,
    'imported_models': {},
    'imported_optimizations': {},
    'meta_histories': {},
    'trained_models': {},
    'last_training_results': [],
    'last_training_config': {},
    'publication_evidence': {},
}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_excel(io.BytesIO(content))
        SESSION_STATE['df'] = df
        SESSION_STATE['original_df'] = df.copy()
        SESSION_STATE['target_col'] = None
        SESSION_STATE['problem_type'] = 'Regression'
        SESSION_STATE['selected_features'] = []
        SESSION_STATE['best_model'] = None
        SESSION_STATE['best_model_name'] = None
        SESSION_STATE['X_train'] = None
        SESSION_STATE['X_test'] = None
        SESSION_STATE['y_train'] = None
        SESSION_STATE['y_test'] = None
        SESSION_STATE['trained_models'] = {}
        SESSION_STATE['preprocess_artifacts'] = {}
        SESSION_STATE['feature_columns'] = []
        SESSION_STATE['base_feature_columns'] = []
        SESSION_STATE['optimization_result'] = None
        SESSION_STATE['optimization_export_package'] = None
        SESSION_STATE['imported_models'] = {}
        SESSION_STATE['imported_optimizations'] = {}
        SESSION_STATE['meta_histories'] = {}
        SESSION_STATE['last_training_results'] = []
        SESSION_STATE['last_training_config'] = {}
        SESSION_STATE['publication_evidence'] = {}
        
        info = {
            "rows": df.shape[0],
            "cols": df.shape[1],
            "columns": df.columns.tolist(),
            "missing_total": int(df.isna().sum().sum()),
            "dtypes": {k: str(v) for k, v in df.dtypes.items()}
        }
        return {"status": "success", "data": info}
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})

@app.post("/api/set_target")
async def set_target(request: Request):
    data = await request.json()
    target_col = data.get("target_col")
    
    df = SESSION_STATE['df']
    if df is None or target_col not in df.columns:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Dataset not loaded or target col invalid."})
        
    SESSION_STATE['target_col'] = target_col
    if df[target_col].dtype == 'object' or df[target_col].nunique() <= 15:
        ptype = "Classification"
    else:
        ptype = "Regression"
        
    SESSION_STATE['problem_type'] = ptype
    return {"status": "success", "problem_type": ptype}

@app.post("/api/set_problem_type")
async def set_problem_type(request: Request):
    data = await request.json()
    problem_type = data.get("problem_type", "Regression")
    if problem_type not in {"Regression", "Classification"}:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid problem type."})

    SESSION_STATE['problem_type'] = problem_type
    return {"status": "success", "problem_type": problem_type}

import core.eda as eda_module

@app.post("/api/eda")
async def generate_eda(request: Request):
    data = await request.json()
    df = SESSION_STATE['df']
    if df is None: 
        return JSONResponse(status_code=400, content={"status":"error", "message":"Dataset missing"})
    
    pt = data.get("plot_type", "hist")
    fmt = data.get("format", "png")
    dpi = data.get("dpi", 300)
    col_x = data.get("col_x")
    col_y = data.get("col_y")
    target_col = SESSION_STATE.get("target_col")
    
    try:
        if pt in ["hist", "box", "violin", "density", "ecdf", "strip"]:
            img = eda_module.get_plot_distribution(df, col_x, pt, fmt, dpi)
        elif pt in ["scatter", "reg", "hex", "joint"]:
            img = eda_module.get_plot_scatter(df, col_x, col_y, fmt, dpi, pt, target_col)
        elif pt == "corr":
            img = eda_module.get_plot_correlation(df, fmt, dpi, "pearson")
        elif pt == "corr_spearman":
            img = eda_module.get_plot_correlation(df, fmt, dpi, "spearman")
        elif pt == "pair":
            img = eda_module.get_plot_pairplot(df, fmt, dpi, target_col)
        elif pt == "missing":
            img = eda_module.get_plot_missingness(df, fmt, dpi)
        elif pt == "target":
            if not target_col:
                return {"status":"error", "message":"Set a target column first."}
            img = eda_module.get_plot_target_relation(df, col_x, target_col, fmt, dpi)
        else:
            return {"status":"error", "message":"Unknown geometry mapping"}
            
        return {"status": "success", "image": img}
    except Exception as e:
        return {"status": "error", "message": f"EDA Engine Error: {str(e)}"}


def _append_zip_item(zip_file, name, data_url):
    mime, ext, payload = _decode_data_url(data_url)
    zip_file.writestr(f"{_slugify_filename(name)}.{ext}", payload)


@app.post("/api/eda/all")
async def generate_all_eda(request: Request):
    data = await request.json()
    df = SESSION_STATE['df']
    if df is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Dataset missing"})

    fmt = data.get("format", "png")
    dpi = int(data.get("dpi", 300))
    target_col = SESSION_STATE.get("target_col")
    columns = df.columns.tolist()
    feature_columns = [col for col in columns if col != target_col] or columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [col for col in numeric_columns if col != target_col]

    items = []
    failures = []

    def collect(name, builder):
        try:
            image = builder()
            if image:
                items.append({"name": name, "data_url": image})
        except Exception as exc:
            failures.append({"name": name, "message": str(exc)})

    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            plot_types = ["hist", "density", "ecdf", "box", "violin", "strip"]
        else:
            plot_types = ["hist"]
        for plot_type in plot_types:
            collect(
                f"EDA_{plot_type}_{col}",
                lambda col=col, plot_type=plot_type: eda_module.get_plot_distribution(df, col, plot_type, fmt, dpi),
            )

    feature_df = df[feature_columns]
    if len(numeric_features) >= 2:
        for method, plot_type in [("pearson", "corr"), ("spearman", "corr_spearman")]:
            collect(
                f"EDA_{plot_type}_all_numeric_features",
                lambda method=method: eda_module.get_plot_correlation(feature_df, fmt, dpi, method),
            )

        collect("EDA_pairplot_all_numeric_features", lambda: eda_module.get_plot_pairplot(df, fmt, dpi, target_col, max_cols=None))
    if df[feature_columns].isna().sum().sum() > 0:
        collect("EDA_missingness_all_features", lambda: eda_module.get_plot_missingness(feature_df, fmt, dpi))

    if target_col:
        for col in feature_columns:
            collect(
                f"EDA_target_{col}_vs_{target_col}",
                lambda col=col: eda_module.get_plot_target_relation(df, col, target_col, fmt, dpi),
            )

    for col_x, col_y in combinations(numeric_features, 2):
        for plot_type in ["scatter", "reg", "hex", "joint"]:
            collect(
                f"EDA_{plot_type}_{col_x}_vs_{col_y}",
                lambda col_x=col_x, col_y=col_y, plot_type=plot_type: eda_module.get_plot_scatter(
                    df, col_x, col_y, fmt, dpi, plot_type, target_col
                ),
            )

    if not items:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "No EDA plots could be generated.", "failures": failures},
        )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in items:
            _append_zip_item(zf, item["name"], item["data_url"])
        if failures:
            report = "\n".join(f"{item['name']}: {item['message']}" for item in failures)
            zf.writestr("EDA_generation_failures.txt", report)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {
        "status": "success",
        "filename": "EDA_All_Feature_Plots.zip",
        "content_base64": encoded,
        "count": len(items),
        "failed_count": len(failures),
        "failures": failures[:30],
    }

import core.preprocessing as pre_module

@app.post("/api/preprocess/outliers")
async def do_outliers(request: Request):
    data = await request.json()
    method = data.get("method", "Isolation Forest")
    contam = data.get("contamination", 0.05)
    df = SESSION_STATE.get('df')
    fmt = data.get("format", "png")
    dpi = data.get("dpi", 300)
    if df is None: return JSONResponse({"status": "error", "message": "No data available."})
    df_before = df.copy(deep=True)
    new_df, count = pre_module.apply_outlier_removal(df_before, method, contam)
    SESSION_STATE['df'] = new_df
    try:
        process_name = f"Outliers Filter ({method})"
        SESSION_STATE['preprocess_artifacts']['outliers'] = {
            "df_old": df_before,
            "df_new": new_df.copy(deep=True),
            "process_name": process_name
        }
        plot_b64 = eda_module.get_preprocessing_plot(df_before, new_df, process_name, SESSION_STATE['target_col'], fmt, dpi)
    except: plot_b64 = None
    return {"status": "success", "removed": count, "rows": len(new_df), "plot": plot_b64}

@app.post("/api/preprocess/scale")
async def do_scale(request: Request):
    data = await request.json()
    method = data.get("method", "StandardScaler")
    df = SESSION_STATE['df']
    fmt = data.get("format", "png")
    dpi = data.get("dpi", 300)
    df_before = df.copy(deep=True)
    new_df = pre_module.apply_scaling(df_before.copy(deep=True), SESSION_STATE['target_col'], method)
    SESSION_STATE['df'] = new_df
    try:
        process_name = f"Scaling ({method})"
        SESSION_STATE['preprocess_artifacts']['scale'] = {
            "df_old": df_before,
            "df_new": new_df.copy(deep=True),
            "process_name": process_name
        }
        plot_b64 = eda_module.get_preprocessing_plot(df_before, new_df, process_name, SESSION_STATE['target_col'], fmt, dpi)
    except: plot_b64 = None
    return {"status": "success", "message": f"{method} applied", "plot": plot_b64}

@app.post("/api/preprocess/augment")
async def do_augment(request: Request):
    data = await request.json()
    method = data.get("method", "SMOTE")
    df = SESSION_STATE['df']
    fmt = data.get("format", "png")
    dpi = data.get("dpi", 300)
    try:
        df_before = df.copy(deep=True)
        new_df = pre_module.apply_augmentation(df_before, SESSION_STATE['target_col'], method, SESSION_STATE['problem_type'])
        SESSION_STATE['df'] = new_df
        try:
            process_name = f"Augmentation ({method})"
            SESSION_STATE['preprocess_artifacts']['augment'] = {
                "df_old": df_before,
                "df_new": new_df.copy(deep=True),
                "process_name": process_name
            }
            plot_b64 = eda_module.get_preprocessing_plot(df_before, new_df, process_name, SESSION_STATE['target_col'], fmt, dpi)
        except: plot_b64 = None
        return {"status": "success", "message": "Augmented", "rows": len(new_df), "plot": plot_b64}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/preprocess/features")
async def do_features(request: Request):
    data = await request.json()
    method = data.get("method", "Random Forest")
    top_k = int(data.get("top_k", 10))
    df = SESSION_STATE['df']
    fmt = data.get("format", "png")
    dpi = data.get("dpi", 300)
    try:
        imps = pre_module.get_feature_importances(df, SESSION_STATE['target_col'], method, SESSION_STATE['problem_type'], top_k)
        selected_cols = [x['Feature'] for x in imps]
        SESSION_STATE['selected_features'] = selected_cols
        try:
            process_name = f"Feature Map ({method})"
            SESSION_STATE['preprocess_artifacts']['features'] = {
                "df_old": df.copy(deep=True),
                "df_new": imps,
                "process_name": process_name
            }
            plot_b64 = eda_module.get_preprocessing_plot(df, imps, process_name, SESSION_STATE['target_col'], fmt, dpi)
        except: plot_b64 = None
        return {"status": "success", "importances": imps, "plot": plot_b64}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/preprocess/plot")
async def regenerate_preprocess_plot(request: Request):
    data = await request.json()
    plot_key = data.get("plot_key")
    fmt = data.get("format", "png")
    dpi = data.get("dpi", 300)

    artifact = SESSION_STATE.get("preprocess_artifacts", {}).get(plot_key)
    if artifact is None:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Preprocessing plot source not found."})

    try:
        plot_b64 = eda_module.get_preprocessing_plot(
            artifact["df_old"],
            artifact["df_new"],
            artifact["process_name"],
            SESSION_STATE['target_col'],
            fmt,
            dpi
        )
        return {"status": "success", "plot": plot_b64}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/preprocess/augment_diagnostics")
async def get_augmentation_diagnostics(request: Request):
    data = await request.json()
    fmt = data.get("format", "png")
    dpi = int(data.get("dpi", 300))
    artifact = SESSION_STATE.get("preprocess_artifacts", {}).get("augment")
    if artifact is None:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Run data augmentation first."})
    try:
        plots = pre_module.generate_augmentation_diagnostics(
            artifact["df_old"],
            artifact["df_new"],
            SESSION_STATE.get("target_col"),
            SESSION_STATE.get("problem_type", "Regression"),
            fmt,
            dpi,
        )
        SESSION_STATE["preprocess_artifacts"]["augment"]["diagnostic_plots"] = plots
        return {"status": "success", "plots": plots}
    except Exception as e:
        return {"status": "error", "message": f"Augmentation diagnostics failed: {str(e)}"}

import core.modeling as mod_module
import core.modeling_plots as mod_plots
import core.model_io as model_io
from core.plotting_utils import NATURE_COLORS, apply_nature_style, fig_to_base64


def _score_predictions(y_true, pred, problem_type):
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true)
    pred = np.asarray(pred)
    if str(problem_type).lower() == "classification":
        return {
            "Accuracy": float(accuracy_score(y_true, pred)),
            "F1_weighted": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
            "N": int(len(y_true)),
        }
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    return {
        "R2": float(r2_score(y_true, pred)),
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(y_true, pred)),
        "N": int(len(y_true)),
    }


def _publication_bar_plot(frame, x_col, y_col, title, fmt="png", dpi=300):
    import matplotlib.pyplot as plt

    apply_nature_style()
    fig, ax = plt.subplots(figsize=(7.2, max(3.8, 0.34 * len(frame) + 1.4)))
    colors = [NATURE_COLORS["red"] if float(v) < 0 else NATURE_COLORS["blue"] for v in frame[y_col]]
    ax.barh(frame[x_col].astype(str), frame[y_col].astype(float), color=colors)
    ax.axvline(0, color=NATURE_COLORS["slate"], linestyle="--", linewidth=1)
    ax.set_xlabel(y_col)
    ax.set_title(title)
    return fig_to_base64(fig, fmt, dpi)


def _publication_scatter_plot(y_true, pred, title, fmt="png", dpi=300):
    import matplotlib.pyplot as plt

    apply_nature_style()
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.scatter(y_true, pred, s=34, alpha=0.74, color=NATURE_COLORS["blue"], edgecolors="white", linewidth=0.35)
    lo, hi = min(np.min(y_true), np.min(pred)), max(np.max(y_true), np.max(pred))
    ax.plot([lo, hi], [lo, hi], color=NATURE_COLORS["red"], linestyle="--")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    return fig_to_base64(fig, fmt, dpi)

@app.get("/api/models")
async def get_models():
    ptype = SESSION_STATE.get('problem_type', 'Regression')
    return {"status": "success", "models": mod_module.get_model_registry(ptype)}

@app.get("/api/model_packages")
async def get_model_packages():
    return {
        "status": "success",
        "trained": list(SESSION_STATE.get("trained_models", {}).keys()),
        "imported": list(SESSION_STATE.get("imported_models", {}).keys())
    }

@app.post("/api/models/export")
async def export_model_package(request: Request):
    data = await request.json()
    model_name = data.get("model_name")
    model = SESSION_STATE.get("trained_models", {}).get(model_name)
    if model is None:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Model not found in trained registry."})

    metrics = next((row for row in SESSION_STATE.get("last_training_results", []) if row.get("Model") == model_name), {})
    package = model_io.build_model_package(
        model_name=model_name,
        model=model,
        problem_type=SESSION_STATE.get("problem_type"),
        feature_columns=SESSION_STATE.get("feature_columns", []),
        selected_features=SESSION_STATE.get("selected_features", []),
        target_col=SESSION_STATE.get("target_col"),
        base_feature_columns=SESSION_STATE.get("base_feature_columns", []),
        metrics=metrics,
        training_config=SESSION_STATE.get("last_training_config", {}),
        meta_history=SESSION_STATE.get("meta_histories", {}).get(model_name),
    )
    blob_b64 = model_io.dump_model_package(package)
    return {"status": "success", "filename": f"{model_name.replace(' ', '_')}.naturemlmodel", "content_base64": blob_b64}

@app.post("/api/models/import")
async def import_model_package(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        package = model_io.load_model_package_from_bytes(raw)
        if package.get("package_type") == "natureml_multiobjective_optimization":
            opt_name = package.get("name") or file.filename or "ImportedOptimization"
            SESSION_STATE["imported_optimizations"][opt_name] = package
            return {
                "status": "success",
                "model_name": opt_name,
                "package_type": package.get("package_type"),
                "target_col": package.get("dataset_context", {}).get("target_col"),
            }
        model_name = package.get("model_name") or file.filename or "ImportedModel"
        SESSION_STATE["imported_models"][model_name] = package
        return {
            "status": "success",
            "model_name": model_name,
            "package_type": package.get("package_type", "natureml_model"),
            "problem_type": package.get("problem_type"),
            "target_col": package.get("target_col"),
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": f"Import failed: {str(e)}"})

@app.post("/api/train")
async def train_models(request: Request):
    data = await request.json()
    selected_models = data.get("models", [])
    use_meta = data.get("use_meta", False)
    meta_algo = data.get("meta_algo", "PSO")
    meta_algos = data.get("meta_algos")
    ensemble = data.get("ensemble", "None")
    test_size = data.get("test_size", 0.2)
    
    df = SESSION_STATE.get('df')
    if df is None: return JSONResponse({"status": "error", "message": "No data available."})
    
    try:
        null_count = int(df.isna().sum().sum())
        inf_count = int(np.isinf(df.select_dtypes(include=[np.number]).to_numpy()).sum()) if not df.select_dtypes(include=[np.number]).empty else 0
        results, payload = mod_module.run_training_pipeline(
            df, SESSION_STATE['target_col'], SESSION_STATE['selected_features'],
            SESSION_STATE['problem_type'], selected_models, use_meta, meta_algo, ensemble, test_size, meta_algos=meta_algos
        )
        trained_models = payload.pop('trained_models', {})
        failed_models = payload.get('failed_models', [])
        SESSION_STATE['trained_models'] = trained_models
        for k, v in payload.items(): SESSION_STATE[k] = v
        SESSION_STATE['last_training_results'] = results
        SESSION_STATE['last_training_config'] = {
            "selected_models": selected_models,
            "use_meta": use_meta,
            "meta_algo": meta_algo,
            "meta_algos": meta_algos,
            "ensemble": ensemble,
            "test_size": test_size,
            "format": data.get("format", "png"),
            "dpi": data.get("dpi", 300),
        }
        fmt = data.get("format", "png")
        dpi = data.get("dpi", 300)
        # Try automatic plotting for the best model as default
        try:
            plots = mod_plots.generate_model_plots(
                payload['best_model'],
                payload['X_train'],
                payload['y_train'],
                payload['X_test'],
                payload['y_test'],
                SESSION_STATE['problem_type'],
                fmt,
                dpi,
                meta_history=payload.get('meta_histories', {}).get(payload['best_model_name']),
            )
        except Exception as e:
            plots = {}
        return {
            "status": "success",
            "results": results,
            "best": payload['best_model_name'],
            "plots": plots,
            "failed_models": failed_models,
        }
    except Exception as e:
        return {"status": "error", "message": f"Training Kernel Error: {str(e)} | dataset NaN={null_count}, dataset Inf={inf_count}"}

@app.post("/api/plot_global")
async def plot_global_comparisons(request: Request):
    try:
        data = await request.json()
        fmt = data.get("format", "png")
        dpi = data.get("dpi", 300)
        min_score = data.get("min_score")
        filter_enabled = bool(data.get("filter_enabled", False))
        trained_models = SESSION_STATE.get('trained_models', {})
        if not trained_models: return JSONResponse({"status": "error", "message": "No trained models found."})
        plots = mod_plots.generate_global_plots(
            trained_models, SESSION_STATE.get('X_train'), SESSION_STATE.get('y_train'), 
            SESSION_STATE.get('X_test'), SESSION_STATE.get('y_test'), SESSION_STATE.get('problem_type'),
            fmt, dpi, min_score=min_score, filter_enabled=filter_enabled,
            meta_histories=SESSION_STATE.get("meta_histories", {})
        )
        return {"status": "success", "plots": plots}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/plot_model")
async def plot_model_ondemand(req: Request):
    try:
        payload = await req.json()
        m_name = payload.get("model_name")
        model = SESSION_STATE.get("trained_models", {}).get(m_name)
        X_train, y_train = SESSION_STATE.get("X_train"), SESSION_STATE.get("y_train")
        X_test, y_test = SESSION_STATE.get("X_test"), SESSION_STATE.get("y_test")
        ptype = SESSION_STATE.get("problem_type")
        
        fmt = payload.get("format", "png")
        dpi = payload.get("dpi", 300)
        interval_bins = payload.get("interval_bins", 4)
        interval_edges = payload.get("interval_edges")
        meta_history = SESSION_STATE.get("meta_histories", {}).get(m_name)
        
        if model is None or X_test is None:
            return {"status": "error", "message": "Model or dataset not found in memory."}
            
        plots = mod_plots.generate_model_plots(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            ptype,
            fmt,
            dpi,
            interval_bins=interval_bins,
            interval_edges=interval_edges,
            meta_history=meta_history,
        )
        return {"status": "success", "plots": plots}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/publication/leakage_audit")
async def publication_leakage_audit(request: Request):
    data = await request.json()
    fmt = data.get("format", "png")
    dpi = int(data.get("dpi", 300))
    df = SESSION_STATE.get("df")
    target = SESSION_STATE.get("target_col")
    if df is None or not target:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Dataset and target must be available."})
    try:
        feature_df = df.drop(columns=[target], errors="ignore")
        exact_duplicate_rows = int(df.duplicated().sum())
        feature_duplicate_rows = int(feature_df.duplicated().sum())
        target_tokens = [t for t in re.split(r"[_\W]+", str(target).lower()) if len(t) >= 3]
        name_risks = []
        for col in feature_df.columns:
            low = str(col).lower()
            if str(target).lower() in low or any(tok in low for tok in target_tokens):
                name_risks.append({"feature": col, "risk": "feature name overlaps target terminology"})

        numeric = df.select_dtypes(include=[np.number])
        corr_rows = []
        if target in numeric.columns:
            for col in numeric.columns:
                if col == target:
                    continue
                vals = numeric[[col, target]].dropna()
                if len(vals) > 4 and vals[col].nunique() > 1 and vals[target].nunique() > 1:
                    corr = float(vals[col].corr(vals[target]))
                    corr_rows.append({"feature": col, "target_correlation": corr, "abs_target_correlation": abs(corr), "risk": "high" if abs(corr) >= 0.95 else "review" if abs(corr) >= 0.85 else "low"})
        corr_table = pd.DataFrame(corr_rows).sort_values("abs_target_correlation", ascending=False).head(30) if corr_rows else pd.DataFrame()

        train_test_overlap = 0
        X_train, X_test = SESSION_STATE.get("X_train"), SESSION_STATE.get("X_test")
        if isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame):
            train_hash = set(pd.util.hash_pandas_object(X_train.reset_index(drop=True), index=False).astype(str))
            test_hash = set(pd.util.hash_pandas_object(X_test.reset_index(drop=True), index=False).astype(str))
            train_test_overlap = len(train_hash.intersection(test_hash))

        summary = pd.DataFrame(
            [
                {"check": "Exact duplicate full rows", "value": exact_duplicate_rows, "severity": "review" if exact_duplicate_rows else "ok"},
                {"check": "Duplicate feature rows", "value": feature_duplicate_rows, "severity": "review" if feature_duplicate_rows else "ok"},
                {"check": "Train-test duplicate encoded feature rows", "value": train_test_overlap, "severity": "high" if train_test_overlap else "ok"},
                {"check": "Feature names overlapping target", "value": len(name_risks), "severity": "review" if name_risks else "ok"},
                {"check": "Numeric features with |corr(target)| >= 0.95", "value": int((corr_table.get("abs_target_correlation", pd.Series(dtype=float)) >= 0.95).sum()) if not corr_table.empty else 0, "severity": "high" if (not corr_table.empty and (corr_table["abs_target_correlation"] >= 0.95).any()) else "ok"},
            ]
        )
        plots = {}
        if not corr_table.empty:
            plot_df = corr_table.sort_values("abs_target_correlation").tail(15)
            plots["Leakage Correlation Sentinel"] = _publication_bar_plot(plot_df, "feature", "abs_target_correlation", "Leakage Correlation Sentinel", fmt, dpi)
        evidence = {
            "leakage_summary": summary.to_dict("records"),
            "target_correlation_table": corr_table.to_dict("records"),
            "name_risks": name_risks,
            "plots": plots,
        }
        SESSION_STATE.setdefault("publication_evidence", {})["leakage"] = evidence
        return {"status": "success", "data": evidence}
    except Exception as e:
        return {"status": "error", "message": f"Leakage audit failed: {str(e)}"}


@app.post("/api/publication/ablation")
async def publication_ablation(request: Request):
    from sklearn.base import clone

    data = await request.json()
    model_name = data.get("model_name") or SESSION_STATE.get("best_model_name")
    top_k = int(data.get("top_k", 10))
    fmt = data.get("format", "png")
    dpi = int(data.get("dpi", 300))
    model = SESSION_STATE.get("trained_models", {}).get(model_name)
    X_train, y_train = SESSION_STATE.get("X_train"), SESSION_STATE.get("y_train")
    X_test, y_test = SESSION_STATE.get("X_test"), SESSION_STATE.get("y_test")
    ptype = SESSION_STATE.get("problem_type")
    if model is None or X_train is None or X_test is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Train a model before ablation."})
    try:
        baseline_pred = model.predict(X_test)
        baseline = _score_predictions(y_test, baseline_pred, ptype)
        score_key = "Accuracy" if str(ptype).lower() == "classification" else "R2"
        baseline_score = baseline[score_key]
        columns = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else [f"x{i}" for i in range(np.asarray(X_train).shape[1])]

        if hasattr(model, "feature_importances_") and len(getattr(model, "feature_importances_")) == len(columns):
            importance = pd.DataFrame({"feature": columns, "importance": np.asarray(model.feature_importances_, dtype=float)})
        else:
            y_arr = np.asarray(y_train)
            rows = []
            for col in columns:
                vals = pd.Series(X_train[col] if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)[:, columns.index(col)])
                if vals.nunique() > 1 and np.std(y_arr) > 1e-12:
                    rows.append({"feature": col, "importance": abs(float(np.corrcoef(vals, y_arr)[0, 1]))})
            importance = pd.DataFrame(rows)
        if importance.empty:
            importance = pd.DataFrame({"feature": columns[:top_k], "importance": np.ones(min(top_k, len(columns)))})
        features = importance.sort_values("importance", ascending=False)["feature"].head(top_k).tolist()

        rows = [{"ablation": "Baseline", "removed_feature": "", score_key: baseline_score, "delta_vs_baseline": 0.0, **baseline}]
        for feature in features:
            keep_cols = [c for c in columns if c != feature]
            if not keep_cols:
                continue
            try:
                candidate = clone(model)
                candidate.fit(X_train[keep_cols], y_train)
                pred = candidate.predict(X_test[keep_cols])
                metrics = _score_predictions(y_test, pred, ptype)
                rows.append({"ablation": f"Remove {feature}", "removed_feature": feature, score_key: metrics[score_key], "delta_vs_baseline": float(metrics[score_key] - baseline_score), **metrics})
            except Exception as exc:
                rows.append({"ablation": f"Remove {feature}", "removed_feature": feature, score_key: np.nan, "delta_vs_baseline": np.nan, "error": str(exc)})
        table = pd.DataFrame(rows)
        plot_df = table[table["ablation"] != "Baseline"].dropna(subset=["delta_vs_baseline"]).sort_values("delta_vs_baseline")
        plots = {}
        if not plot_df.empty:
            plots["Ablation Performance Drop"] = _publication_bar_plot(plot_df, "removed_feature", "delta_vs_baseline", "Ablation Performance Drop", fmt, dpi)
        evidence = {"model_name": model_name, "score_key": score_key, "baseline": baseline, "table": table.to_dict("records"), "plots": plots}
        SESSION_STATE.setdefault("publication_evidence", {})["ablation"] = evidence
        return {"status": "success", "data": evidence}
    except Exception as e:
        return {"status": "error", "message": f"Ablation failed: {str(e)}"}


@app.post("/api/publication/external_validate")
async def publication_external_validate(file: UploadFile = File(...), model_name: str = None, format: str = "png", dpi: int = 300):
    model_name = model_name or SESSION_STATE.get("best_model_name")
    model = SESSION_STATE.get("trained_models", {}).get(model_name)
    target = SESSION_STATE.get("target_col")
    if model is None or not target:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Train a model and set target before external validation."})
    try:
        raw = await file.read()
        if str(file.filename or "").lower().endswith(".csv"):
            ext_df = pd.read_csv(io.BytesIO(raw))
        else:
            ext_df = pd.read_excel(io.BytesIO(raw))
        if target not in ext_df.columns:
            return JSONResponse(status_code=400, content={"status": "error", "message": f"External dataset must contain target column: {target}"})
        X_ext = mod_module.prepare_design_matrix(ext_df, target, SESSION_STATE.get("selected_features"), SESSION_STATE.get("feature_columns"))
        y_ext = ext_df[target]
        pred = model.predict(X_ext)
        metrics = _score_predictions(y_ext, pred, SESSION_STATE.get("problem_type"))
        result_table = _series_with_prediction(model, X_ext, y_ext, "external")
        plots = {}
        if str(SESSION_STATE.get("problem_type")).lower() == "regression":
            plots["External Validation Observed vs Predicted"] = _publication_scatter_plot(np.asarray(y_ext, dtype=float), np.asarray(pred, dtype=float), "External Validation Observed vs Predicted", format, int(dpi))
        evidence = {
            "model_name": model_name,
            "filename": file.filename,
            "metrics": metrics,
            "rows": int(len(ext_df)),
            "columns": ext_df.columns.tolist(),
            "prediction_table": result_table.to_dict("records"),
            "plots": plots,
        }
        SESSION_STATE.setdefault("publication_evidence", {})["external_validation"] = evidence
        return {"status": "success", "data": evidence}
    except Exception as e:
        return {"status": "error", "message": f"External validation failed: {str(e)}"}


@app.get("/api/publication/results_text")
async def publication_results_text():
    results = SESSION_STATE.get("last_training_results", [])
    best = SESSION_STATE.get("best_model_name")
    ptype = SESSION_STATE.get("problem_type")
    evidence = SESSION_STATE.get("publication_evidence", {})
    lines = []
    if results:
        best_row = next((r for r in results if r.get("Model") == best), results[0])
        metric_bits = ", ".join(f"{k}={v:.4f}" for k, v in best_row.items() if isinstance(v, (int, float)) and not pd.isna(v))
        lines.append(f"The best-performing model was {best_row.get('Model', best)}, with {metric_bits} on the held-out test set for the {ptype.lower()} task.")
    if evidence.get("external_validation"):
        metrics = evidence["external_validation"].get("metrics", {})
        metric_bits = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float)))
        lines.append(f"External validation was performed on {evidence['external_validation'].get('rows')} independent samples, yielding {metric_bits}.")
    if evidence.get("leakage"):
        high = [r for r in evidence["leakage"].get("leakage_summary", []) if r.get("severity") == "high" and r.get("value", 0)]
        lines.append("Data leakage screening identified no high-severity leakage sentinel." if not high else f"Data leakage screening flagged {len(high)} high-severity sentinel(s), which should be resolved before manuscript submission.")
    if evidence.get("ablation"):
        rows = [r for r in evidence["ablation"].get("table", []) if r.get("removed_feature")]
        rows = sorted(rows, key=lambda r: r.get("delta_vs_baseline", 0))
        if rows:
            lines.append(f"Ablation analysis showed the strongest performance decrease after removing {rows[0].get('removed_feature')}, supporting its importance for model performance.")
    if not lines:
        lines.append("Run model training and publication evidence modules to generate manuscript-ready results text.")
    return {"status": "success", "data": {"paragraphs": lines, "text": "\n\n".join(lines)}}


@app.get("/api/publication/storyboard")
async def publication_storyboard():
    groups = [
        {"figure": "Fig. 1", "title": "Dataset and problem definition", "recommended_panels": ["Data audit summary", "Target distribution", "Correlation heatmap", "Missingness map"]},
        {"figure": "Fig. 2", "title": "Model performance and external validation", "recommended_panels": ["Model Leaderboard", "Prediction vs True", "External Validation Observed vs Predicted", "Ablation Performance Drop"]},
        {"figure": "Fig. 3", "title": "Generalization, overfitting, and reliability", "recommended_panels": ["Train vs Test R2", "Learning Curve", "Bootstrap Stability", "Applicability Domain Reliability"]},
        {"figure": "Fig. 4", "title": "Mechanistic interpretation and feature effects", "recommended_panels": ["Feature Importance", "SHAP Dependence Panel", "One-Factor Response Sensitivity Grid", "Feature-Binned Residual Heatmap"]},
        {"figure": "Fig. 5", "title": "Optimization or design validation", "recommended_panels": ["Pareto Front", "Meta Convergence Comparison", "Counterfactual Feature Shifts", "Error Tolerance Utility Curve"]},
        {"figure": "Supplementary", "title": "Full diagnostic atlas", "recommended_panels": ["All categorized model diagnostics with plot reconstruction data"]},
    ]
    return {"status": "success", "data": groups}


@app.get("/api/publication/reproducibility_package")
async def publication_reproducibility_package():
    try:
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            metadata = {
                "target_col": SESSION_STATE.get("target_col"),
                "problem_type": SESSION_STATE.get("problem_type"),
                "best_model_name": SESSION_STATE.get("best_model_name"),
                "feature_columns": SESSION_STATE.get("feature_columns"),
                "selected_features": SESSION_STATE.get("selected_features"),
                "training_config": SESSION_STATE.get("last_training_config"),
            }
            zf.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2, default=str))
            zf.writestr("model_leaderboard.json", json.dumps(SESSION_STATE.get("last_training_results", []), ensure_ascii=False, indent=2, default=str))
            zf.writestr("publication_evidence.json", json.dumps(_strip_data_urls(SESSION_STATE.get("publication_evidence", {})), ensure_ascii=False, indent=2, default=str))
            for name, obj in [
                ("current_dataset.csv", SESSION_STATE.get("df")),
                ("X_train.csv", SESSION_STATE.get("X_train")),
                ("X_test.csv", SESSION_STATE.get("X_test")),
                ("y_train.csv", _to_frame(SESSION_STATE.get("y_train"), "target")),
                ("y_test.csv", _to_frame(SESSION_STATE.get("y_test"), "target")),
            ]:
                frame = _to_frame(obj)
                if not frame.empty:
                    zf.writestr(name, frame.to_csv(index=False))
            text = (await publication_results_text())["data"]["text"]
            zf.writestr("manuscript_results_draft.txt", text)
        encoded = base64.b64encode(output.getvalue()).decode("utf-8")
        return {"status": "success", "filename": "NatureML_reproducibility_package.zip", "content_base64": encoded}
    except Exception as e:
        return {"status": "error", "message": f"Reproducibility package failed: {str(e)}"}


@app.post("/api/publication/advanced_discovery")
async def publication_advanced_discovery(request: Request):
    from sklearn.base import clone
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.metrics import accuracy_score, make_scorer, r2_score
    from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, cross_val_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    data = await request.json()
    model_name = data.get("model_name") or SESSION_STATE.get("best_model_name")
    fmt = data.get("format", "png")
    dpi = int(data.get("dpi", 300))
    top_k = int(data.get("top_k", 12))
    model = SESSION_STATE.get("trained_models", {}).get(model_name)
    X_train, y_train = SESSION_STATE.get("X_train"), SESSION_STATE.get("y_train")
    X_test, y_test = SESSION_STATE.get("X_test"), SESSION_STATE.get("y_test")
    ptype = SESSION_STATE.get("problem_type")
    if model is None or X_train is None or X_test is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Train a model before advanced discovery."})

    try:
        plots = {}
        tables = {}
        X_all = pd.concat([_to_frame(X_train), _to_frame(X_test)], ignore_index=True)
        y_all = pd.concat([pd.Series(y_train), pd.Series(y_test)], ignore_index=True)

        # 1. Repeated validation score distribution.
        try:
            if str(ptype).lower() == "classification":
                cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
                scoring = make_scorer(accuracy_score)
                score_name = "RepeatedCV_Accuracy"
            else:
                cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
                scoring = make_scorer(r2_score)
                score_name = "RepeatedCV_R2"
            scores = cross_val_score(clone(model), X_all, y_all, cv=cv, scoring=scoring, error_score=np.nan)
            cv_table = pd.DataFrame({"fold": np.arange(1, len(scores) + 1), score_name: scores})
            tables["Repeated Validation Scores"] = cv_table.to_dict("records")
            apply_nature_style()
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7.2, 4.6))
            ax.plot(cv_table["fold"], cv_table[score_name], marker="o", color=NATURE_COLORS["blue"], linewidth=1.8)
            ax.axhline(np.nanmean(scores), color=NATURE_COLORS["red"], linestyle="--", label=f"mean={np.nanmean(scores):.3f}")
            ax.fill_between(cv_table["fold"], np.nanmean(scores) - np.nanstd(scores), np.nanmean(scores) + np.nanstd(scores), color=NATURE_COLORS["blue"], alpha=0.12, label="mean +/- SD")
            ax.set_xlabel("Repeated CV fold")
            ax.set_ylabel(score_name)
            ax.set_title("Repeated Nested-Style Validation Stability")
            ax.legend()
            plots["Repeated Validation Stability"] = fig_to_base64(fig, fmt, dpi)
        except Exception as exc:
            tables["Repeated Validation Scores"] = [{"error": str(exc)}]

        pred = np.asarray(model.predict(X_test))
        if str(ptype).lower() == "regression":
            residual = np.asarray(y_test, dtype=float) - np.asarray(pred, dtype=float)
            abs_err = np.abs(residual)
        else:
            residual = (np.asarray(pred) != np.asarray(y_test)).astype(float)
            abs_err = residual

        # 2. Subgroup failure discovery.
        subgroup_rows = []
        numeric_test = _to_frame(X_test).select_dtypes(include=[np.number])
        for col in numeric_test.columns[:30]:
            series = numeric_test[col]
            if series.nunique(dropna=True) < 3:
                continue
            try:
                bins = pd.qcut(series, q=min(5, series.nunique()), duplicates="drop")
            except Exception:
                bins = pd.cut(series, bins=min(5, series.nunique()), duplicates="drop")
            frame = pd.DataFrame({"bin": bins.astype(str), "error": abs_err, "residual": residual})
            base = float(np.mean(abs_err) + 1e-9)
            grouped = frame.groupby("bin", observed=False).agg(n=("error", "size"), mean_error=("error", "mean"), mean_residual=("residual", "mean")).reset_index()
            for _, row in grouped.iterrows():
                if int(row["n"]) >= max(3, len(frame) * 0.04):
                    subgroup_rows.append({
                        "feature": col,
                        "segment": row["bin"],
                        "n": int(row["n"]),
                        "mean_error": float(row["mean_error"]),
                        "error_lift_vs_average": float(row["mean_error"] / base),
                        "mean_residual_or_error_flag": float(row["mean_residual"]),
                    })
        subgroup_table = pd.DataFrame(subgroup_rows).sort_values("error_lift_vs_average", ascending=False).head(top_k) if subgroup_rows else pd.DataFrame()
        tables["Subgroup Failure Discovery"] = subgroup_table.to_dict("records")
        if not subgroup_table.empty:
            apply_nature_style()
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8.2, max(4.2, 0.36 * len(subgroup_table) + 1.4)))
            labels = subgroup_table["feature"].astype(str) + " | " + subgroup_table["segment"].astype(str)
            ax.barh(labels[::-1], subgroup_table["error_lift_vs_average"].to_numpy()[::-1], color=NATURE_COLORS["red"])
            ax.axvline(1, color=NATURE_COLORS["slate"], linestyle="--")
            ax.set_xlabel("Error lift vs test average")
            ax.set_title("Subgroup Failure Discovery")
            plots["Subgroup Failure Discovery"] = fig_to_base64(fig, fmt, dpi)

        # 3. Train-test domain shift.
        shift_rows = []
        train_num = _to_frame(X_train).select_dtypes(include=[np.number])
        test_num = _to_frame(X_test).select_dtypes(include=[np.number])
        common = [c for c in train_num.columns if c in test_num.columns]
        for col in common[:40]:
            tr = pd.to_numeric(train_num[col], errors="coerce").dropna()
            te = pd.to_numeric(test_num[col], errors="coerce").dropna()
            if len(tr) < 4 or len(te) < 4:
                continue
            pooled = np.sqrt((tr.var() + te.var()) / 2) + 1e-9
            smd = float((te.mean() - tr.mean()) / pooled)
            qshift = float(np.mean(np.abs(np.quantile(te, [0.1, 0.5, 0.9]) - np.quantile(tr, [0.1, 0.5, 0.9]))) / (pooled + 1e-9))
            shift_rows.append({"feature": col, "standardized_mean_difference": smd, "absolute_smd": abs(smd), "quantile_shift": qshift})
        shift_table = pd.DataFrame(shift_rows).sort_values("absolute_smd", ascending=False).head(top_k) if shift_rows else pd.DataFrame()
        tables["Train Test Domain Shift"] = shift_table.to_dict("records")
        if not shift_table.empty:
            plots["Train-Test Domain Shift"] = _publication_bar_plot(shift_table.sort_values("absolute_smd"), "feature", "absolute_smd", "Train-Test Domain Shift", fmt, dpi)

        # 4. Active learning priority from feature-space novelty and error proxy.
        try:
            if common:
                scaler = StandardScaler()
                train_scaled = scaler.fit_transform(train_num[common].fillna(train_num[common].median()))
                test_scaled = scaler.transform(test_num[common].fillna(train_num[common].median()))
                nn = NearestNeighbors(n_neighbors=min(5, len(train_scaled))).fit(train_scaled)
                dist, _ = nn.kneighbors(test_scaled)
                novelty = dist.mean(axis=1)
                priority = pd.DataFrame({
                    "test_sample_index": np.arange(len(test_num)),
                    "novelty_distance": novelty,
                    "observed_error_proxy": abs_err,
                    "priority_score": (novelty - novelty.min()) / (novelty.ptp() + 1e-9) + (abs_err - abs_err.min()) / (abs_err.ptp() + 1e-9),
                }).sort_values("priority_score", ascending=False).head(top_k)
                tables["Active Learning Priority"] = priority.to_dict("records")
                apply_nature_style()
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6.5, 5.2))
                sc = ax.scatter(novelty, abs_err, c=priority.index.isin(priority.index), cmap="coolwarm", s=42, alpha=0.78, edgecolors="white", linewidth=0.3)
                ax.set_xlabel("Feature-space novelty distance")
                ax.set_ylabel("Observed error proxy")
                ax.set_title("Active Learning Priority Map")
                plots["Active Learning Priority Map"] = fig_to_base64(fig, fmt, dpi)
        except Exception as exc:
            tables["Active Learning Priority"] = [{"error": str(exc)}]

        # 5. Symbolic surrogate of model behavior for regression.
        symbolic = {"available": False}
        if str(ptype).lower() == "regression":
            try:
                numeric_train = _to_frame(X_train).select_dtypes(include=[np.number])
                use_cols = numeric_train.var().sort_values(ascending=False).head(min(6, numeric_train.shape[1])).index.tolist()
                if use_cols:
                    scaler = StandardScaler()
                    Xs = scaler.fit_transform(numeric_train[use_cols].fillna(numeric_train[use_cols].median()))
                    poly = PolynomialFeatures(degree=2, include_bias=False)
                    Xp = poly.fit_transform(Xs)
                    target_pred = np.asarray(model.predict(X_train), dtype=float)
                    surrogate = Lasso(alpha=0.002, max_iter=10000).fit(Xp, target_pred)
                    coef = pd.DataFrame({"term": poly.get_feature_names_out(use_cols), "coefficient": surrogate.coef_})
                    coef["abs_coefficient"] = coef["coefficient"].abs()
                    coef = coef.sort_values("abs_coefficient", ascending=False).head(top_k)
                    intercept = float(surrogate.intercept_)
                    terms = [f"{row.coefficient:+.4g}*{row.term}" for row in coef.itertuples() if abs(row.coefficient) > 1e-9]
                    equation = f"prediction ~= {intercept:.4g} " + " ".join(terms[:8])
                    symbolic = {
                        "available": True,
                        "surrogate_r2_to_model_predictions": float(surrogate.score(Xp, target_pred)),
                        "equation": equation,
                        "top_terms": coef.to_dict("records"),
                    }
                    tables["Symbolic Surrogate Terms"] = coef.to_dict("records")
                    if not coef.empty:
                        plots["Symbolic Surrogate Terms"] = _publication_bar_plot(coef.sort_values("coefficient"), "term", "coefficient", "Symbolic Surrogate Terms", fmt, dpi)
            except Exception as exc:
                symbolic = {"available": False, "error": str(exc)}

        evidence = {
            "model_name": model_name,
            "plots": plots,
            "tables": tables,
            "symbolic_surrogate": symbolic,
            "summary": [
                "Repeated validation estimates score stability beyond a single split.",
                "Subgroup discovery identifies feature ranges where the model fails disproportionately.",
                "Domain-shift analysis quantifies train/test distribution mismatch.",
                "Active-learning priority ranks samples that are both novel and high-risk.",
                "Symbolic surrogate approximates model behavior with a compact formula when regression data are numeric enough.",
            ],
        }
        SESSION_STATE.setdefault("publication_evidence", {})["advanced_discovery"] = evidence
        return {"status": "success", "data": evidence}
    except Exception as e:
        return {"status": "error", "message": f"Advanced discovery failed: {str(e)}"}


@app.get("/api/publication/readiness_score")
async def publication_readiness_score():
    try:
        evidence = SESSION_STATE.get("publication_evidence", {}) or {}
        results = SESSION_STATE.get("last_training_results", []) or []
        df = SESSION_STATE.get("df")
        target = SESSION_STATE.get("target_col")
        ptype = SESSION_STATE.get("problem_type")
        best = SESSION_STATE.get("best_model_name")

        def dimension(name, weight, score, status, evidence_text, gaps=None, actions=None):
            return {
                "dimension": name,
                "weight": weight,
                "score": max(0, min(100, float(score))),
                "weighted_score": max(0, min(100, float(score))) * float(weight) / 100.0,
                "status": status,
                "evidence": evidence_text,
                "gaps": gaps or [],
                "recommended_actions": actions or [],
            }

        dims = []

        # 1. Data quality and auditability.
        if df is not None and target:
            missing_rate = float(df.isna().sum().sum() / max(1, df.size))
            duplicate_rate = float(df.duplicated().sum() / max(1, len(df)))
            rows, cols = df.shape
            score = 80
            if rows >= 200:
                score += 8
            elif rows < 50:
                score -= 20
            score -= min(25, missing_rate * 100)
            score -= min(15, duplicate_rate * 100)
            status = "strong" if score >= 80 else "review" if score >= 60 else "weak"
            dims.append(dimension(
                "Data quality",
                12,
                score,
                status,
                f"Dataset has {rows} rows, {cols} columns, missing rate {missing_rate:.2%}, duplicate row rate {duplicate_rate:.2%}.",
                gaps=(["Sample size is small for high-impact claims."] if rows < 100 else []) + (["Missingness should be justified or imputed with sensitivity analysis."] if missing_rate > 0.05 else []),
                actions=["Run complete data audit and report sample inclusion/exclusion.", "Document missing-value handling and duplicate policy."],
            ))
        else:
            dims.append(dimension("Data quality", 12, 0, "missing", "No dataset/target is available.", ["Upload dataset and set target."], ["Upload data and define the prediction target."]))

        # 2. Predictive performance.
        best_row = next((row for row in results if row.get("Model") == best), results[0] if results else {})
        numeric_metrics = {k: v for k, v in best_row.items() if isinstance(v, (int, float)) and not pd.isna(v)}
        if numeric_metrics:
            if str(ptype).lower() == "classification":
                primary = numeric_metrics.get("Accuracy", numeric_metrics.get("F1", max(numeric_metrics.values())))
                score = float(primary) * 100 if primary <= 1.5 else min(100, float(primary))
                label = f"primary classification score={primary:.4f}"
            else:
                primary = numeric_metrics.get("R2", numeric_metrics.get("Test R2", numeric_metrics.get("R2 Score", 0)))
                score = 50 + 50 * max(-1, min(1, float(primary)))
                label = f"primary regression R2-like score={primary:.4f}"
            dims.append(dimension(
                "Predictive performance",
                14,
                score,
                "strong" if score >= 80 else "review" if score >= 60 else "weak",
                f"Best model: {best_row.get('Model', best)} with {label}.",
                gaps=[] if score >= 75 else ["Held-out performance is not yet strong enough for a central claim."],
                actions=["Compare against strong literature and simple-rule baselines.", "Report confidence intervals for all primary metrics."],
            ))
        else:
            dims.append(dimension("Predictive performance", 14, 0, "missing", "No trained model leaderboard is available.", ["Train and evaluate candidate models."], ["Run model training."]))

        # 3. Generalization and robustness.
        advanced = evidence.get("advanced_discovery") or {}
        repeated = (advanced.get("tables") or {}).get("Repeated Validation Scores")
        external = evidence.get("external_validation")
        score = 30
        ev_bits = []
        gaps = []
        if repeated and not (isinstance(repeated, list) and repeated and "error" in repeated[0]):
            vals = []
            for row in repeated:
                for key, val in row.items():
                    if key != "fold" and isinstance(val, (int, float)) and not pd.isna(val):
                        vals.append(float(val))
            if vals:
                stability = max(0, 100 - np.nanstd(vals) * 200)
                score += min(35, stability * 0.35)
                ev_bits.append(f"Repeated validation available, SD={np.nanstd(vals):.4f}.")
        else:
            gaps.append("Repeated validation has not been run.")
        if external:
            score += 25
            ev_bits.append(f"External validation available with {external.get('rows')} samples.")
        else:
            gaps.append("Independent external validation is missing.")
        dims.append(dimension(
            "Generalization evidence",
            14,
            score,
            "strong" if score >= 80 else "review" if score >= 55 else "weak",
            " ".join(ev_bits) or "Only internal split evidence is currently available.",
            gaps,
            ["Run Advanced Scientific Discovery for repeated validation.", "Upload an independent external validation dataset from a different batch/source."],
        ))

        # 4. Leakage and bias safeguards.
        leakage = evidence.get("leakage") or {}
        leakage_summary = leakage.get("leakage_summary") or []
        high_flags = [r for r in leakage_summary if r.get("severity") == "high" and r.get("value", 0)]
        review_flags = [r for r in leakage_summary if r.get("severity") == "review" and r.get("value", 0)]
        if leakage_summary:
            score = 90 - 35 * len(high_flags) - 8 * len(review_flags)
            status = "strong" if score >= 80 else "review" if score >= 55 else "weak"
            gaps = [f"High leakage sentinel: {r.get('check')}={r.get('value')}" for r in high_flags]
            gaps += [f"Review leakage sentinel: {r.get('check')}={r.get('value')}" for r in review_flags[:3]]
            ev = f"Leakage audit completed with {len(high_flags)} high-severity and {len(review_flags)} review-level sentinel(s)."
        else:
            score, status, gaps, ev = 20, "missing", ["Leakage audit has not been run."], "No leakage audit evidence is available."
        dims.append(dimension("Leakage and bias safeguards", 12, score, status, ev, gaps, ["Run leakage audit and resolve high-severity sentinels before submission."]))

        # 5. Interpretability and mechanism support.
        has_shap = bool(SESSION_STATE.get("best_model")) and bool(SESSION_STATE.get("X_train") is not None)
        symbolic = (advanced.get("symbolic_surrogate") or {}).get("available", False)
        ablation = evidence.get("ablation")
        score = 25 + (25 if has_shap else 0) + (25 if symbolic else 0) + (25 if ablation else 0)
        gaps = []
        if not ablation:
            gaps.append("Ablation evidence is missing.")
        if not symbolic:
            gaps.append("Symbolic surrogate/mechanistic formula has not been generated.")
        dims.append(dimension(
            "Interpretability and mechanism",
            12,
            score,
            "strong" if score >= 80 else "review" if score >= 55 else "weak",
            f"Ablation={'yes' if ablation else 'no'}, symbolic surrogate={'yes' if symbolic else 'no'}, model explanation ready={'yes' if has_shap else 'no'}.",
            gaps,
            ["Run ablation study.", "Run advanced scientific discovery.", "Connect interpretation results to domain mechanism, not only correlation."],
        ))

        # 6. Domain shift and failure analysis.
        adv_tables = advanced.get("tables") or {}
        has_shift = bool(adv_tables.get("Train Test Domain Shift"))
        has_subgroup = bool(adv_tables.get("Subgroup Failure Discovery"))
        has_active = bool(adv_tables.get("Active Learning Priority"))
        score = 20 + 30 * int(has_shift) + 30 * int(has_subgroup) + 20 * int(has_active)
        dims.append(dimension(
            "Failure modes and domain shift",
            10,
            score,
            "strong" if score >= 80 else "review" if score >= 55 else "weak",
            f"Domain shift={'yes' if has_shift else 'no'}, subgroup failure={'yes' if has_subgroup else 'no'}, active-learning priority={'yes' if has_active else 'no'}.",
            [x for x, ok in [("Domain-shift decomposition is missing.", has_shift), ("Subgroup failure discovery is missing.", has_subgroup), ("Next-experiment priority list is missing.", has_active)] if not ok],
            ["Run Advanced Scientific Discovery and inspect high-risk subgroups."],
        ))

        # 7. Reproducibility package.
        has_splits = SESSION_STATE.get("X_train") is not None and SESSION_STATE.get("X_test") is not None
        has_config = bool(SESSION_STATE.get("last_training_config"))
        has_feature_schema = bool(SESSION_STATE.get("feature_columns"))
        score = 20 + 30 * int(has_splits) + 25 * int(has_config) + 25 * int(has_feature_schema)
        dims.append(dimension(
            "Reproducibility readiness",
            12,
            score,
            "strong" if score >= 80 else "review" if score >= 55 else "weak",
            f"Splits={'yes' if has_splits else 'no'}, training config={'yes' if has_config else 'no'}, feature schema={'yes' if has_feature_schema else 'no'}.",
            [x for x, ok in [("Train/test splits are not stored.", has_splits), ("Training configuration is missing.", has_config), ("Feature schema is missing.", has_feature_schema)] if not ok],
            ["Export the reproducibility package and archive it with manuscript materials."],
        ))

        # 8. Manuscript assembly.
        text_available = bool(evidence) and bool(results)
        storyboard_available = True
        figures_available = bool(SESSION_STATE.get("publication_evidence")) or bool(results)
        score = 20 + 30 * int(text_available) + 25 * int(storyboard_available) + 25 * int(figures_available)
        dims.append(dimension(
            "Manuscript assembly",
            14,
            score,
            "strong" if score >= 80 else "review" if score >= 55 else "weak",
            f"Results evidence={'yes' if text_available else 'no'}, storyboard template=yes, figure candidates={'yes' if figures_available else 'no'}.",
            [] if score >= 80 else ["Run evidence modules before generating final captions and main/supplementary figure plan."],
            ["Generate results text, storyboard, captions, and one-click submission package."],
        ))

        total = float(sum(d["weighted_score"] for d in dims))
        blocking = []
        if any(d["status"] == "missing" for d in dims):
            blocking.append("One or more evidence dimensions are missing.")
        if external is None:
            blocking.append("External validation is missing.")
        if high_flags:
            blocking.append("High-severity leakage sentinel(s) must be resolved.")
        if total >= 82 and not blocking:
            verdict = "Ready for manuscript assembly"
        elif total >= 62:
            verdict = "Major revision needed before NC-level submission"
        else:
            verdict = "Not ready for high-impact submission"

        prioritized_actions = []
        for dim in sorted(dims, key=lambda x: x["weighted_score"]):
            for action in dim["recommended_actions"]:
                if action not in prioritized_actions:
                    prioritized_actions.append(action)
            if len(prioritized_actions) >= 8:
                break

        return {
            "status": "success",
            "data": {
                "total_score": round(total, 2),
                "verdict": verdict,
                "blocking_issues": blocking,
                "dimensions": dims,
                "prioritized_actions": prioritized_actions[:8],
            },
        }
    except Exception as e:
        return {"status": "error", "message": f"Readiness scoring failed: {str(e)}"}


def _caption_for_plot(name, context="modeling"):
    name = str(name or "Figure")
    caption_map = {
        "Leakage Correlation Sentinel": "Data leakage sentinel analysis. Bars show the absolute correlation between each numeric predictor and the target variable; features approaching unity were flagged for manual review to exclude target-derived or post-outcome information.",
        "Ablation Performance Drop": "Feature ablation analysis. Each bar reports the change in the primary model score after removing one feature and retraining the model; negative values indicate that the removed feature was necessary for predictive performance.",
        "External Validation Observed vs Predicted": "External validation performance. Observed target values are plotted against model predictions on an independent validation set; the diagonal line represents perfect agreement.",
        "Repeated Validation Stability": "Repeated cross-validation stability. Points denote scores from repeated folds, the dashed line denotes the mean score, and the shaded region denotes one standard deviation, quantifying sensitivity to data partitioning.",
        "Subgroup Failure Discovery": "Subgroup failure discovery. Feature intervals were screened for elevated error relative to the test-set average; bars greater than one identify input regions where the model is less reliable.",
        "Train-Test Domain Shift": "Train-test domain-shift analysis. Bars show standardized distribution shifts between training and test sets for each feature, highlighting covariates with potential extrapolation risk.",
        "Active Learning Priority Map": "Active-learning priority analysis. Samples with high feature-space novelty and high error risk are prioritized for additional measurement or experimental validation.",
        "Symbolic Surrogate Terms": "Symbolic surrogate model. A sparse polynomial surrogate was fitted to approximate the trained model; positive and negative coefficients indicate terms that increase or decrease predicted response, respectively.",
    }
    if name in caption_map:
        return caption_map[name]
    return f"{name}. This panel summarizes {context} evidence generated by the analysis workflow. The underlying reconstruction data should be reported with sample size, model name, validation split, and plotting parameters in the supplementary tables."


@app.get("/api/publication/reviewer_simulator")
async def publication_reviewer_simulator():
    readiness = await publication_readiness_score()
    data = readiness.get("data", {}) if isinstance(readiness, dict) else {}
    evidence = SESSION_STATE.get("publication_evidence", {}) or {}
    critiques = []

    def add(severity, topic, criticism, response, required_action):
        critiques.append({
            "severity": severity,
            "topic": topic,
            "likely_reviewer_criticism": criticism,
            "recommended_response": response,
            "required_action": required_action,
        })

    if not evidence.get("external_validation"):
        add("major", "External validation", "The model is evaluated mainly on internal splits; this does not demonstrate generalization to independent experimental conditions.", "Add an independent validation cohort or clearly downgrade claims to internal predictive performance.", "Upload and report an external validation dataset from a different batch, time period, laboratory, or public source.")
    if any("leakage" in issue.lower() for issue in data.get("blocking_issues", [])):
        add("major", "Data leakage", "Potential leakage sentinels remain unresolved, so the reported performance may be inflated.", "Manually audit flagged variables, remove target-derived/post-outcome fields, and rerun training.", "Resolve all high-severity leakage checks and export the audit table.")
    if not evidence.get("ablation"):
        add("major", "Ablation evidence", "The manuscript does not show whether the claimed key variables are necessary for model performance.", "Run feature ablation and report the effect of removing top predictors.", "Run Ablation Study for the final selected model.")
    if not evidence.get("advanced_discovery"):
        add("moderate", "Robustness and failure modes", "The model lacks systematic analysis of split stability, subgroup failures, and domain shift.", "Add repeated validation, subgroup failure discovery, and domain-shift analysis to define model applicability.", "Run Advanced Scientific Discovery.")
    if not SESSION_STATE.get("last_training_results"):
        add("major", "Baseline comparison", "No model leaderboard or baseline comparison is available.", "Benchmark the final model against simple and strong baseline methods.", "Train a baseline set including linear, tree, kernel, ensemble, and simple models.")
    add("moderate", "Mechanistic interpretation", "Feature-importance and SHAP-style analyses may be correlational and do not establish mechanism.", "Frame interpretations as hypotheses unless supported by domain theory or experimental perturbation.", "Connect key features to literature mechanisms and add validation experiments where possible.")
    add("moderate", "Reproducibility", "The analysis must be reproducible from raw data to figures.", "Provide data splits, model parameters, figure reconstruction data, code/environment metadata, and a run manifest.", "Export the reproducibility/submission package and archive it with the manuscript.")

    SESSION_STATE.setdefault("publication_evidence", {})["reviewer_simulator"] = {"critiques": critiques}
    return {"status": "success", "data": {"critiques": critiques, "readiness_verdict": data.get("verdict"), "readiness_score": data.get("total_score")}}


@app.post("/api/publication/robustness_battery")
async def publication_robustness_battery(request: Request):
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split

    data = await request.json()
    model_name = data.get("model_name") or SESSION_STATE.get("best_model_name")
    repeats = int(data.get("repeats", 8))
    test_size = float(data.get("test_size", SESSION_STATE.get("last_training_config", {}).get("test_size", 0.2)))
    fmt = data.get("format", "png")
    dpi = int(data.get("dpi", 300))
    df = SESSION_STATE.get("df")
    target = SESSION_STATE.get("target_col")
    model = SESSION_STATE.get("trained_models", {}).get(model_name)
    if df is None or not target or model is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Dataset, target, and trained model are required."})
    try:
        X = mod_module.prepare_design_matrix(df, target, SESSION_STATE.get("selected_features"), SESSION_STATE.get("feature_columns"))
        y = df[target]
        rows = []
        ptype = SESSION_STATE.get("problem_type")
        for seed in range(repeats):
            stratify = y if str(ptype).lower() == "classification" and pd.Series(y).nunique() > 1 else None
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed + 101, stratify=stratify)
            try:
                candidate = clone(model)
                candidate.fit(X_tr, y_tr)
                pred = candidate.predict(X_te)
                metrics = _score_predictions(y_te, pred, ptype)
                rows.append({"seed": seed + 101, **metrics})
            except Exception as exc:
                rows.append({"seed": seed + 101, "error": str(exc)})
        table = pd.DataFrame(rows)
        numeric = table.select_dtypes(include=[np.number])
        primary = "Accuracy" if str(ptype).lower() == "classification" else "R2"
        plots = {}
        if primary in table.columns:
            apply_nature_style()
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7.2, 4.6))
            ax.plot(table["seed"], table[primary], marker="o", color=NATURE_COLORS["blue"])
            ax.axhline(table[primary].mean(), color=NATURE_COLORS["red"], linestyle="--", label=f"mean={table[primary].mean():.3f}")
            ax.fill_between(table["seed"], table[primary].mean() - table[primary].std(), table[primary].mean() + table[primary].std(), color=NATURE_COLORS["blue"], alpha=0.12, label="mean +/- SD")
            ax.set_xlabel("Random seed")
            ax.set_ylabel(primary)
            ax.set_title("Robustness Battery Split Sensitivity")
            ax.legend()
            plots["Robustness Battery Split Sensitivity"] = fig_to_base64(fig, fmt, dpi)
        summary = numeric.agg(["mean", "std", "min", "max"]).reset_index().rename(columns={"index": "statistic"}) if not numeric.empty else pd.DataFrame()
        evidence = {"model_name": model_name, "table": table.to_dict("records"), "summary": summary.to_dict("records"), "plots": plots}
        SESSION_STATE.setdefault("publication_evidence", {})["robustness_battery"] = evidence
        return {"status": "success", "data": evidence}
    except Exception as e:
        return {"status": "error", "message": f"Robustness battery failed: {str(e)}"}


@app.get("/api/publication/figure_manuscript_builder")
async def publication_figure_manuscript_builder():
    evidence = SESSION_STATE.get("publication_evidence", {}) or {}
    base = (await publication_storyboard()).get("data", [])
    captions = []
    available = []
    for block in base:
        panels = []
        for panel in block.get("recommended_panels", []):
            panels.append({"panel": panel, "caption": _caption_for_plot(panel), "available": panel in json.dumps(_strip_data_urls(evidence), ensure_ascii=False)})
        available.append({**block, "panels": panels})
    for key in ["leakage", "ablation", "external_validation", "advanced_discovery", "robustness_battery"]:
        section = evidence.get(key) or {}
        for plot_name in (section.get("plots") or {}).keys():
            captions.append({"plot": plot_name, "caption": _caption_for_plot(plot_name, key), "source": key})
    if not captions:
        captions = [{"plot": "Model Leaderboard", "caption": _caption_for_plot("Model Leaderboard", "model comparison"), "source": "default"}]
    result = {"storyboard": available, "captions": captions}
    SESSION_STATE.setdefault("publication_evidence", {})["figure_manuscript_builder"] = result
    return {"status": "success", "data": result}


@app.get("/api/publication/submission_package")
async def publication_submission_package():
    try:
        readiness = await publication_readiness_score()
        reviewer = await publication_reviewer_simulator()
        builder = await publication_figure_manuscript_builder()
        results_text = await publication_results_text()
        output = io.BytesIO()
        evidence_clean = _strip_data_urls(SESSION_STATE.get("publication_evidence", {}))
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("00_readiness_score.json", json.dumps(readiness.get("data", {}), ensure_ascii=False, indent=2, default=str))
            zf.writestr("01_reviewer_criticism_simulator.json", json.dumps(reviewer.get("data", {}), ensure_ascii=False, indent=2, default=str))
            zf.writestr("02_figure_to_manuscript_builder.json", json.dumps(builder.get("data", {}), ensure_ascii=False, indent=2, default=str))
            zf.writestr("03_results_draft.txt", results_text.get("data", {}).get("text", ""))
            zf.writestr("04_publication_evidence_manifest.json", json.dumps(evidence_clean, ensure_ascii=False, indent=2, default=str))
            zf.writestr("05_model_leaderboard.json", json.dumps(SESSION_STATE.get("last_training_results", []), ensure_ascii=False, indent=2, default=str))
            zf.writestr("06_training_config.json", json.dumps(SESSION_STATE.get("last_training_config", {}), ensure_ascii=False, indent=2, default=str))
            if SESSION_STATE.get("df") is not None:
                zf.writestr("tables/current_dataset.csv", _to_frame(SESSION_STATE.get("df")).to_csv(index=False))
            for key, section in (SESSION_STATE.get("publication_evidence", {}) or {}).items():
                plot_items = (section.get("plots") or {}).items() if isinstance(section, dict) else []
                for plot_name, data_url in plot_items:
                    try:
                        _, ext, payload = _decode_data_url(data_url)
                        zf.writestr(f"figures/{_slugify_filename(key+'_'+plot_name)}.{ext}", payload)
                    except Exception:
                        pass
        encoded = base64.b64encode(output.getvalue()).decode("utf-8")
        return {"status": "success", "filename": "NatureML_submission_support_package.zip", "content_base64": encoded}
    except Exception as e:
        return {"status": "error", "message": f"Submission package failed: {str(e)}"}

import core.interpretation as int_module
import core.optimization as opt_module
import core.figure_studio as figure_module
import core.data_audit as audit_module


@app.post("/api/audit/run")
async def run_data_audit(request: Request):
    data = await request.json()
    df = SESSION_STATE.get("df")
    if df is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Dataset missing."})
    fmt = data.get("format", "png")
    dpi = int(data.get("dpi", 300))
    try:
        result = audit_module.run_data_audit(
            df,
            target_col=SESSION_STATE.get("target_col"),
            problem_type=SESSION_STATE.get("problem_type", "Regression"),
            fmt=fmt,
            dpi=dpi,
        )
        SESSION_STATE["audit_result"] = result
        public_result = {k: v for k, v in result.items() if k != "plot_data"}
        return {"status": "success", "data": public_result}
    except Exception as e:
        return {"status": "error", "message": f"Data audit failed: {str(e)}"}

@app.post("/api/interpret/shap")
async def get_shap(request: Request):
    data = await request.json()
    model = SESSION_STATE.get('best_model')
    X_train = SESSION_STATE.get('X_train')
    
    if model is None or X_train is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "You must train a model first before explaining."})
        
    fmt = data.get("format", "png")
    dpi = data.get("dpi", 300)
    sample_idx = int(data.get("sample_idx", 0))
    try:
        plots = int_module.get_shap_plots(model, X_train, sample_idx, fmt, dpi)
        return {"status": "success", "plots": plots}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/interpret/advanced_xai")
async def get_advanced_xai(request: Request):
    data = await request.json()
    model = SESSION_STATE.get('best_model')
    X_train = SESSION_STATE.get('X_train')
    y_train = SESSION_STATE.get('y_train')
    if model is None or X_train is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "You must train a model first before generating Advanced XAI Atlas."})
    fmt = data.get("format", "png")
    dpi = int(data.get("dpi", 300))
    try:
        plots = int_module.get_advanced_xai_atlas(model, X_train, y_train, fmt, dpi)
        return {"status": "success", "plots": plots}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/interpret/pdp")
async def get_pdp(request: Request):
    data = await request.json()
    model = SESSION_STATE.get('best_model')
    X_train = SESSION_STATE.get('X_train')
    feature = data.get("feature")
    if feature is not None: feature = str(feature)
    
    if model is None or X_train is None or not feature: 
        return JSONResponse(status_code=400, content={"status": "error", "message": "Missing dependencies for PDP."})
    fmt = data.get("format", "png")
    dpi = data.get("dpi", 300)
    
    feature2 = data.get("feature2", None)
    if feature2 == "None" or feature2 is None:
        feature2 = None
    else:
        feature2 = str(feature2)

    feature3 = data.get("feature3", None)
    if feature3 == "None" or feature3 is None:
        feature3 = None
    else:
        feature3 = str(feature3)
        
    try:
        plots = int_module.get_pdp_plot(model, X_train, feature, feature2, feature3, fmt, dpi)
        return {"status": "success", "plots": plots}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/interpret/permutation")
async def get_permutation(request: Request):
    data = await request.json()
    model = SESSION_STATE.get('best_model')
    X_test = SESSION_STATE.get('X_test')
    y_test = SESSION_STATE.get('y_test')

    if model is None or X_test is None or y_test is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "You must train a model first before running permutation importance."})

    fmt = data.get("format", "png")
    dpi = data.get("dpi", 300)
    try:
        plots = int_module.get_permutation_importance_plots(model, X_test, y_test, fmt, dpi)
        return {"status": "success", "plots": plots}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/interpret/causal")
async def get_causal(request: Request):
    data = await request.json()
    df = SESSION_STATE.get('df_clean') if SESSION_STATE.get('df_clean') is not None else SESSION_STATE.get('df')
    target = SESSION_STATE.get('target_col')
    treatment = data.get("treatment")

    if df is None or not target:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Dataset or target column missing."})
    
    fmt = data.get("format", "png")
    dpi = data.get("dpi", 300)
    try:
        treatment = treatment if treatment not in ["", "ALL", None] else None
        res = int_module.get_causal_effect(df, target, treatment, fmt, dpi)
        return {"status": "success", "data": res}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/interpret/counterfactual")
async def get_counterfactual(request: Request):
    data = await request.json()
    model = SESSION_STATE.get("best_model")
    X_train = SESSION_STATE.get("X_train")
    if model is None or X_train is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "You must train a model first before generating counterfactuals."})
    fmt = data.get("format", "png")
    dpi = data.get("dpi", 300)
    sample_idx = int(data.get("sample_idx", 0))
    desired_value = data.get("desired_value")
    try:
        res = int_module.get_counterfactual_plots(model, X_train, sample_idx, desired_value, fmt, dpi)
        return {"status": "success", "data": res}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/optimization/schema")
async def get_optimization_schema():
    df = SESSION_STATE.get("df")
    target = SESSION_STATE.get("target_col")
    if df is None or not target:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Dataset or target column missing."})
    model_registry = {}
    model_registry.update(SESSION_STATE.get("trained_models", {}))
    model_registry.update({k: v.get("model") for k, v in SESSION_STATE.get("imported_models", {}).items()})
    schema = opt_module.get_optimization_schema(df, target, model_registry)
    schema["regression_models"] = mod_module.get_model_registry("Regression")
    return {"status": "success", "data": schema}


def _parse_numeric_grid(value, default_values=None):
    default_values = default_values or [1, 3, 7, 14, 28, 56, 90, 180, 365]
    if isinstance(value, (list, tuple)):
        raw_items = value
    else:
        raw_items = re.split(r"[,，\s]+", str(value or "").strip())
    points = []
    for item in raw_items:
        if item in (None, ""):
            continue
        try:
            points.append(float(item))
        except (TypeError, ValueError):
            continue
    points = sorted(set(points))
    return points or default_values


def _default_feature_value(df, feature):
    if df is None or feature not in df.columns:
        return 0.0
    series = df[feature].dropna()
    if series.empty:
        return 0.0
    if pd.api.types.is_numeric_dtype(series):
        return float(series.median())
    mode = series.mode()
    return mode.iloc[0] if not mode.empty else series.iloc[0]


def _infer_time_feature(feature_names, explicit=None):
    if explicit:
        return explicit
    patterns = ["age", "time", "day", "days", "龄期", "时间", "天", "duration"]
    for feature in feature_names:
        lower = str(feature).lower()
        if any(token in lower for token in patterns):
            return feature
    return feature_names[-1] if feature_names else None


def _prediction_interval_from_estimators(model, X_candidate):
    estimators = getattr(model, "estimators_", None)
    if estimators is None:
        return None
    try:
        flat_estimators = np.ravel(estimators)
        predictions = []
        for estimator in flat_estimators:
            if estimator is None or not hasattr(estimator, "predict"):
                continue
            predictions.append(np.asarray(estimator.predict(X_candidate), dtype=float))
        if len(predictions) < 5:
            return None
        matrix = np.vstack(predictions)
        return {
            "lower": np.nanquantile(matrix, 0.10, axis=0),
            "median": np.nanquantile(matrix, 0.50, axis=0),
            "upper": np.nanquantile(matrix, 0.90, axis=0),
            "std": np.nanstd(matrix, axis=0),
        }
    except Exception:
        return None


def _build_shrinkage_curve_evidence(result, shrinkage_config, imported_models, df, fmt, dpi):
    config = shrinkage_config or {}
    model_name = config.get("model_name")
    if not model_name:
        return None
    package = imported_models.get(model_name)
    if not package or package.get("model") is None:
        raise ValueError(f"Shrinkage model package not found: {model_name}")

    model = package["model"]
    feature_columns = package.get("feature_columns", []) or []
    raw_features = package.get("base_feature_columns") or package.get("selected_features") or feature_columns
    raw_features = [f for f in raw_features if f]
    if not raw_features:
        raise ValueError(f"Shrinkage model has no feature schema: {model_name}")

    time_feature = _infer_time_feature(raw_features, config.get("time_feature"))
    if not time_feature:
        raise ValueError("Cannot infer shrinkage time/age feature from imported model.")
    if time_feature not in raw_features:
        raw_features.append(time_feature)

    time_points = _parse_numeric_grid(config.get("time_points"))
    top_n = max(1, min(int(config.get("top_n") or 5), 12))
    probabilistic = bool(config.get("probabilistic", True))
    solutions = result.get("solutions") or []
    if not solutions:
        return None

    rows = []
    for sol_idx, solution in enumerate(solutions[:top_n], start=1):
        solution_label = f"Pareto {sol_idx}"
        for point in time_points:
            raw_row = {}
            for feature in raw_features:
                if feature == time_feature:
                    raw_row[feature] = point
                elif feature in solution:
                    raw_row[feature] = solution[feature]
                else:
                    raw_row[feature] = _default_feature_value(df, feature)
            rows.append({
                "solution_id": sol_idx,
                "solution_label": solution_label,
                time_feature: point,
                **raw_row,
            })

    curve_source = pd.DataFrame(rows)
    X_raw = curve_source[raw_features].copy()
    X_candidate = mod_module.prepare_design_matrix(X_raw, None, None, feature_columns)
    prediction = np.asarray(model.predict(X_candidate), dtype=float)
    curve_source["predicted_shrinkage"] = prediction

    interval = _prediction_interval_from_estimators(model, X_candidate) if probabilistic else None
    if interval:
        curve_source["lower_shrinkage"] = interval["lower"]
        curve_source["median_shrinkage"] = interval["median"]
        curve_source["upper_shrinkage"] = interval["upper"]
        curve_source["prediction_std"] = interval["std"]

    plot = opt_module.build_shrinkage_curve_plot(
        curve_source,
        time_col=time_feature,
        value_col="predicted_shrinkage",
        fmt=fmt,
        dpi=dpi,
    )
    return {
        "plot": plot,
        "data": curve_source.round(6).replace({np.nan: None}).to_dict(orient="records"),
        "config": {
            "model_name": model_name,
            "target_col": package.get("target_col"),
            "time_feature": time_feature,
            "time_points": time_points,
            "top_n": top_n,
            "probabilistic": probabilistic,
            "feature_columns": feature_columns,
            "raw_features": raw_features,
        },
    }


@app.post("/api/optimization/run")
async def run_optimization(request: Request):
    data = await request.json()
    df = SESSION_STATE.get("df")
    target = SESSION_STATE.get("target_col")
    model_name = data.get("model_name")
    algorithm = data.get("algorithm", "NSGA-II")
    objectives = data.get("objectives", [])
    constraints = data.get("constraints", [])
    bounds = data.get("variables", [])
    surrogate_models = data.get("surrogate_models", [])
    compare_algorithms = data.get("compare_algorithms", [])
    pop_size = int(data.get("pop_size", 64))
    generations = int(data.get("generations", 30))
    fmt = data.get("format", "png")
    dpi = int(data.get("dpi", 300))
    shrinkage_curve = data.get("shrinkage_curve") or {}

    if df is None or not target:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Dataset or target column missing."})
    if not bounds:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Optimization variables are required."})
    if not objectives:
        return JSONResponse(status_code=400, content={"status": "error", "message": "At least one objective is required."})

    imported_models = SESSION_STATE.get("imported_models", {})
    model = SESSION_STATE.get("trained_models", {}).get(model_name) if model_name else None
    feature_columns = SESSION_STATE.get("feature_columns", [])
    main_feature_columns = feature_columns
    if model is None and model_name in imported_models:
        model = imported_models[model_name].get("model")
        main_feature_columns = imported_models[model_name].get("feature_columns", feature_columns)
    selected_features = SESSION_STATE.get("selected_features", [])
    trained_surrogates = {}
    surrogate_packages = {}

    for spec in surrogate_models:
        target_col = spec.get("target_col")
        alias = spec.get("alias") or f"{target_col}_pred"
        source = spec.get("source", "dataset")
        if source == "imported_model":
            imported_name = spec.get("imported_model_name")
            imported = imported_models.get(imported_name)
            if imported is None:
                return {"status": "error", "message": f"Imported model not found: {imported_name}"}
            surrogate_model = imported["model"]
            feature_names = [item["name"] for item in bounds]
            feature_cols = imported.get("feature_columns", [])
            surrogate_packages[alias] = {
                "source": "imported_model",
                "imported_model_name": imported_name,
                "model_package": imported,
                "target_col": imported.get("target_col") or target_col,
                "feature_names": feature_names,
                "feature_columns": feature_cols,
            }
        else:
            surrogate_model_name = spec.get("model_name", "Random Forest")
            if not target_col or target_col not in df.columns:
                return {"status": "error", "message": f"Optimization surrogate target not found: {target_col}"}
            feature_names = [item["name"] for item in bounds if item["name"] in df.columns and item["name"] != target_col]
            if not feature_names:
                return {"status": "error", "message": f"No optimization variables matched dataset columns for surrogate target {target_col}."}
            surrogate_df = df[feature_names + [target_col]].dropna()
            if len(surrogate_df) < 20:
                return {"status": "error", "message": f"Not enough valid rows to train surrogate model for {target_col}."}
            model_registry = mod_module.get_models("Regression")
            if surrogate_model_name not in model_registry:
                return {"status": "error", "message": f"Unsupported surrogate regression model: {surrogate_model_name}"}
            surrogate_model = model_registry[surrogate_model_name]
            X_sur = mod_module.prepare_design_matrix(surrogate_df[feature_names], None, None)
            y_sur = surrogate_df[target_col]
            surrogate_model.fit(X_sur, y_sur)
            feature_cols = X_sur.columns.tolist()
            surrogate_packages[alias] = {
                "source": "dataset",
                "target_col": target_col,
                "model_name": surrogate_model_name,
                "feature_names": feature_names,
                "model_package": model_io.build_model_package(
                    model_name=f"{alias} ({surrogate_model_name})",
                    model=surrogate_model,
                    problem_type="Regression",
                    feature_columns=feature_cols,
                    selected_features=feature_names,
                    target_col=target_col,
                    base_feature_columns=feature_names,
                    training_config={
                        "source": "optimization_surrogate",
                        "row_count": int(len(surrogate_df)),
                        "requested_model": surrogate_model_name,
                    },
                ),
            }
        trained_surrogates[alias] = {
            "model": surrogate_model,
            "feature_columns": feature_cols,
            "feature_names": feature_names,
        }

    def prediction_fn(candidate_df):
        context = {}
        if model is not None:
            X_candidate = mod_module.prepare_design_matrix(candidate_df, None, selected_features, main_feature_columns)
            context["prediction"] = np.asarray(model.predict(X_candidate), dtype=float)
        else:
            context["prediction"] = np.zeros(len(candidate_df), dtype=float)
        for alias, payload in trained_surrogates.items():
            X_candidate = mod_module.prepare_design_matrix(candidate_df[payload["feature_names"]], None, None, payload["feature_columns"])
            context[alias] = np.asarray(payload["model"].predict(X_candidate), dtype=float)
        return context

    try:
        result = opt_module.run_multiobjective_optimization(
            bounds,
            objectives,
            constraints,
            algorithm,
            prediction_fn,
            pop_size=pop_size,
            generations=generations,
            random_state=42,
            fmt=fmt,
            dpi=dpi,
            compare_algorithms=compare_algorithms,
        )
        shrinkage_evidence = _build_shrinkage_curve_evidence(result, shrinkage_curve, imported_models, df, fmt, dpi)
        if shrinkage_evidence:
            result.setdefault("plots", {})["Optimized Shrinkage Curves"] = shrinkage_evidence["plot"]
            result["shrinkage_curve_data"] = shrinkage_evidence["data"]
            result["shrinkage_curve_config"] = shrinkage_evidence["config"]
        SESSION_STATE["optimization_result"] = result
        request_config = {
            "model_name": model_name,
            "algorithm": algorithm,
            "compare_algorithms": compare_algorithms,
            "objectives": objectives,
            "constraints": constraints,
            "surrogate_models": surrogate_models,
            "variables": bounds,
            "pop_size": pop_size,
            "generations": generations,
            "random_state": 42,
            "format": fmt,
            "dpi": dpi,
            "shrinkage_curve": shrinkage_curve,
        }
        if model is None:
            main_model_package = None
        elif model_name in imported_models:
            main_model_package = imported_models[model_name]
        else:
            metrics = next((row for row in SESSION_STATE.get("last_training_results", []) if row.get("Model") == model_name), {})
            main_model_package = model_io.build_model_package(
                model_name=model_name,
                model=model,
                problem_type=SESSION_STATE.get("problem_type"),
                feature_columns=main_feature_columns,
                selected_features=selected_features,
                target_col=target,
                base_feature_columns=SESSION_STATE.get("base_feature_columns", []),
                metrics=metrics,
                training_config=SESSION_STATE.get("last_training_config", {}),
                meta_history=SESSION_STATE.get("meta_histories", {}).get(model_name),
            )
        SESSION_STATE["optimization_export_package"] = model_io.build_optimization_package(
            name=f"{result.get('best_algorithm', algorithm)} Optimization",
            request_config=request_config,
            optimization_result=result,
            main_model_package=main_model_package,
            surrogate_packages=surrogate_packages,
            dataset_context={
                "target_col": target,
                "problem_type": SESSION_STATE.get("problem_type"),
                "columns": df.columns.tolist(),
                "rows": int(len(df)),
            },
        )
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": f"Optimization engine failed: {str(e)}"}


@app.post("/api/optimization/export")
async def export_optimization_package(request: Request):
    package = SESSION_STATE.get("optimization_export_package")
    if package is None:
        return JSONResponse(status_code=404, content={"status": "error", "message": "No optimization package is available. Run optimization first."})
    blob_b64 = model_io.dump_model_package(package)
    filename = f"{_slugify_filename(package.get('name') or 'NatureML_Optimization')}.naturemlopt"
    return {"status": "success", "filename": filename, "content_base64": blob_b64}


@app.post("/api/figure/compose")
async def compose_figure(request: Request):
    data = await request.json()
    panels = data.get("panels", [])
    fmt = data.get("format", "png")
    dpi = int(data.get("dpi", 300))
    try:
        image = figure_module.compose_figure(
            panels=panels,
            canvas_width=float(data.get("canvas_width", 7.2)),
            canvas_height=float(data.get("canvas_height", 5.4)),
            label_size=float(data.get("label_size", 16)),
            label_style=data.get("label_style", "lower"),
            gap=float(data.get("gap", 0.025)),
            margin=float(data.get("margin", 0.05)),
            columns=int(data.get("columns", 2)),
            preset=data.get("preset", "Auto Grid"),
            fmt=fmt,
            dpi=dpi,
        )
        return {"status": "success", "image": image}
    except Exception as e:
        return {"status": "error", "message": f"Figure composition failed: {str(e)}"}


@app.post("/api/export/plot_data")
async def export_plot_data(request: Request):
    try:
        data = await request.json()
        context = data.get("context", "dataset")
        plot_name = data.get("plot_name", "Plot")
        payload = data.get("payload", {}) or {}
        df = SESSION_STATE.get("df")
        target_col = SESSION_STATE.get("target_col")
        metadata = {
            "plot_name": plot_name,
            "plot_type": _plot_type_label(plot_name, context, payload),
            "context": context,
            "target_col": target_col,
            "problem_type": SESSION_STATE.get("problem_type"),
            "notes": "This workbook prioritizes plot reconstruction data: derived coordinates, bins, summaries, curves, or matrices used to reproduce the exported figure. Raw model/source rows are included only as traceability sheets when available.",
        }
        sheets = []

        if context == "eda":
            if df is None:
                return JSONResponse(status_code=400, content={"status": "error", "message": "Dataset missing."})
            cols = [payload.get("col_x"), payload.get("col_y"), target_col]
            cols = [c for c in cols if c and c in df.columns]
            if not cols:
                cols = df.columns.tolist()
            metadata.update({"plot_request": payload, "columns_included": cols})
            sheets.append(("Plot_Source_Data", df[cols].copy()))
            sheets.append(("Full_Dataset", df.copy()))
            pt = payload.get("plot_type")
            if pt in {"corr", "corr_spearman"}:
                method = "spearman" if pt == "corr_spearman" else "pearson"
                sheets.append((f"{method}_correlation", df.select_dtypes(include=[np.number]).corr(method=method)))

        elif context == "preprocess":
            plot_key = payload.get("plot_key")
            artifact = SESSION_STATE.get("preprocess_artifacts", {}).get(plot_key)
            if artifact is None:
                return JSONResponse(status_code=404, content={"status": "error", "message": "Preprocessing source data not found."})
            metadata.update({"plot_key": plot_key, "process_name": artifact.get("process_name")})
            sheets.append(("Before_Preprocess", artifact.get("df_old")))
            sheets.append(("After_or_Result", artifact.get("df_new")))
            if df is not None:
                sheets.append(("Current_Dataset", df.copy()))

        elif context == "global_model":
            metadata.update({"filter": payload})
            leaderboard = SESSION_STATE.get("last_training_results", [])
            sheets.append(("Model_Leaderboard", leaderboard))
            lb = _to_frame(leaderboard)
            if not lb.empty:
                numeric_cols = lb.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    sheets.append(("Plot_Reconstruction_Data", lb[["Model"] + numeric_cols].copy() if "Model" in lb.columns else lb[numeric_cols].copy()))
            X_train, y_train = SESSION_STATE.get("X_train"), SESSION_STATE.get("y_train")
            X_test, y_test = SESSION_STATE.get("X_test"), SESSION_STATE.get("y_test")
            sheets.append(("Train_Features_Target", _series_with_prediction(None, X_train, y_train, "train")))
            sheets.append(("Test_Features_Target", _series_with_prediction(None, X_test, y_test, "test")))

        elif context == "model":
            model_name = payload.get("model_name") or SESSION_STATE.get("best_model_name")
            model = SESSION_STATE.get("trained_models", {}).get(model_name)
            if model is None:
                return JSONResponse(status_code=404, content={"status": "error", "message": "Model not found in memory."})
            metadata.update({"model_name": model_name, "model_plot_request": payload})
            sheets.append(("Model_Leaderboard", SESSION_STATE.get("last_training_results", [])))
            for sheet_name, sheet_data in _model_plot_reconstruction_sheets(
                plot_name,
                model,
                SESSION_STATE.get("X_test"),
                SESSION_STATE.get("y_test"),
                SESSION_STATE.get("X_train"),
                SESSION_STATE.get("y_train"),
            ):
                sheets.append((sheet_name, sheet_data))
            sheets.append(("Train_Raw_With_Prediction", _series_with_prediction(model, SESSION_STATE.get("X_train"), SESSION_STATE.get("y_train"), "train")))
            sheets.append(("Test_Raw_With_Prediction", _series_with_prediction(model, SESSION_STATE.get("X_test"), SESSION_STATE.get("y_test"), "test")))
            meta_history = SESSION_STATE.get("meta_histories", {}).get(model_name)
            if meta_history is not None:
                sheets.append(("Hyperparameter_History", meta_history))

        elif context == "shap":
            X_train = SESSION_STATE.get("X_train")
            sample_idx = int(payload.get("sample_idx", 0))
            metadata.update({"sample_idx": sample_idx, "best_model_name": SESSION_STATE.get("best_model_name")})
            sheets.append(("Train_Feature_Matrix", X_train))
            if X_train is not None and len(X_train) > 0:
                sample_idx = max(0, min(sample_idx, len(X_train) - 1))
                sheets.append(("Explained_Sample", X_train.iloc[[sample_idx]] if hasattr(X_train, "iloc") else _to_frame(X_train).iloc[[sample_idx]]))

        elif context == "pdp":
            X_train = SESSION_STATE.get("X_train")
            features = [payload.get("feature"), payload.get("feature2"), payload.get("feature3")]
            features = [f for f in features if f and f != "None"]
            metadata.update({"features": features, "best_model_name": SESSION_STATE.get("best_model_name")})
            sheets.append(("Train_Feature_Matrix", X_train))
            if X_train is not None and features:
                valid = [f for f in features if f in X_train.columns]
                if valid:
                    sheets.append(("PDP_Feature_Source", X_train[valid].copy()))

        elif context == "permutation":
            model = SESSION_STATE.get("best_model")
            metadata.update({"best_model_name": SESSION_STATE.get("best_model_name")})
            sheets.append(("Test_Raw_With_Prediction", _series_with_prediction(model, SESSION_STATE.get("X_test"), SESSION_STATE.get("y_test"), "test")))

        elif context == "counterfactual":
            metadata.update({"counterfactual_request": payload, "best_model_name": SESSION_STATE.get("best_model_name")})
            sheets.append(("Train_Feature_Matrix", SESSION_STATE.get("X_train")))
            if isinstance(payload.get("counterfactuals"), list):
                sheets.append(("Counterfactuals", payload.get("counterfactuals")))

        elif context == "causal":
            source = SESSION_STATE.get("df_clean") if SESSION_STATE.get("df_clean") is not None else df
            metadata.update({"treatment": payload.get("treatment")})
            sheets.append(("Causal_Source_Data", source))
            if isinstance(payload.get("summary_table"), list):
                sheets.append(("Effect_Summary_Table", payload.get("summary_table")))

        elif context == "optimization":
            result = SESSION_STATE.get("optimization_result") or {}
            metadata.update({"optimization_plot": plot_name, "best_algorithm": result.get("best_algorithm")})
            for key in ["solutions", "pareto", "history", "summary", "comparison_metrics", "shrinkage_curve_data"]:
                if key in result and result[key] is not None:
                    sheets.append((key, result[key]))
            if result.get("best_solution") is not None:
                sheets.append(("best_solution", result.get("best_solution")))
            if result.get("shrinkage_curve_config") is not None:
                sheets.append(("shrinkage_curve_config", result.get("shrinkage_curve_config")))

        elif context == "figure":
            clean_payload = _strip_data_urls(payload)
            metadata.update({"figure_request": clean_payload})
            sheets.append(("Figure_Panels", clean_payload.get("panels", []) if isinstance(clean_payload, dict) else []))

        elif context == "audit":
            audit = SESSION_STATE.get("audit_result") or {}
            metadata.update({"audit_plot": plot_name})
            source = (audit.get("plot_data") or {}).get(plot_name)
            if source is not None:
                sheets.append(("Plot_Source_Data", source))
            for key, rows in (audit.get("tables") or {}).items():
                sheets.append((key, rows))

        else:
            if df is None:
                return JSONResponse(status_code=400, content={"status": "error", "message": "No dataset available."})
            sheets.append(("Full_Dataset", df.copy()))

        return _write_workbook(f"{context}_{plot_name}_data", metadata, sheets)
    except Exception as e:
        return {"status": "error", "message": f"Excel data export failed: {str(e)}"}


@app.post("/api/export/zip")
async def export_zip_bundle(request: Request):
    try:
        data = await request.json()
        bundle_name = _slugify_filename(data.get("bundle_name", "NatureML_Export"))
        items = data.get("items", [])
        if not items:
            return JSONResponse(status_code=400, content={"status": "error", "message": "No export items were provided."})

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            used_names = set()
            for idx, item in enumerate(items, start=1):
                data_url = item.get("data_url")
                if not data_url:
                    continue
                _, ext, payload = _decode_data_url(data_url)
                base_name = _slugify_filename(item.get("name", f"plot_{idx}"))
                file_name = f"{base_name}.{ext}"
                suffix = 2
                while file_name in used_names:
                    file_name = f"{base_name}_{suffix}.{ext}"
                    suffix += 1
                used_names.add(file_name)
                zf.writestr(file_name, payload)

        zip_b64 = base64.b64encode(zip_buffer.getvalue()).decode("utf-8")
        return {
            "status": "success",
            "filename": f"{bundle_name}.zip",
            "content_base64": zip_b64,
        }
    except Exception as e:
        return {"status": "error", "message": f"ZIP export failed: {str(e)}"}

# Mount static files to serve the Vue frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
