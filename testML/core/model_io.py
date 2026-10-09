import base64
import io
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

import joblib


PACKAGE_VERSION = "1.1"


def _safe_params(obj):
    if obj is None or not hasattr(obj, "get_params"):
        return {}
    try:
        return obj.get_params(deep=True)
    except Exception:
        return {}


def _dependency_versions():
    packages = ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost", "joblib"]
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def build_model_package(
    model_name,
    model,
    problem_type=None,
    feature_columns=None,
    selected_features=None,
    target_col=None,
    base_feature_columns=None,
    metrics=None,
    training_config=None,
    meta_history=None,
):
    return {
        "package_type": "natureml_model",
        "package_version": PACKAGE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dependency_versions": _dependency_versions(),
        "model_name": model_name,
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}" if model is not None else None,
        "model_params": _safe_params(model),
        "model": model,
        "problem_type": problem_type,
        "target_col": target_col,
        "feature_columns": feature_columns or [],
        "selected_features": selected_features or [],
        "base_feature_columns": base_feature_columns or [],
        "metrics": metrics or {},
        "training_config": training_config or {},
        "meta_history": meta_history,
    }


def build_optimization_package(
    name,
    request_config,
    optimization_result,
    main_model_package=None,
    surrogate_packages=None,
    dataset_context=None,
):
    return {
        "package_type": "natureml_multiobjective_optimization",
        "package_version": PACKAGE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dependency_versions": _dependency_versions(),
        "name": name,
        "request_config": request_config or {},
        "best_algorithm": (optimization_result or {}).get("best_algorithm"),
        "requested_algorithm": (optimization_result or {}).get("requested_algorithm"),
        "selected_algorithm": (optimization_result or {}).get("selected_algorithm"),
        "metrics": (optimization_result or {}).get("metrics", {}),
        "comparison_metrics": (optimization_result or {}).get("comparison_metrics", []),
        "algorithm_selection_metrics": (optimization_result or {}).get("algorithm_selection_metrics", []),
        "best_solution": (optimization_result or {}).get("best_solution"),
        "solutions": (optimization_result or {}).get("solutions", []),
        "history": (optimization_result or {}).get("history", []),
        "objective_names": (optimization_result or {}).get("objective_names", []),
        "shrinkage_curve_data": (optimization_result or {}).get("shrinkage_curve_data", []),
        "shrinkage_curve_config": (optimization_result or {}).get("shrinkage_curve_config", {}),
        "main_model_package": main_model_package,
        "surrogate_packages": surrogate_packages or {},
        "dataset_context": dataset_context or {},
    }


def dump_model_package(package):
    buffer = io.BytesIO()
    joblib.dump(package, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def load_model_package_from_bytes(raw_bytes):
    buffer = io.BytesIO(raw_bytes)
    return joblib.load(buffer)
