import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearn.base import clone
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.ensemble import (
        AdaBoostClassifier,
        AdaBoostRegressor,
        BaggingClassifier,
        BaggingRegressor,
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
        StackingClassifier,
        StackingRegressor,
        VotingClassifier,
        VotingRegressor,
    )
    from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import (
        ARDRegression,
        BayesianRidge,
        ElasticNet,
        ElasticNetCV,
        GammaRegressor,
        HuberRegressor,
        Lasso,
        LassoLars,
        Lars,
        LinearRegression,
        LogisticRegression,
        OrthogonalMatchingPursuit,
        PassiveAggressiveClassifier,
        PassiveAggressiveRegressor,
        PoissonRegressor,
        QuantileRegressor,
        RANSACRegressor,
        Ridge,
        RidgeClassifier,
        TheilSenRegressor,
        TweedieRegressor,
    )
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.naive_bayes import BernoulliNB, GaussianNB
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier, RadiusNeighborsRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.svm import LinearSVR, LinearSVC, NuSVC, NuSVR, SVC, SVR
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


def _fill_missing_frame(df):
    work_df = df.copy()
    for col in work_df.columns:
        series = work_df[col]
        if pd.api.types.is_numeric_dtype(series):
            fill_value = series.median()
            if pd.isna(fill_value):
                fill_value = 0.0
            work_df[col] = series.fillna(fill_value)
        else:
            non_null = series.dropna()
            fill_value = non_null.mode().iloc[0] if not non_null.empty else "Missing"
            work_df[col] = series.fillna(fill_value)
    return work_df


def prepare_design_matrix(df, target_col=None, selected_features=None, feature_columns=None):
    work_df = df.copy()
    if target_col and target_col in work_df.columns:
        work_df = work_df.drop(columns=[target_col])
    if selected_features:
        valid_cols = [c for c in selected_features if c in work_df.columns]
        if valid_cols:
            work_df = work_df[valid_cols]
    work_df = _fill_missing_frame(work_df)

    encoded = pd.get_dummies(work_df, drop_first=True)
    encoded = encoded.apply(pd.to_numeric, errors="coerce")
    encoded = encoded.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if feature_columns is not None:
        encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    return encoded


def _sanitize_ml_matrix(X, y, ptype):
    X_clean = X.copy()
    if isinstance(X_clean, pd.DataFrame):
        X_clean = X_clean.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        X_clean = np.nan_to_num(np.asarray(X_clean, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    y_series = y.copy() if isinstance(y, pd.Series) else pd.Series(y)
    y_series = y_series.replace([np.inf, -np.inf], np.nan)
    if ptype == "Classification":
        y_series = y_series.fillna("Missing").astype(str)
    else:
        y_series = pd.to_numeric(y_series, errors="coerce")
        valid_mask = y_series.notna()
        if isinstance(X_clean, pd.DataFrame):
            X_clean = X_clean.loc[valid_mask].reset_index(drop=True)
        else:
            X_clean = X_clean[np.asarray(valid_mask)]
        y_series = y_series.loc[valid_mask].reset_index(drop=True)
    return X_clean, y_series


def _matrix_diag(X, y=None, label="X"):
    try:
        if isinstance(X, pd.DataFrame):
            x_nan = int(X.isna().sum().sum())
            x_inf = int(np.isinf(X.select_dtypes(include=[np.number]).to_numpy()).sum()) if not X.select_dtypes(include=[np.number]).empty else 0
            shape = tuple(X.shape)
            dtypes = sorted({str(v) for v in X.dtypes})
        else:
            arr = np.asarray(X)
            x_nan = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.number) else 0
            x_inf = int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.number) else 0
            shape = tuple(arr.shape)
            dtypes = [str(arr.dtype)]
    except Exception:
        x_nan, x_inf, shape, dtypes = -1, -1, "unknown", ["unknown"]

    message = f"{label} shape={shape}, NaN={x_nan}, Inf={x_inf}, dtypes={dtypes}"
    if y is not None:
        try:
            y_series = y if isinstance(y, pd.Series) else pd.Series(y)
            y_nan = int(y_series.isna().sum())
            message += f", y NaN={y_nan}, y dtype={str(y_series.dtype)}"
        except Exception:
            message += ", y diagnostics unavailable"
    return message


def predict_with_training_schema(model, candidate_df, feature_columns, target_col=None, selected_features=None):
    X_candidate = prepare_design_matrix(candidate_df, target_col, selected_features, feature_columns)
    return model.predict(X_candidate)


def _cv_score(model, X, y, ptype):
    scoring = "accuracy" if ptype == "Classification" else "r2"
    try:
        scores = cross_val_score(clone(model), X, y, cv=3, scoring=scoring, n_jobs=1)
        return float(scores.mean()), float(scores.std())
    except Exception:
        return None, None


def get_models(ptype="Classification"):
    if not HAS_SKLEARN:
        return {}

    if ptype == "Classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1500),
            "Logistic Regression (L1)": LogisticRegression(max_iter=1500, penalty="l1", solver="liblinear"),
            "Ridge Classifier": RidgeClassifier(),
            "Passive Aggressive": PassiveAggressiveClassifier(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Radius Neighbors": RadiusNeighborsClassifier(outlier_label="most_frequent"),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Extra Trees": ExtraTreesClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "HistGradientBoosting": HistGradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "Bagging Classifier": BaggingClassifier(),
            "MLP (Neural Network)": MLPClassifier(max_iter=1200),
            "SVC (Support Vector)": SVC(probability=True),
            "Nu-SVC": NuSVC(probability=True),
            "Linear SVC": LinearSVC(),
            "Gaussian Process": GaussianProcessClassifier(),
            "Gaussian Naive Bayes": GaussianNB(),
            "Bernoulli Naive Bayes": BernoulliNB(),
            "Linear Discriminant": LinearDiscriminantAnalysis(),
            "Quadratic Discriminant": QuadraticDiscriminantAnalysis(),
        }
        if HAS_IMBLEARN:
            models["Balanced Random Forest"] = BalancedRandomForestClassifier(random_state=42)
        if HAS_XGBOOST:
            models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        if HAS_LIGHTGBM:
            models["LightGBM"] = LGBMClassifier(verbose=-1)
        if HAS_CATBOOST:
            models["CatBoost"] = CatBoostClassifier(verbose=0)
        return models

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "ElasticNetCV": ElasticNetCV(cv=3),
        "Bayesian Ridge": BayesianRidge(),
        "ARD Regression": ARDRegression(),
        "Huber Regressor": HuberRegressor(),
        "TheilSen Regressor": TheilSenRegressor(max_iter=300),
        "Poisson Regressor": PoissonRegressor(),
        "Tweedie Regressor": TweedieRegressor(),
        "Gamma Regressor": GammaRegressor(),
        "Passive Aggressive": PassiveAggressiveRegressor(),
        "Quantile Regressor": QuantileRegressor(alpha=0.0, solver="highs"),
        "K-Nearest Neighbors": KNeighborsRegressor(),
        "Radius Neighbors": RadiusNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Extra Trees": ExtraTreesRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "HistGradientBoosting": HistGradientBoostingRegressor(),
        "AdaBoost": AdaBoostRegressor(),
        "Bagging Regressor": BaggingRegressor(),
        "MLP (Neural Network)": MLPRegressor(max_iter=1200),
        "SVR (Support Vector)": SVR(),
        "Nu-SVR": NuSVR(),
        "Linear SVR": LinearSVR(),
        "Gaussian Process": GaussianProcessRegressor(),
        "Kernel Ridge": KernelRidge(),
        "PLS Regression": PLSRegression(n_components=2),
        "Orthogonal Matching Pursuit": OrthogonalMatchingPursuit(),
        "RANSAC Regressor": RANSACRegressor(random_state=42),
        "LassoLars": LassoLars(),
        "Lars": Lars(),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor()
    if HAS_LIGHTGBM:
        models["LightGBM"] = LGBMRegressor(verbose=-1)
    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostRegressor(verbose=0)
    return models


def get_model_registry(ptype="Classification"):
    return list(get_models(ptype).keys())


def _model_supports_meta(model_name):
    supported = ["Random Forest", "Extra Trees", "Gradient Boosting", "HistGradientBoosting", "XGBoost", "LightGBM", "CatBoost"]
    return any(token in str(model_name) for token in supported)


def _apply_meta_params(model, model_name, n_est, mdepth):
    params = model.get_params() if hasattr(model, "get_params") else {}
    updates = {}
    if "n_estimators" in params:
        updates["n_estimators"] = int(n_est)
    if "max_depth" in params:
        updates["max_depth"] = int(mdepth)
    if "max_leaf_nodes" in params and "HistGradientBoosting" in str(model_name):
        updates["max_leaf_nodes"] = int(np.clip(2 ** min(int(mdepth), 8), 8, 255))
    if "num_leaves" in params:
        updates["num_leaves"] = int(np.clip(2 ** min(int(mdepth), 8), 8, 255))
    if "iterations" in params:
        updates["iterations"] = int(n_est)
    if "depth" in params:
        updates["depth"] = int(np.clip(mdepth, 2, 10))
    if "random_state" in params:
        updates["random_state"] = 42
    if updates:
        model = clone(model).set_params(**updates)
    return model


def try_optimize_meta(X, y, ptype, meta_algo="PSO", base_model_name="Random Forest", base_model=None, epochs=18):
    def fitness_function(n_est, mdepth):
        try:
            if base_model is None:
                if ptype == "Classification":
                    model = RandomForestClassifier(n_estimators=n_est, max_depth=mdepth, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=n_est, max_depth=mdepth, random_state=42)
            else:
                model = _apply_meta_params(base_model, base_model_name, n_est, mdepth)
            score = cross_val_score(model, X, y, cv=3, scoring="accuracy" if ptype == "Classification" else "r2", n_jobs=1).mean()
        except Exception as exc:
            raise Exception(
                f"Meta optimization failed at n_estimators={n_est}, max_depth={mdepth}: {str(exc)} | "
                f"{_matrix_diag(X, y, 'meta_X')}"
            )
        return float(score)

    rng = np.random.default_rng(abs(hash(meta_algo)) % (2**32))
    best_score = -np.inf
    best_pair = (120, 8)
    history = []
    seed_grid = [(60, 4), (90, 8), (120, 12), (160, 6), (200, 14), (240, 10)]
    for epoch in range(1, int(epochs) + 1):
        if epoch <= len(seed_grid):
            n_est, mdepth = seed_grid[epoch - 1]
        else:
            base_est, base_depth = best_pair
            explore = 1.0 if epoch < 12 else 0.65
            n_est = int(np.clip(base_est + rng.normal(0, 42 * explore), 20, 320))
            mdepth = int(np.clip(base_depth + rng.normal(0, 6 * explore), 2, 36))
        score = fitness_function(n_est, mdepth)
        if score > best_score:
            best_score = score
            best_pair = (n_est, mdepth)
        history.append(
            {
                "epoch": epoch,
                "algorithm": meta_algo,
                "model": base_model_name,
                "n_estimators": int(n_est),
                "max_depth": int(mdepth),
                "score": float(score),
                "best_score": float(best_score),
            }
        )
    return int(best_pair[0]), int(best_pair[1]), history


def run_training_pipeline(df, target_col, selected_features, ptype, selected_models, use_meta=False, meta_algo="PSO", ensemble_method="None", test_size=0.2, meta_algos=None):
    if not HAS_SKLEARN:
        raise Exception("ML libraries missing. Please install scikit-learn and optional boosters.")

    work_df = df.copy()
    if target_col not in work_df.columns:
        raise Exception(f"Target column not found: {target_col}")
    work_df = work_df.replace([np.inf, -np.inf], np.nan)
    work_df = work_df.dropna(subset=[target_col]).reset_index(drop=True)
    if work_df.empty:
        raise Exception("Target column contains only missing values after cleaning.")

    if ptype == "Classification":
        mode_vals = work_df[target_col].dropna().mode()
        target_fill = mode_vals.iloc[0] if not mode_vals.empty else "Missing"
        work_df[target_col] = work_df[target_col].fillna(target_fill)
    else:
        numeric_target = pd.to_numeric(work_df[target_col], errors="coerce")
        valid_target = numeric_target.notna()
        work_df = work_df.loc[valid_target].reset_index(drop=True)
        numeric_target = numeric_target.loc[valid_target].reset_index(drop=True)
        if numeric_target.empty:
            raise Exception("Regression target becomes fully missing after numeric conversion.")
        work_df[target_col] = numeric_target

    y = work_df[target_col]
    X_base = prepare_design_matrix(work_df, target_col, selected_features)
    if X_base.empty:
        raise Exception("No usable feature columns available after preprocessing.")
    X_base, y = _sanitize_ml_matrix(X_base, y, ptype)
    if isinstance(X_base, pd.DataFrame) and int(X_base.isna().sum().sum()) > 0:
        raise Exception("Feature matrix still contains NaN after sanitization.")
    if isinstance(y, pd.Series) and int(y.isna().sum()) > 0:
        raise Exception("Target vector still contains NaN after sanitization.")
    if len(selected_models) == 0:
        raise Exception("No models selected for training.")
    X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=float(test_size), random_state=42)
    X_train, y_train = _sanitize_ml_matrix(X_train, y_train, ptype)
    X_test, y_test = _sanitize_ml_matrix(X_test, y_test, ptype)
    if isinstance(X_train, pd.DataFrame) and int(X_train.isna().sum().sum()) > 0:
        raise Exception(f"Training split still contains NaN after sanitization. | {_matrix_diag(X_train, y_train, 'X_train')}")
    if isinstance(X_test, pd.DataFrame) and int(X_test.isna().sum().sum()) > 0:
        raise Exception(f"Test split still contains NaN after sanitization. | {_matrix_diag(X_test, y_test, 'X_test')}")
    all_models = get_models(ptype)

    results = []
    trained_models = {}
    meta_histories = {}
    failed_models = []

    requested_jobs = [(model_name, model_name, None) for model_name in selected_models]
    if use_meta:
        algos = meta_algos if meta_algos else [meta_algo]
        for model_name in selected_models:
            if model_name in all_models and _model_supports_meta(model_name):
                for algo in algos:
                    requested_jobs.append((f"{algo}-{model_name}", model_name, algo))

    for display_name, model_name, tuning_algo in requested_jobs:
        if model_name not in all_models:
            continue
        try:
            model = all_models[model_name]

            if tuning_algo:
                n_est, mdepth, meta_history = try_optimize_meta(
                    X_train,
                    y_train,
                    ptype,
                    tuning_algo,
                    base_model_name=model_name,
                    base_model=model,
                    epochs=10 if len(requested_jobs) > len(selected_models) + 4 else 18,
                )
                model = _apply_meta_params(model, model_name, n_est, mdepth)
                meta_histories[display_name] = meta_history

            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0
            y_pred = model.predict(X_test)

            if ptype == "Classification":
                train_score = accuracy_score(y_train, model.predict(X_train))
                test_score = accuracy_score(y_test, y_pred)
                cv_mean, cv_std = _cv_score(model, X_train, y_train, ptype)
                results.append(
                    {
                        "Model": display_name,
                        "Accuracy": round(test_score, 4),
                        "F1": round(f1_score(y_test, y_pred, average="weighted"), 4),
                        "TrainAcc": round(train_score, 4),
                        "Gap": round(train_score - test_score, 4),
                        "CV Mean": round(cv_mean, 4) if cv_mean is not None else None,
                        "CV Std": round(cv_std, 4) if cv_std is not None else None,
                        "Time(s)": round(train_time, 2),
                    }
                )
            else:
                train_r2 = r2_score(y_train, model.predict(X_train))
                test_r2 = r2_score(y_test, y_pred)
                cv_mean, cv_std = _cv_score(model, X_train, y_train, ptype)
                results.append(
                    {
                        "Model": display_name,
                        "R2": round(test_r2, 4),
                        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
                        "TrainR2": round(train_r2, 4),
                        "Gap": round(train_r2 - test_r2, 4),
                        "CV Mean": round(cv_mean, 4) if cv_mean is not None else None,
                        "CV Std": round(cv_std, 4) if cv_std is not None else None,
                        "Time(s)": round(train_time, 2),
                    }
                )
            trained_models[display_name] = model
        except Exception as exc:
            failed_models.append(
                {
                    "Model": display_name,
                    "Stage": "train/eval",
                    "Error": (
                        f"{str(exc)} | "
                        f"{_matrix_diag(X_train, y_train, 'X_train')} | {_matrix_diag(X_test, y_test, 'X_test')}"
                    ),
                }
            )
            continue

    if ensemble_method != "None" and len(trained_models) > 1:
        estimators = [(name, model) for name, model in trained_models.items()]
        t0 = time.time()
        if ptype == "Classification":
            ensemble = VotingClassifier(estimators, voting="hard") if ensemble_method == "Voting" else StackingClassifier(estimators)
        else:
            ensemble = VotingRegressor(estimators) if ensemble_method == "Voting" else StackingRegressor(estimators)
        try:
            ensemble.fit(X_train, y_train)
            train_time = time.time() - t0
            y_pred = ensemble.predict(X_test)
        except Exception as exc:
            failed_models.append(
                {
                    "Model": f"{ensemble_method} Ensemble",
                    "Stage": "ensemble",
                    "Error": (
                        f"{str(exc)} | "
                        f"{_matrix_diag(X_train, y_train, 'X_train')} | {_matrix_diag(X_test, y_test, 'X_test')}"
                    ),
                }
            )
            ensemble = None
        ensemble_name = f"{ensemble_method} Ensemble"
        if ensemble is None:
            pass
        elif ptype == "Classification":
            train_score = accuracy_score(y_train, ensemble.predict(X_train))
            test_score = accuracy_score(y_test, y_pred)
            results.append(
                {
                    "Model": ensemble_name,
                    "Accuracy": round(test_score, 4),
                    "F1": round(f1_score(y_test, y_pred, average="weighted"), 4),
                    "TrainAcc": round(train_score, 4),
                    "Gap": round(train_score - test_score, 4),
                    "CV Mean": None,
                    "CV Std": None,
                    "Time(s)": round(train_time, 2),
                }
            )
        else:
            train_r2 = r2_score(y_train, ensemble.predict(X_train))
            test_r2 = r2_score(y_test, y_pred)
            results.append(
                {
                    "Model": ensemble_name,
                    "R2": round(test_r2, 4),
                    "RMSE": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
                    "TrainR2": round(train_r2, 4),
                    "Gap": round(train_r2 - test_r2, 4),
                    "CV Mean": None,
                    "CV Std": None,
                    "Time(s)": round(train_time, 2),
                }
            )
        trained_models[ensemble_name] = ensemble

    sort_key = "Accuracy" if ptype == "Classification" else "R2"
    if not results:
        failure_summary = "; ".join([f"{item['Model']}: {item['Error']}" for item in failed_models[:5]]) or "No successful models."
        raise Exception(f"All selected models failed. {failure_summary}")
    results = sorted(results, key=lambda item: item[sort_key], reverse=True)
    best_model_name = results[0]["Model"]
    best_model = trained_models[best_model_name]

    return results, {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "best_model": best_model,
        "best_model_name": best_model_name,
        "trained_models": trained_models,
        "meta_histories": meta_histories,
        "failed_models": failed_models,
        "feature_columns": X_base.columns.tolist(),
        "base_feature_columns": [c for c in df.columns if c != target_col],
    }
