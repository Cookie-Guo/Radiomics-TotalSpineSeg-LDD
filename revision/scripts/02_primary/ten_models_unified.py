#!/usr/bin/env python3
"""
Ten classifiers on the primary 517 3D features and the patient-level split.
CatBoost hyperparameters are fixed to the primary (macro AUC 0.936).
Writes revision/results/02_primary/ten_models_test_performance.csv.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "02_primary"
DATA = ROOT / "data"
SPLITS = ROOT / "splits" / "assignments.csv"
EXTRACTED = DATA / "extracted_data.xlsx"
LABELS = DATA / "labels.csv"
FEATS = OUT / "selected_features_3d.csv"

SEED = 4321
N_BOOT = 1000
CLASSES = [1, 2, 3, 4, 5]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def macro_auc_ovr(y_true, proba) -> float:
    yb = label_binarize(y_true, classes=CLASSES)
    aucs = []
    for i in range(5):
        if yb[:, i].sum() in (0, len(yb)):
            continue
        aucs.append(roc_auc_score(yb[:, i], proba[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")


def metrics_block(y_true, y_pred, proba) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    recalls, specs = [], []
    for i in range(5):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        recalls.append(tp / (tp + fn) if (tp + fn) else 0.0)
        specs.append(tn / (tn + fp) if (tn + fp) else 0.0)
    return {
        "macro_AUC": macro_auc_ovr(y_true, proba),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_sensitivity": float(np.mean(recalls)),
        "macro_specificity": float(np.mean(specs)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "kappa_quadratic": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
    }


def bootstrap_ci(y_true, y_pred, proba, n_boot=N_BOOT, seed=SEED) -> dict:
    rng = np.random.default_rng(seed)
    keys = list(metrics_block(y_true, y_pred, proba).keys())
    store = {k: [] for k in keys}
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = metrics_block(y_true[idx], y_pred[idx], proba[idx])
        for k in keys:
            if np.isfinite(m[k]):
                store[k].append(m[k])
    out = {}
    for k in keys:
        arr = np.asarray(store[k], float)
        out[f"{k}_ci_low"] = float(np.percentile(arr, 2.5)) if len(arr) else float("nan")
        out[f"{k}_ci_high"] = float(np.percentile(arr, 97.5)) if len(arr) else float("nan")
    return out


def proba_5(model, X, classes_order) -> np.ndarray:
    """Align predict_proba to columns for grades 1..5."""
    raw = model.predict_proba(X)
    out = np.zeros((len(X), 5))
    for j, c in enumerate(classes_order):
        out[:, int(c) - 1] = raw[:, j]
    return out


def load_matrix():
    assign = pd.read_csv(SPLITS)
    labels = pd.read_csv(LABELS)
    ex = pd.read_excel(EXTRACTED)
    assert len(ex) == len(labels)
    feats = pd.read_csv(FEATS)["Feature"].tolist()
    missing = [f for f in feats if f not in ex.columns]
    if missing:
        raise SystemExit(f"missing features in extracted_data: {missing[:5]}")
    df = labels.copy()
    df = df.merge(assign[["disc_id", "holdout", "cv_rep01"]], on="disc_id", how="left")
    for f in feats:
        df[f] = pd.to_numeric(ex[f].values, errors="coerce")
    return df, feats


def patient_fold_splits(train_df: pd.DataFrame):
    """Yield (tr_idx, va_idx) positional indices into train_df for cv_rep01 folds."""
    folds = sorted(train_df["cv_rep01"].dropna().unique())
    for f in folds:
        va = np.where(train_df["cv_rep01"].values == f)[0]
        tr = np.where(train_df["cv_rep01"].values != f)[0]
        if len(va) and len(tr):
            yield tr, va


def cv_macro_auc(model_factory, X, y, train_df) -> float:
    scores = []
    for tr, va in patient_fold_splits(train_df):
        m = model_factory()
        m.fit(X[tr], y[tr])
        if hasattr(m, "classes_"):
            classes = m.classes_
        else:
            classes = np.unique(y[tr])
        # pipeline
        if hasattr(m, "named_steps"):
            classes = m.named_steps[list(m.named_steps)[-1]].classes_
        proba = proba_5(m, X[va], classes)
        pred = proba.argmax(axis=1) + 1
        scores.append(macro_auc_ovr(y[va], proba))
    return float(np.nanmean(scores)) if scores else float("nan")


def tune_and_fit(name: str, Xtr, ytr, train_df, sample_weight=None):
    """Return fitted model and best params dict."""
    rng = SEED

    if name == "DecisionTree":
        grid = [0.0, 0.01, 0.02, 0.05, 0.1]
        best, best_p = -1, 0.01
        for cp in grid:
            # sklearn uses ccp_alpha roughly analogous
            fac = lambda a=cp: DecisionTreeClassifier(
                ccp_alpha=a, random_state=rng, class_weight="balanced"
            )
            s = cv_macro_auc(fac, Xtr, ytr, train_df)
            if s > best:
                best, best_p = s, cp
        model = DecisionTreeClassifier(
            ccp_alpha=best_p, random_state=rng, class_weight="balanced"
        )
        model.fit(Xtr, ytr)
        return model, {"ccp_alpha": best_p, "cv_macro_AUC": best}

    if name == "RandomForest":
        grid = [10, 20, 30, 40]
        p = Xtr.shape[1]
        grid = [min(m, p) for m in grid]
        best, best_m = -1, 20
        for mtry in grid:
            fac = lambda m=mtry: RandomForestClassifier(
                n_estimators=300, max_features=m, random_state=rng,
                class_weight="balanced_subsample", n_jobs=-1,
            )
            s = cv_macro_auc(fac, Xtr, ytr, train_df)
            if s > best:
                best, best_m = s, mtry
        model = RandomForestClassifier(
            n_estimators=300, max_features=best_m, random_state=rng,
            class_weight="balanced_subsample", n_jobs=-1,
        )
        model.fit(Xtr, ytr)
        return model, {"max_features": best_m, "cv_macro_AUC": best}

    if name == "XGBoost":
        if not HAS_XGB:
            raise RuntimeError("xgboost not installed")
        y0 = ytr - 1
        grid = [
            dict(max_depth=3, learning_rate=0.1, n_estimators=200),
            dict(max_depth=4, learning_rate=0.05, n_estimators=300),
            dict(max_depth=2, learning_rate=0.1, n_estimators=150),
        ]
        best, best_p = -1, grid[0]
        for p in grid:
            def fac(pp=p):
                return XGBClassifier(
                    objective="multi:softprob", num_class=5,
                    subsample=0.8, colsample_bytree=0.8,
                    min_child_weight=1, gamma=0,
                    random_state=rng, n_jobs=-1, verbosity=0,
                    **pp,
                )
            scores = []
            for tr, va in patient_fold_splits(train_df):
                m = fac()
                m.fit(Xtr[tr], y0[tr])
                proba = m.predict_proba(Xtr[va])  # cols = class 0..4 == grade 1..5
                scores.append(macro_auc_ovr(ytr[va], proba))
            s = float(np.mean(scores))
            if s > best:
                best, best_p = s, p
        model = XGBClassifier(
            objective="multi:softprob", num_class=5,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=1, gamma=0,
            random_state=rng, n_jobs=-1, verbosity=0,
            **best_p,
        )
        model.fit(Xtr, y0)
        return model, {**best_p, "cv_macro_AUC": best, "label_coding": "0..4 for grades 1..5"}

    if name == "Lasso":
        grid = np.logspace(-3, 0, 8)
        best, best_l = -1, 0.01
        for lam in grid:
            C = 1.0 / lam
            fac = lambda C=C: Pipeline([
                ("sc", StandardScaler()),
                ("clf", LogisticRegression(
                    penalty="l1", solver="saga", C=C, max_iter=5000,
                    class_weight="balanced", random_state=rng,
                )),
            ])
            s = cv_macro_auc(fac, Xtr, ytr, train_df)
            if s > best:
                best, best_l = s, lam
        model = Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l1", solver="saga", C=1.0 / best_l, max_iter=5000,
                class_weight="balanced", random_state=rng,
            )),
        ])
        model.fit(Xtr, ytr)
        return model, {"lambda": best_l, "cv_macro_AUC": best}

    if name == "Ridge":
        grid = np.logspace(-3, 1, 8)
        best, best_l = -1, 0.03
        for lam in grid:
            C = 1.0 / lam
            fac = lambda C=C: Pipeline([
                ("sc", StandardScaler()),
                ("clf", LogisticRegression(
                    penalty="l2", solver="lbfgs", C=C, max_iter=5000,
                    class_weight="balanced", random_state=rng,
                )),
            ])
            s = cv_macro_auc(fac, Xtr, ytr, train_df)
            if s > best:
                best, best_l = s, lam
        model = Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2", solver="lbfgs", C=1.0 / best_l, max_iter=5000,
                class_weight="balanced", random_state=rng,
            )),
        ])
        model.fit(Xtr, ytr)
        return model, {"lambda": best_l, "cv_macro_AUC": best}

    if name == "NeuralNetwork":
        grid = [(5, 0.5), (10, 0.1), (20, 0.01)]
        best, best_p = -1, (5, 0.5)
        for size, decay in grid:
            fac = lambda s=size, d=decay: Pipeline([
                ("sc", StandardScaler()),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(s,), alpha=d, max_iter=500,
                    random_state=rng, early_stopping=True,
                )),
            ])
            s = cv_macro_auc(fac, Xtr, ytr, train_df)
            if s > best:
                best, best_p = s, (size, decay)
        model = Pipeline([
            ("sc", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(best_p[0],), alpha=best_p[1], max_iter=500,
                random_state=rng, early_stopping=True,
            )),
        ])
        model.fit(Xtr, ytr)
        return model, {"size": best_p[0], "decay": best_p[1], "cv_macro_AUC": best}

    if name == "MultinomialLogistic":
        grid = [0.0, 0.1, 0.5, 1.0]
        best, best_d = -1, 0.5
        for decay in grid:
            # alpha ~ decay; use C = 1/(decay+1e-6) style
            C = 1.0 / (decay + 1e-3)
            fac = lambda C=C: Pipeline([
                ("sc", StandardScaler()),
                ("clf", LogisticRegression(
                    penalty="l2", solver="lbfgs", C=C, max_iter=5000,
                    class_weight="balanced", random_state=rng,
                )),
            ])
            s = cv_macro_auc(fac, Xtr, ytr, train_df)
            if s > best:
                best, best_d = s, decay
        C = 1.0 / (best_d + 1e-3)
        model = Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2", solver="lbfgs", C=C, max_iter=5000,
                class_weight="balanced", random_state=rng,
            )),
        ])
        model.fit(Xtr, ytr)
        return model, {"decay": best_d, "cv_macro_AUC": best}

    if name == "KNN":
        grid = [3, 5, 7, 9, 11]
        best, best_k = -1, 7
        for k in grid:
            fac = lambda k=k: Pipeline([
                ("sc", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=k)),
            ])
            s = cv_macro_auc(fac, Xtr, ytr, train_df)
            if s > best:
                best, best_k = s, k
        model = Pipeline([
            ("sc", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=best_k)),
        ])
        model.fit(Xtr, ytr)
        return model, {"k": best_k, "cv_macro_AUC": best}

    if name == "NaiveBayes":
        # sklearn GaussianNB; optional var_smoothing grid
        grid = [1e-9, 1e-8, 1e-7, 1e-6]
        best, best_v = -1, 1e-9
        for v in grid:
            fac = lambda v=v: Pipeline([
                ("sc", StandardScaler()),
                ("clf", GaussianNB(var_smoothing=v)),
            ])
            s = cv_macro_auc(fac, Xtr, ytr, train_df)
            if s > best:
                best, best_v = s, v
        model = Pipeline([
            ("sc", StandardScaler()),
            ("clf", GaussianNB(var_smoothing=best_v)),
        ])
        model.fit(Xtr, ytr)
        return model, {"var_smoothing": best_v, "cv_macro_AUC": best}

    if name == "CatBoost":
        bc = np.bincount(ytr, minlength=6)[1:6].astype(float)
        w = (bc.sum() / (5.0 * np.maximum(bc, 1))).tolist()
        cw = {i + 1: w[i] for i in range(5)}
        model = CatBoostClassifier(
            depth=2, learning_rate=0.05, l2_leaf_reg=1, iterations=223,
            loss_function="MultiClass", random_seed=SEED, verbose=False,
            allow_writing_files=False, class_weights=cw,
        )
        model.fit(Xtr, ytr)
        return model, {
            "depth": 2, "learning_rate": 0.05, "l2_leaf_reg": 1,
            "iterations": 223, "note": "fixed to match 02_primary",
        }

    raise ValueError(name)


def get_classes(model):
    if hasattr(model, "named_steps"):
        last = list(model.named_steps.values())[-1]
        return last.classes_
    return model.classes_


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if not FEATS.exists():
        raise SystemExit("Run unified_primary.py first")

    df, feats = load_matrix()
    train = df[df["holdout"] == "train"].copy().reset_index(drop=True)
    test = df[df["holdout"] == "test"].copy().reset_index(drop=True)

    med = train[feats].median()
    Xtr = train[feats].fillna(med).values.astype(float)
    Xte = test[feats].fillna(med).values.astype(float)
    ytr = train["pfirrmann"].astype(int).values
    yte = test["pfirrmann"].astype(int).values

    names = [
        "DecisionTree", "RandomForest", "XGBoost", "Lasso", "Ridge",
        "NeuralNetwork", "MultinomialLogistic", "KNN", "NaiveBayes", "CatBoost",
    ]
    if not HAS_XGB:
        print("WARNING: xgboost missing — will skip XGBoost")
        names = [n for n in names if n != "XGBoost"]

    rows = []
    param_rows = []
    class_rows = []

    for name in names:
        print(f"=== {name} ===")
        t1 = time.time()
        model, params = tune_and_fit(name, Xtr, ytr, train)
        if name == "XGBoost":
            # proba columns are grades 1..5 in order
            proba = model.predict_proba(Xte)
            pred = proba.argmax(axis=1) + 1
        else:
            classes = get_classes(model)
            proba = proba_5(model, Xte, classes)
            pred = proba.argmax(axis=1) + 1
        m = metrics_block(yte, pred, proba)
        m.update(bootstrap_ci(yte, pred, proba))
        m["Model"] = name
        m["seconds"] = round(time.time() - t1, 1)
        m["n_features"] = len(feats)
        rows.append(m)
        for k, v in params.items():
            param_rows.append({"Model": name, "Parameter": k, "Value": v})
        yb = label_binarize(yte, classes=CLASSES)
        for i, g in enumerate(CLASSES):
            n_pos = int(yb[:, i].sum())
            auc = float(roc_auc_score(yb[:, i], proba[:, i])) if 0 < n_pos < len(yte) else float("nan")
            class_rows.append({
                "Model": name, "Class": g, "AUC": auc,
                "N_test_pos": n_pos, "N_test_total": len(yte),
            })
        print(f"  macro_AUC={m['macro_AUC']:.4f}  acc={m['accuracy']:.4f}  ({m['seconds']}s)")

    perf = pd.DataFrame(rows)
    # column order
    front = ["Model", "macro_AUC", "macro_AUC_ci_low", "macro_AUC_ci_high",
             "accuracy", "macro_sensitivity", "macro_specificity",
             "kappa", "kappa_quadratic", "n_features", "seconds"]
    cols = front + [c for c in perf.columns if c not in front]
    perf = perf[cols].sort_values("macro_AUC", ascending=False)
    perf.to_csv(OUT / "ten_models_test_performance.csv", index=False)
    pd.DataFrame(param_rows).to_csv(OUT / "ten_models_best_parameters.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUT / "ten_models_class_auc.csv", index=False)

    # ranking
    perf[["Model", "macro_AUC"]].to_csv(OUT / "ten_models_ranking.csv", index=False)

    # assert CatBoost close to primary 0.936
    cb = perf.loc[perf["Model"] == "CatBoost", "macro_AUC"].iloc[0]
    primary = pd.read_csv(OUT / "primary_performance.csv")
    p3 = float(primary.loc[primary["config"] == "3D_primary", "macro_AUC"].iloc[0])
    if abs(cb - p3) > 0.005:
        print(f"WARNING: CatBoost {cb:.4f} vs primary 3D {p3:.4f} differ >0.005")
    else:
        print(f"OK: CatBoost {cb:.4f} matches primary 3D {p3:.4f}")

    meta = {
        "script": Path(__file__).name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "n_features": len(feats),
        "feature_list": str(FEATS),
        "split": str(SPLITS),
        "cv_column": "cv_rep01",
        "seconds": round(time.time() - t0, 1),
        "models": names,
        "has_xgboost": HAS_XGB,
        "catboost_macro_AUC": float(cb),
        "primary_3d_macro_AUC": p3,
        "inputs_md5": {
            "assignments.csv": md5(SPLITS),
            "selected_features_3d.csv": md5(FEATS),
            "extracted_data.xlsx": md5(EXTRACTED),
        },
        "policy": "All 10 models share unified 517-feature reduction; CatBoost fixed to primary hyperparameters",
    }
    (OUT / "ten_models.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(perf[["Model", "macro_AUC", "accuracy", "kappa_quadratic"]].to_string(index=False))
    print(f"Wrote {OUT / 'ten_models_test_performance.csv'}")


if __name__ == "__main__":
    main()
