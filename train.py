"""
Train and honestly compare multi-label odor-descriptor models.

Methodology (no test-set peeking):
  * One fixed random split (seed 42): 64% train / 16% validation / 20% test.
  * Every candidate model is fitted on TRAIN only.
  * The winner is chosen by macro-AUROC on the VALIDATION split.
  * Only after the winner is fixed is the TEST split touched - once - to
    report final numbers for every candidate.
  * The saved model (smell_model.joblib) is the winning candidate, fitted on
    the train split. predict.py uses it as-is.

Candidates: logistic regression (3 values of C), random forest, a small MLP,
and a fingerprint-similarity nearest-neighbour baseline.
Results are written to results.json.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from features import FP_SIZE, featurize

SEED = 42
MIN_LABEL_POSITIVES = 30  # keep labels with at least this many positives overall


# --------------------------------------------------------------------------- #
# a tiny similarity-based baseline (not sklearn - handled specially)          #
# --------------------------------------------------------------------------- #
class TanimotoKNN:
    """Predict a label score as the Tanimoto-similarity-weighted fraction of
    the k most similar training molecules carrying that label."""

    def __init__(self, k=20):
        self.k = k

    def fit(self, X_fp, y):
        self.X_fp = X_fp
        self.y = y
        return self

    def predict_proba(self, X_fp):
        a = X_fp @ self.X_fp.T                       # intersection sizes
        denom = X_fp.sum(1, keepdims=True) + self.X_fp.sum(1) - a
        sim = np.where(denom > 0, a / np.maximum(denom, 1e-9), 0.0)
        out = np.zeros((X_fp.shape[0], self.y.shape[1]))
        for i in range(X_fp.shape[0]):
            idx = np.argsort(sim[i])[::-1][: self.k]
            w = sim[i][idx]
            if w.sum() == 0:
                out[i] = self.y.mean(0)
            else:
                out[i] = (w @ self.y[idx]) / w.sum()
        return out


def macro_auroc(y_true, scores):
    """Mean per-label AUROC; labels lacking positives or negatives are skipped."""
    per_label = {}
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if col.sum() == 0 or col.sum() == len(col):
            per_label[j] = None
        else:
            per_label[j] = float(roc_auc_score(col, scores[:, j]))
    vals = [v for v in per_label.values() if v is not None]
    return float(np.mean(vals)), per_label


def ovr_proba(model, X):
    """OneVsRestClassifier -> (n, n_labels) probability matrix."""
    return np.column_stack([est.predict_proba(X)[:, 1] for est in model.estimators_])


def main():
    df = pd.read_csv("data/molecule_labels.csv")
    all_labels = [c for c in df.columns if c not in ("SMILES", "Name")]
    counts = df[all_labels].sum()
    labels = [c for c in all_labels if counts[c] >= MIN_LABEL_POSITIVES]
    dropped = {c: int(counts[c]) for c in all_labels if c not in labels}
    print(f"{len(df)} molecules | keeping {len(labels)} labels "
          f"(>= {MIN_LABEL_POSITIVES} positives); dropped: {dropped}")

    X, valid = featurize(df["SMILES"])
    assert valid.all(), "all dataset SMILES should parse - prepare_data checked this"
    y = df[labels].to_numpy(dtype=int)

    # scale only the 8 descriptor columns (fingerprint bits stay 0/1)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(df))
    n_test = int(round(0.20 * len(df)))
    n_val = int(round(0.16 * len(df)))
    test_idx, val_idx, train_idx = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
    print(f"split sizes: train {len(train_idx)} | val {len(val_idx)} | test {len(test_idx)}")

    scaler = StandardScaler().fit(X[train_idx, FP_SIZE:])
    Xs = X.copy()
    Xs[:, FP_SIZE:] = scaler.transform(X[:, FP_SIZE:])

    Xtr, ytr = Xs[train_idx], y[train_idx]
    Xva, yva = Xs[val_idx], y[val_idx]
    Xte, yte = Xs[test_idx], y[test_idx]

    candidates = {}
    for C in (0.1, 1.0, 10.0):
        candidates[f"logreg_C{C}"] = OneVsRestClassifier(
            LogisticRegression(C=C, class_weight="balanced", max_iter=2000,
                               random_state=SEED)).fit(Xtr, ytr)
    candidates["random_forest"] = OneVsRestClassifier(
        RandomForestClassifier(n_estimators=400, n_jobs=-1,
                               random_state=SEED)).fit(Xtr, ytr)
    for alpha in (0.1, 1.0):
        candidates[f"mlp_alpha{alpha}"] = OneVsRestClassifier(
            MLPClassifier(hidden_layer_sizes=(256,), alpha=alpha, max_iter=600,
                          early_stopping=True, random_state=SEED)).fit(Xtr, ytr)
    knn = TanimotoKNN(k=20).fit(X[train_idx, :FP_SIZE], ytr)
    candidates["tanimoto_knn"] = knn

    def scores_for(name, model, Xq, Xq_fp):
        if name == "tanimoto_knn":
            return model.predict_proba(Xq_fp)
        return ovr_proba(model, Xq)

    # ---- model selection on validation ------------------------------------ #
    val_results = {}
    for name, model in candidates.items():
        s = scores_for(name, model, Xva, X[val_idx, :FP_SIZE])
        macro, _ = macro_auroc(yva, s)
        val_results[name] = macro
        print(f"  val macro-AUROC  {name:18s} {macro:.4f}")

    winner = max(val_results, key=val_results.get)
    print(f"winner on validation: {winner} ({val_results[winner]:.4f})")

    # ---- single final evaluation on test ----------------------------------- #
    test_results, per_label_test = {}, {}
    for name, model in candidates.items():
        s = scores_for(name, model, Xte, X[test_idx, :FP_SIZE])
        macro, per = macro_auroc(yte, s)
        test_results[name] = macro
        per_label_test[name] = {labels[j]: per[j] for j in range(len(labels))}
        print(f"  test macro-AUROC {name:18s} {macro:.4f}")

    # ---- save the winning model + everything predict.py needs -------------- #
    joblib.dump(
        {
            "model_name": winner,
            "model": candidates[winner],
            "labels": labels,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "fp_size": FP_SIZE,
            "knn_train_fp": X[train_idx, :FP_SIZE] if winner == "tanimoto_knn" else None,
            "knn_train_y": ytr if winner == "tanimoto_knn" else None,
        },
        "smell_model.joblib",
    )

    results = {
        "dataset": {
            "name": "Keller & Vosshall 2016 (DREAM olfaction data), via pyrfume-data GitHub repo",
            "n_molecules": int(len(df)),
            "n_labels_total": len(all_labels),
            "n_labels_used": len(labels),
            "labels_used": labels,
            "labels_dropped_too_few_positives": dropped,
            "label_positive_counts": {c: int(counts[c]) for c in labels},
            "binarisation": "descriptor applied in >=15% of pooled subject ratings",
        },
        "split": {"seed": SEED, "train": int(len(train_idx)),
                  "validation": int(len(val_idx)), "test": int(len(test_idx))},
        "validation_macro_auroc": val_results,
        "test_macro_auroc": test_results,
        "winner_by_validation": winner,
        "test_per_label_auroc": per_label_test,
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved smell_model.joblib and results.json")


if __name__ == "__main__":
    main()
