"""
Sanity check inspired by the Principal Odor Map paper's striking finding:
pairs of molecules that are STRUCTURALLY dissimilar (low fingerprint
Tanimoto similarity) can still SMELL similar (similar descriptor profiles).

We look, among the dataset molecules, for pairs with:
  * Tanimoto similarity of Morgan fingerprints <= 0.25  (different structures)
  * cosine similarity of the model's predicted descriptor profiles >= 0.90
and print the best examples together with their ground-truth shared labels.
Run after train.py (uses smell_model.joblib).
"""

import numpy as np
import pandas as pd

from features import FP_SIZE, featurize
from predict import load_bundle

TANIMOTO_MAX = 0.25
PROFILE_COS_MIN = 0.90


def batch_profiles(bundle, X):
    """Predicted descriptor profiles for all rows of X at once.

    (Batched on purpose: calling predict_proba molecule-by-molecule through
    the one-vs-rest forests is orders of magnitude slower.)
    """
    Xs = X.copy()
    Xs[:, FP_SIZE:] = (Xs[:, FP_SIZE:] - bundle["scaler_mean"]) / bundle["scaler_scale"]
    if bundle["model_name"] == "tanimoto_knn":
        raise SystemExit("similarity_check currently expects a sklearn model bundle")
    return np.column_stack(
        [est.predict_proba(Xs)[:, 1] for est in bundle["model"].estimators_])


def main():
    bundle = load_bundle()
    df = pd.read_csv("data/molecule_labels.csv")
    labels = bundle["labels"]
    X, _ = featurize(df["SMILES"])
    fps = X[:, :FP_SIZE]

    # predicted profile for every dataset molecule
    P = batch_profiles(bundle, X)

    # pairwise Tanimoto (structure) and cosine (predicted smell profile)
    inter = fps @ fps.T
    denom = fps.sum(1, keepdims=True) + fps.sum(1) - inter
    tanimoto = inter / np.maximum(denom, 1e-9)
    pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
    cos = pn @ pn.T

    n = len(df)
    hits = []
    for i in range(n):
        for j in range(i + 1, n):
            if tanimoto[i, j] <= TANIMOTO_MAX and cos[i, j] >= PROFILE_COS_MIN:
                shared = [lab for lab in labels
                          if df.iloc[i][lab] == 1 and df.iloc[j][lab] == 1]
                hits.append((cos[i, j], tanimoto[i, j], i, j, shared))
    hits.sort(reverse=True)

    print(f"searched {n * (n - 1) // 2} molecule pairs; "
          f"{len(hits)} pairs with Tanimoto <= {TANIMOTO_MAX} and "
          f"predicted-profile cosine >= {PROFILE_COS_MIN}\n")
    for c, t, i, j, shared in hits[:3]:
        a, b = df.iloc[i], df.iloc[j]
        print(f"* {a['Name']}  <->  {b['Name']}")
        print(f"    structural Tanimoto similarity: {t:.2f} (dissimilar)")
        print(f"    predicted smell-profile cosine: {c:.3f} (very similar)")
        print(f"    ground-truth labels they share: "
              f"{', '.join(shared) if shared else '(none)'}")
        print(f"    A: {a['SMILES']}")
        print(f"    B: {b['SMILES']}\n")
    if not hits:
        print("No pairs passed both thresholds - see README for discussion.")


if __name__ == "__main__":
    main()
