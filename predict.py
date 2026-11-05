"""
Predict the smell of a molecule from its SMILES string.

Usage:
    python predict.py "COc1cc(C=O)ccc1O"          # vanillin
    python predict.py "CC1=CCC(CC1)C(=C)C"        # limonene
    python predict.py "c1ccccc1C=O"               # benzaldehyde

Prints the top predicted odor descriptors with probabilities as little bars.
The model is the validation-selected winner saved by train.py - see
README.md / results.json for how it was trained and how well it does.
This is a toy model trained on 474 molecules of open data, not Osmo's POM.
"""

import sys

import joblib
import numpy as np
import pandas as pd

from features import FP_SIZE, featurize, mol_from_smiles
from rdkit import Chem

TOP_N = 8
BAR_WIDTH = 30


def load_bundle():
    return joblib.load("smell_model.joblib")


def predict_scores(bundle, smiles):
    """Return a {label: probability} dict for one SMILES string."""
    X, valid = featurize([smiles])
    if not valid[0]:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    X = X.copy()
    X[:, FP_SIZE:] = (X[:, FP_SIZE:] - bundle["scaler_mean"]) / bundle["scaler_scale"]
    labels = bundle["labels"]

    if bundle["model_name"] == "tanimoto_knn":
        # similarity-weighted vote of the 20 nearest training molecules
        Xtr = bundle["knn_train_fp"]
        ytr = bundle["knn_train_y"]
        fp = X[:, :FP_SIZE]
        a = fp @ Xtr.T
        denom = fp.sum(1, keepdims=True) + Xtr.sum(1) - a
        sim = np.where(denom > 0, a / np.maximum(denom, 1e-9), 0.0)
        idx = np.argsort(sim[0])[::-1][:20]
        w = sim[0][idx]
        scores = (w @ ytr[idx]) / w.sum() if w.sum() > 0 else ytr.mean(0)
    else:
        model = bundle["model"]
        scores = np.array([est.predict_proba(X)[0, 1] for est in model.estimators_])
    return dict(zip(labels, scores))


def dataset_lookup(smiles):
    """If this molecule is in the Keller 2016 dataset, return (name, true labels)."""
    try:
        df = pd.read_csv("data/molecule_labels.csv")
    except FileNotFoundError:
        return None, None
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None, None
    canon = Chem.MolToSmiles(mol)
    hit = df[df["SMILES"] == canon]
    if hit.empty:
        return None, None
    row = hit.iloc[0]
    label_cols = [c for c in df.columns if c not in ("SMILES", "Name")]
    truth = [c for c in label_cols if row[c] == 1]
    return row["Name"], truth


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    smiles = sys.argv[1]
    bundle = load_bundle()
    try:
        scores = predict_scores(bundle, smiles)
    except ValueError as e:
        print(f"error: {e}")
        sys.exit(1)

    name, truth = dataset_lookup(smiles)
    print(f"SMILES: {smiles}")
    if name:
        print(f"Dataset molecule: {name}")
    print(f"Model: {bundle['model_name']}  (toy model - see README)")
    print(f"\nTop {TOP_N} predicted odor descriptors:")
    for label, p in sorted(scores.items(), key=lambda kv: -kv[1])[:TOP_N]:
        bar = "█" * int(round(p * BAR_WIDTH))
        print(f"  {label:16s} {p:0.2f} |{bar}")
    if truth is not None:
        print(f"\nGround-truth labels in the dataset (rate >= 15%): "
              f"{', '.join(truth) if truth else '(none)'}")


if __name__ == "__main__":
    main()
