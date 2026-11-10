"""
Predict the smell of a molecule with the v2 GNN (see train_gnn.py).

Usage:
    python predict_v2.py "COc1cc(C=O)ccc1O"          # vanillin
    python predict_v2.py "CC1=CCC(CC1)C(=C)C"        # limonene
    python predict_v2.py "C1=CC=C(C=C1)C=O"          # benzaldehyde

Prints the top predicted odor descriptors with probabilities as little bars,
plus the ground-truth labels from the merged v2 dataset when the molecule
is in it. Still a toy model - see the v2 section of README.md.
"""

import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

from graphs import mol_to_graph
from train_gnn import SmellGNN

TOP_N = 8
BAR_WIDTH = 30


def load_model():
    ckpt = torch.load("smell_gnn.pt", weights_only=False)
    model = SmellGNN(n_labels=len(ckpt["labels"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["labels"]


def predict_scores(model, labels, smiles):
    g = mol_to_graph(smiles)
    if g is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    with torch.no_grad():
        probs = torch.sigmoid(model(Batch.from_data_list([g]))).numpy()[0]
    return dict(zip(labels, probs))


def dataset_lookup(smiles):
    """Ground truth from the merged v2 dataset (only labels a source could
    express are listed - see prepare_data_v2.py)."""
    try:
        df = pd.read_csv("data/molecule_labels_v2.csv")
    except FileNotFoundError:
        return None, None, None
    from graphs import mol_to_graph  # noqa: F401  (keeps RDKit logging quiet)
    from rdkit import Chem
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None, None, None
    canon = Chem.MolToSmiles(mol, isomericSmiles=False)
    hit = df[df["SMILES"] == canon]
    if hit.empty:
        return None, None, None
    row = hit.iloc[0]
    label_cols = [c for c in df.columns
                  if c not in ("SMILES", "Name", "Sources", "Split")]
    truth = [c for c in label_cols if row[c] == 1]
    return row["Name"], truth, row["Sources"]


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    smiles = sys.argv[1]
    model, labels = load_model()
    try:
        scores = predict_scores(model, labels, smiles)
    except ValueError as e:
        print(f"error: {e}")
        sys.exit(1)

    name, truth, sources = dataset_lookup(smiles)
    print(f"SMILES: {smiles}")
    if name:
        print(f"Dataset molecule: {name} (sources: {sources})")
    print("Model: SmellGNN v2 (toy model - see README)")

    print(f"\nTop {TOP_N} predicted odor descriptors:")
    for label, p in sorted(scores.items(), key=lambda kv: -kv[1])[:TOP_N]:
        bar = "█" * int(round(p * BAR_WIDTH))
        print(f"  {label:16s} {p:0.2f} |{bar}")
    if truth is not None:
        print(f"\nGround-truth labels in the merged dataset: "
              f"{', '.join(truth) if truth else '(none)'}")


if __name__ == "__main__":
    main()
