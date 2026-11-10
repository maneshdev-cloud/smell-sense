"""
SmellSense v2: a graph neural network on the enlarged (merged) dataset,
compared fairly against v1's random forest.

Design (all choices fixed before looking at test):
  * Data: data/molecule_labels_v2.csv (Keller 2016 + Dravnieks 1985 +
    Leffingwell, see prepare_data_v2.py). Labels = v1's 18 descriptors.
  * Split: validation (76) and test (95) are EXACTLY v1's Keller molecules;
    train is v1's 303 Keller train molecules + every new-source molecule.
    So v1 RF and v2 models are scored on identical evaluation molecules.
  * Partial labels: some sources cannot express some labels (mask file).
    The GNN trains with a masked BCE loss; the RF control trains one forest
    per label on only the molecules that observe that label.
  * Selection: the GNN checkpoint with the best VALIDATION macro-AUROC is
    kept; test is evaluated once, afterwards, for every model.
  * RF control: v1's winner recipe (400-tree random forest on Morgan
    fingerprints + 8 descriptors) retrained on the enlarged train pool, so
    "more data" and "new architecture" effects can be told apart.

Outputs: smell_gnn.pt (best GNN checkpoint), results_v2.json.
"""

import copy
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool

from features import FP_SIZE, featurize
from graphs import N_ATOM_FEATURES, N_BOND_FEATURES, mol_to_graph

SEED = 42
LABELS18 = json.load(open("results.json"))["dataset"]["labels_used"]
HIDDEN = 128
EPOCHS = 400
PATIENCE = 60          # early stop after this many epochs without val gain
BATCH = 64
LR = 1e-3
SHOWCASE = {
    "vanillin": "COc1cc(C=O)ccc1O",
    "limonene": "CC1=CCC(CC1)C(=C)C",
    "benzaldehyde": "C1=CC=C(C=C1)C=O",
}


def macro_auroc(y_true, scores):
    """Mean per-label AUROC (all 18 labels are fully observed on val/test,
    which are pure Keller molecules). Returns (macro, {label: auroc})."""
    per = {}
    for j, lab in enumerate(LABELS18):
        per[lab] = float(roc_auc_score(y_true[:, j], scores[:, j]))
    return float(np.mean(list(per.values()))), per


# --------------------------------------------------------------------------- #
# the GNN: 3 x GINE message-passing layers, mean+max pooling, MLP head        #
# --------------------------------------------------------------------------- #
class SmellGNN(torch.nn.Module):
    def __init__(self, n_labels=len(LABELS18)):
        super().__init__()
        dims = [N_ATOM_FEATURES, HIDDEN, HIDDEN, HIDDEN]
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for i in range(3):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(dims[i], HIDDEN), torch.nn.ReLU(),
                torch.nn.Linear(HIDDEN, HIDDEN))
            self.convs.append(GINEConv(mlp, edge_dim=N_BOND_FEATURES))
            self.norms.append(torch.nn.BatchNorm1d(HIDDEN))
        self.head = torch.nn.Sequential(
            torch.nn.Linear(2 * HIDDEN, HIDDEN), torch.nn.ReLU(),
            torch.nn.Dropout(0.3), torch.nn.Linear(HIDDEN, n_labels))

    def forward(self, data):
        h = data.x
        for conv, norm in zip(self.convs, self.norms):
            h = F.relu(norm(conv(h, data.edge_index, data.edge_attr)))
            h = F.dropout(h, p=0.2, training=self.training)
        pooled = torch.cat([global_mean_pool(h, data.batch),
                            global_max_pool(h, data.batch)], dim=1)
        return self.head(pooled)


def masked_bce(logits, y, mask, pos_weight):
    """BCE averaged per label over the molecules that observe that label."""
    loss_mat = F.binary_cross_entropy_with_logits(
        logits, y, pos_weight=pos_weight, reduction="none")
    per_label = (loss_mat * mask).sum(0) / mask.sum(0).clamp(min=1)
    return per_label.mean()


@torch.no_grad()
def gnn_scores(model, loader):
    model.eval()
    out = []
    for batch in loader:
        out.append(torch.sigmoid(model(batch)).numpy())
    return np.vstack(out)


def train_gnn(graphs, y, mask, idx):
    """Fit the GNN on train idx; return (best_state, best_val_macro, epoch)."""
    tr = [graphs[i] for i in idx["train"]]
    for g, yi, mi in zip(tr, y[idx["train"]], mask[idx["train"]]):
        g.y = torch.tensor(yi, dtype=torch.float)
        g.m = torch.tensor(mi, dtype=torch.float)
    train_loader = DataLoader(tr, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader([graphs[i] for i in idx["val"]], batch_size=BATCH)
    test_loader = DataLoader([graphs[i] for i in idx["test"]], batch_size=BATCH)

    ytr, mtr = y[idx["train"]], mask[idx["train"]]
    pos = (ytr * mtr).sum(0)
    neg = ((1 - ytr) * mtr).sum(0)
    pos_weight = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1.0, 20.0),
                              dtype=torch.float)

    torch.manual_seed(SEED)
    model = SmellGNN()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val, best_state, best_epoch, bad = -1.0, None, -1, 0
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            # PyG concatenates per-graph attributes flat; restore (batch, 18)
            y_b = batch.y.view(-1, len(LABELS18))
            m_b = batch.m.view(-1, len(LABELS18))
            loss = masked_bce(model(batch), y_b, m_b, pos_weight)
            loss.backward()
            opt.step()
        val_macro, _ = macro_auroc(y[idx["val"]], gnn_scores(model, val_loader))
        if val_macro > best_val:
            best_val, best_epoch = val_macro, epoch
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"  early stop at epoch {epoch} "
                      f"(best val {best_val:.4f} @ epoch {best_epoch})")
                break
    model.load_state_dict(best_state)
    return model, best_val, best_epoch, val_loader, test_loader


def rf_control(df, y, mask, idx):
    """v1's RF recipe, one forest per label, each trained on the train-pool
    molecules that observe that label. Returns (val_scores, test_scores)."""
    X, valid = featurize(df["SMILES"])
    assert valid.all()
    scaler = StandardScaler().fit(X[idx["train"], FP_SIZE:])
    Xs = X.copy()
    Xs[:, FP_SIZE:] = scaler.transform(X[:, FP_SIZE:])
    scores = {}
    for split in ("val", "test"):
        scores[split] = np.zeros((len(idx[split]), len(LABELS18)))
    for j, lab in enumerate(LABELS18):
        obs = mask[idx["train"], j] == 1
        rf = RandomForestClassifier(n_estimators=400, n_jobs=-1,
                                    random_state=SEED)
        rf.fit(Xs[idx["train"]][obs], y[idx["train"]][obs, j])
        for split in ("val", "test"):
            scores[split][:, j] = rf.predict_proba(Xs[idx[split]])[:, 1]
        print(f"  RF control: label {lab:16s} trained on {int(obs.sum())} molecules")
    return scores["val"], scores["test"]


def main():
    t0 = time.time()
    df = pd.read_csv("data/molecule_labels_v2.csv")
    msk = pd.read_csv("data/molecule_mask_v2.csv").set_index("SMILES").loc[df["SMILES"]]
    y = df[LABELS18].fillna(0).to_numpy(dtype=np.float32)
    mask = msk[LABELS18].to_numpy(dtype=np.float32)
    idx = {name: np.where(df["Split"] == name)[0]
           for name in ("train", "val", "test")}
    print(f"dataset: {len(df)} molecules | train {len(idx['train'])} "
          f"| val {len(idx['val'])} | test {len(idx['test'])}")
    print(f"labels observed in train: "
          f"{ {lab: int(mask[idx['train'], j].sum()) for j, lab in enumerate(LABELS18) if mask[idx['train'], j].sum() < len(idx['train'])} } "
          f"(others: all train molecules)")

    print("\nbuilding molecular graphs...")
    graphs = [mol_to_graph(s) for s in df["SMILES"]]
    assert all(g is not None for g in graphs)

    print("\ntraining GNN (masked BCE, checkpoint selected on validation)...")
    model, gnn_val, best_epoch, val_loader, test_loader = train_gnn(
        graphs, y, mask, idx)
    gnn_val_scores = gnn_scores(model, val_loader)
    gnn_test_scores = gnn_scores(model, test_loader)
    gnn_val_macro, _ = macro_auroc(y[idx["val"]], gnn_val_scores)
    gnn_test_macro, gnn_test_per = macro_auroc(y[idx["test"]], gnn_test_scores)
    print(f"  GNN val macro-AUROC {gnn_val_macro:.4f} | test {gnn_test_macro:.4f}")

    print("\ntraining RF control on the enlarged pool (per-label forests)...")
    rf_val_scores, rf_test_scores = rf_control(df, y, mask, idx)
    rf_val_macro, _ = macro_auroc(y[idx["val"]], rf_val_scores)
    rf_test_macro, rf_test_per = macro_auroc(y[idx["test"]], rf_test_scores)
    print(f"  RF control val {rf_val_macro:.4f} | test {rf_test_macro:.4f}")

    torch.save({"state_dict": model.state_dict(), "labels": LABELS18,
                "hidden": HIDDEN, "best_epoch": best_epoch,
                "val_macro_auroc": gnn_val_macro}, "smell_gnn.pt")
    size_mb = __import__("os").path.getsize("smell_gnn.pt") / 1e6
    print(f"\nsaved smell_gnn.pt ({size_mb:.2f} MB), best epoch {best_epoch}")

    # showcase predictions with the selected checkpoint
    showcase = {}
    for name, smi in SHOWCASE.items():
        g = mol_to_graph(smi)
        from torch_geometric.data import Batch
        with torch.no_grad():
            probs = torch.sigmoid(model(Batch.from_data_list([g]))).numpy()[0]
        top = sorted(zip(LABELS18, probs), key=lambda kv: -kv[1])[:8]
        showcase[name] = {lab: round(float(p), 2) for lab, p in top}
        print(f"  {name}: " + " | ".join(f"{lab} {p:.2f}" for lab, p in top))

    v1 = json.load(open("results.json"))
    results = {
        "dataset_v2": {
            "sources": ["keller_2016 (panel ratings, pyrfume-data)",
                        "dravnieks_1985 (panel applicability, pyrfume-data)",
                        "leffingwell (catalogue annotations, pyrfume-data)"],
            "n_molecules": int(len(df)),
            "n_train": int(len(idx["train"])),
            "split_note": "val/test are exactly v1's 76/95 Keller molecules; "
                          "every non-Keller molecule is in train",
            "labels": LABELS18,
            "label_observed_counts_train": {
                lab: int(mask[idx["train"], j].sum())
                for j, lab in enumerate(LABELS18)},
            "label_positive_counts_train": {
                lab: int((y[idx["train"], j] * mask[idx["train"], j]).sum())
                for j, lab in enumerate(LABELS18)},
        },
        "gnn": {
            "architecture": "3x GINEConv (hidden 128) + BatchNorm + ReLU, "
                            "mean+max global pooling, MLP head; masked BCE "
                            "with per-label pos_weight; Adam lr 1e-3",
            "best_epoch": int(best_epoch),
            "checkpoint": "smell_gnn.pt",
            "checkpoint_size_mb": round(size_mb, 3),
        },
        "val_macro_auroc": {
            "random_forest_v1": v1["validation_macro_auroc"]["random_forest"],
            "random_forest_v2_control": rf_val_macro,
            "gnn_v2": gnn_val_macro,
        },
        "test_macro_auroc": {
            "random_forest_v1": v1["test_macro_auroc"]["random_forest"],
            "random_forest_v2_control": rf_test_macro,
            "gnn_v2": gnn_test_macro,
        },
        "test_per_label_auroc": {
            "random_forest_v1": v1["test_per_label_auroc"]["random_forest"],
            "random_forest_v2_control": rf_test_per,
            "gnn_v2": gnn_test_per,
        },
        "showcase_predictions_gnn": showcase,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open("results_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved results_v2.json")
    print(f"\nSUMMARY  val: RFv1 {results['val_macro_auroc']['random_forest_v1']:.3f} "
          f"| RFv2 {rf_val_macro:.3f} | GNN {gnn_val_macro:.3f}")
    print(f"SUMMARY test: RFv1 {results['test_macro_auroc']['random_forest_v1']:.3f} "
          f"| RFv2 {rf_test_macro:.3f} | GNN {gnn_test_macro:.3f}")


if __name__ == "__main__":
    main()
