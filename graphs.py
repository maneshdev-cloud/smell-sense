"""
Molecular graphs for the v2 GNN: SMILES -> PyTorch Geometric Data.

Each atom becomes a node with a 42-dim one-hot feature vector (element,
degree, formal charge, hybridisation, aromaticity, H count, ring membership,
chirality). Each bond becomes two directed edges with a 4-dim one-hot bond
type. This is the standard "what is this atom and how is it connected"
description a message-passing network needs - the graph analogue of v1's
fingerprints.
"""

import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType, ChiralType, HybridizationType
from rdkit import RDLogger
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")

ELEMENTS = ["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"]
HYBRIDISATIONS = [HybridizationType.SP, HybridizationType.SP2,
                  HybridizationType.SP3, HybridizationType.SP3D,
                  HybridizationType.SP3D2, HybridizationType.S,
                  HybridizationType.UNSPECIFIED]
CHIRALITIES = [ChiralType.CHI_UNSPECIFIED, ChiralType.CHI_TETRAHEDRAL_CW,
               ChiralType.CHI_TETRAHEDRAL_CCW, ChiralType.CHI_OTHER]
BOND_TYPES = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE,
              BondType.AROMATIC]

N_ATOM_FEATURES = (len(ELEMENTS) + 1 + 6 + 5 + len(HYBRIDISATIONS)
                   + 1 + 5 + 1 + len(CHIRALITIES))  # = 42
N_BOND_FEATURES = len(BOND_TYPES)


def _one_hot(idx, size):
    v = [0.0] * size
    v[idx] = 1.0
    return v


def atom_features(atom):
    feats = []
    symbol = atom.GetSymbol()
    feats += _one_hot(ELEMENTS.index(symbol) if symbol in ELEMENTS
                      else len(ELEMENTS), len(ELEMENTS) + 1)      # + "other"
    feats += _one_hot(min(atom.GetDegree(), 5), 6)
    feats += _one_hot(int(atom.GetFormalCharge()) + 2, 5)         # charges -2..2
    feats += _one_hot(HYBRIDISATIONS.index(atom.GetHybridization()),
                      len(HYBRIDISATIONS))
    feats += [1.0 if atom.GetIsAromatic() else 0.0]
    feats += _one_hot(min(atom.GetTotalNumHs(), 4), 5)
    feats += [1.0 if atom.IsInRing() else 0.0]
    feats += _one_hot(CHIRALITIES.index(atom.GetChiralTag()), len(CHIRALITIES))
    assert len(feats) == N_ATOM_FEATURES
    return feats


def mol_to_graph(smiles):
    """SMILES -> PyG Data with x (atoms), edge_index, edge_attr; None if the
    SMILES does not parse."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()],
                     dtype=torch.float)
    src, dst, attr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = _one_hot(BOND_TYPES.index(bond.GetBondType()), N_BOND_FEATURES)
        for a, b in [(i, j), (j, i)]:          # undirected bond, two messages
            src.append(a)
            dst.append(b)
            attr.append(bt)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(attr, dtype=torch.float).reshape(-1, N_BOND_FEATURES)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
