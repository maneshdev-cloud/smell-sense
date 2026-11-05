"""
Shared featurisation: SMILES -> [Morgan fingerprint bits | RDKit descriptors].

Used by both train.py (fitting) and predict.py (inference) so the feature
layout can never drift between the two.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors, Crippen
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

FP_RADIUS = 2
FP_SIZE = 2048
_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)

# The 8 simple physicochemical descriptors appended after the fingerprint.
DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "NumHDonors", "NumHAcceptors",
    "TPSA", "NumRotatableBonds", "NumAromaticRings", "FractionCSP3",
]
N_FEATURES = FP_SIZE + len(DESCRIPTOR_NAMES)


def mol_from_smiles(smiles):
    """Parse a SMILES string; return an RDKit Mol or None if unparseable."""
    if smiles is None:
        return None
    return Chem.MolFromSmiles(str(smiles))


def descriptor_vector(mol):
    """The 8 descriptors for one molecule, as a float array."""
    return np.array([
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
    ], dtype=np.float64)


def fingerprint_array(mol):
    """2048-bit Morgan fingerprint (radius 2) as a float array of 0/1."""
    return np.array(_morgan.GetFingerprint(mol), dtype=np.float64)


def featurize(smiles_list):
    """
    Return (X, valid_mask): X has one row per parseable molecule -
    2048 fingerprint bits followed by the 8 raw descriptors (unscaled;
    scaling is the caller's job so train/predict share one scaler).
    """
    rows, valid = [], []
    for smi in smiles_list:
        mol = mol_from_smiles(smi)
        if mol is None:
            valid.append(False)
            continue
        valid.append(True)
        rows.append(np.concatenate([fingerprint_array(mol), descriptor_vector(mol)]))
    X = np.vstack(rows) if rows else np.zeros((0, N_FEATURES))
    return X, np.array(valid)
