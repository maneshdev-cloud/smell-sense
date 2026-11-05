"""
Prepare the molecule -> odor-descriptor dataset from the raw Keller & Vosshall
(2016) files hosted in the pyrfume-data GitHub repository.

Raw encoding notes (verified by inspecting the files, see README):
  * data/behavior.csv is long-format: one row per stimulus x subject x question.
  * For the 20 odor-descriptor questions, a subject who *applied* the descriptor
    to a stimulus has a (numeric) value in `Value`; everyone else is NaN.
    So "descriptor applied" == Value is not NaN. The numeric content is an
    artefact of the original spreadsheet and is ignored.
  * data/stimuli.csv maps each stimulus to exactly one molecule (CID) - there
    are no mixtures. Each molecule was tested at 1-2 dilutions, sometimes with
    replicate trials.

Aggregation: for each molecule x descriptor, pool every subject rating of
every stimulus of that molecule and take the fraction of ratings in which the
descriptor was applied -> a rate in [0, 1]. Binarise at RATE_THRESHOLD.
Molecules sharing the same RDKit-canonical SMILES are merged (rates averaged).
Output: data/molecule_labels.csv with columns SMILES, Name + one 0/1 column
per descriptor, plus the raw rates in data/molecule_rates.csv.
"""

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # keep the console readable

DESCRIPTORS = [
    "EDIBLE", "BAKERY", "SWEET", "FRUIT", "FISH", "GARLIC", "SPICES", "COLD",
    "SOUR", "BURNT", "ACID", "WARM", "MUSKY", "SWEATY", "AMMONIA/URINOUS",
    "DECAYED", "WOOD", "GRASS", "FLOWER", "CHEMICAL",
]
RATE_THRESHOLD = 0.15  # descriptor counts as "present" if >=15% of ratings applied it


def main():
    behavior = pd.read_csv(
        "data/behavior.csv", usecols=["Stimulus", "MeasurementValue", "Value"]
    )
    stimuli = pd.read_csv("data/stimuli.csv")
    molecules = pd.read_csv("data/molecules.csv")

    # --- descriptor application rates per stimulus -------------------------
    desc = behavior[behavior["MeasurementValue"].isin(DESCRIPTORS)].copy()
    desc["applied"] = desc["Value"].notna().astype(int)
    stim_rates = (
        desc.groupby(["Stimulus", "MeasurementValue"])["applied"].mean().unstack()
    )
    print(f"stimuli with descriptor ratings: {len(stim_rates)}")

    # --- stimulus -> molecule (CID) ----------------------------------------
    # Quirk of the source files: 4 stimuli rows carry a CAS number instead of
    # a CID in the CIDs column; resolve those through the molecules table.
    cas2cid = molecules.dropna(subset=["CAS"]).set_index("CAS")["CID"].astype(int)

    def to_cid(x):
        x = str(x)
        return int(x) if x.isdigit() else int(cas2cid[x])

    stim2cid = stimuli.set_index("Stimulus")["CIDs"].map(to_cid)
    stim_rates["CID"] = stim_rates.index.map(stim2cid).astype(int)

    # --- pool stimuli of the same molecule ----------------------------------
    mol_rates = stim_rates.groupby("CID")[DESCRIPTORS].mean()
    print(f"molecules with ratings: {len(mol_rates)}")

    # --- attach names + SMILES, canonicalise with RDKit, merge duplicates ---
    molecules["CID"] = molecules["CID"].astype(int)
    info = (
        molecules.dropna(subset=["CanonicalSMILES"])
        .drop_duplicates("CID")
        .set_index("CID")
    )
    mol_rates = mol_rates.join(info[["OdorName", "CanonicalSMILES"]], how="inner")
    print(f"molecules with both ratings and SMILES: {len(mol_rates)}")

    def canon(smi):
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol is not None else None

    mol_rates["SMILES"] = mol_rates["CanonicalSMILES"].map(canon)
    n_bad = mol_rates["SMILES"].isna().sum()
    mol_rates = mol_rates.dropna(subset=["SMILES"])
    print(f"dropped {n_bad} molecules whose SMILES RDKit could not parse")

    merged = (
        mol_rates.groupby("SMILES")
        .agg({**{d: "mean" for d in DESCRIPTORS}, "OdorName": "first"})
        .reset_index()
    )
    print(f"unique molecules after merging duplicate SMILES: {len(merged)}")

    # --- binarise -------------------------------------------------------------
    labels = (merged[DESCRIPTORS] >= RATE_THRESHOLD).astype(int)
    labels.columns = [d.lower().replace("/", "_") for d in DESCRIPTORS]
    out = pd.concat([merged[["SMILES", "OdorName"]].rename(columns={"OdorName": "Name"}),
                     labels], axis=1)
    rates_out = merged[["SMILES", "OdorName"] + DESCRIPTORS].rename(
        columns={"OdorName": "Name"})
    out.to_csv("data/molecule_labels.csv", index=False)
    rates_out.to_csv("data/molecule_rates.csv", index=False)

    print(f"\nbinarisation threshold: rate >= {RATE_THRESHOLD}")
    print("positive examples per label:")
    print(labels.sum().sort_values(ascending=False).to_string())
    print(f"\nsaved data/molecule_labels.csv ({len(out)} rows) and data/molecule_rates.csv")


if __name__ == "__main__":
    main()
