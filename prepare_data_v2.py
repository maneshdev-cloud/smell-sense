"""
Prepare the v2 dataset: Keller 2016 (v1) merged with Dravnieks 1985 and
Leffingwell (PMP 2001), all from the official pyrfume-data GitHub repo.

Why these two extra sources, and how the merge works
----------------------------------------------------
* Dravnieks 1985 ("Atlas of Odor Character Profiles"): a trained panel rated
  ~140 odorants on 146 descriptors. behaviour_1.csv is the *applicability*
  measure - the percentage of panelists applying each descriptor - which is
  the same quantity Keller pooled, so we binarise at the same >= 15% rule.
  Its 146 words are mapped onto Keller's 20 with the family table below
  (a family's score is the MAX of its member words: if any specific word
  for "fruit" applies, the generic Keller word "fruit" applies).
  Only Keller's `edible` has no Dravnieks counterpart -> unobserved there.
* Leffingwell (from "Perfume and Flavor Chemicals", PMP 2001, as processed
  by Google/pyrfume): ~3,500 molecules with binary catalogue annotations
  over 113 words - expert catalogue tags, NOT panel ratings. A listed word
  counts as positive; an unlisted word counts as negative *within the
  mapped families only* (that is how this dataset is conventionally used,
  e.g. by the POM authors, but it is weaker evidence than panel data -
  see the README). 16 of Keller's 20 labels have a Leffingwell family;
  `edible`, `cold`, `acid`, `chemical` do not -> unobserved there.
* Keller 2012 was considered and excluded: its descriptor table covers
  only 3 odorants (the rest is intensity/pleasantness), so it cannot
  enlarge a descriptor dataset.

Labels a source cannot express are NOT silently set to 0. The outputs keep
a per-molecule, per-label "observed" mask; v2 training uses masked losses /
per-label training subsets, and a molecule's label is the union (OR) of the
sources that observed it.

Split: the 474 Keller molecules keep their exact v1 assignment (seed-42
permutation of data/molecule_labels.csv). Every molecule that only exists
in the new sources is assigned to TRAIN, so validation/test stay byte-for-
byte v1's - the v1-vs-v2 comparison is on identical evaluation molecules.

Outputs:
  data/molecule_labels_v2.csv  SMILES, Name, Sources, Split + 20 label
                               columns (0/1 where observed, empty where not)
  data/molecule_mask_v2.csv    SMILES + 20 columns of 1/0 "label observed"
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

LABELS20 = [
    "edible", "bakery", "sweet", "fruit", "fish", "garlic", "spices", "cold",
    "sour", "burnt", "acid", "warm", "musky", "sweaty", "ammonia_urinous",
    "decayed", "wood", "grass", "flower", "chemical",
]
APPLICABILITY_THRESHOLD = 15.0  # percent, mirroring Keller's >=15% rule

# --------------------------------------------------------------------------- #
# descriptor family maps (target Keller word -> source words; family = max)   #
# --------------------------------------------------------------------------- #
DRAVNIEKS_MAP = {
    "bakery": ["BAKERY (FRESH BREAD)"],
    "sweet": ["SWEET"],
    "fruit": ["FRUITY,CITRUS", "FRUITY,OTHER THAN CITRUS", "LEMON",
              "GRAPEFRUIT", "ORANGE", "PINEAPPLE", "GRAPE JUICE", "STRAWBERRY",
              "APPLE, FRUIT", "PEAR", "CANTALOUPE, HONEYDEW MELON",
              "PEACH FRUIT", "BANANA", "CHERRY, BERRY"],
    "fish": ["FISHY", "KIPPERY,SMOKED FISH"],
    "garlic": ["GARLIC, ONION"],
    "spices": ["SPICY", "CLOVE", "CINNAMON", "BLACK PEPPER",
               "SEASONING FOR MEAT", "LAUREL LEAVES", "ANISE, LICORICE",
               "CARAWAY", "DILL"],
    "cold": ["COOL,COOLING"],
    "sour": ["SOUR, VINEGAR"],
    "burnt": ["BURNT,SMOKY", "BURNT PAPER", "BURNT MILK", "BURNT RUBBER",
              "BURNT CANDLE"],
    "acid": ["SHARP, PUNGENT, ACID"],
    "warm": ["WARM"],
    "musky": ["MUSK"],
    "sweaty": ["SWEATY"],
    "ammonia_urinous": ["AMMONIA", "URINE", "CAT URINE"],
    "decayed": ["PUTRID, FOUL, DECAYED", "FECAL, LIKE MANURE", "CADAVEROUS",
                "RANCID", "FERMENTED, ROTTEN FRUIT"],
    "wood": ["WOODY, RESINOUS", "OAK WOOD,COGNAC", "CEDARWOOD",
             "BARK,BIRCHBARK", "CORK"],
    "grass": ["CRUSHED GRASS", "HERBAL, GREEN,CUTGRASS", "CRUSHED WEEDS"],
    "flower": ["FLORAL", "ROSE", "VIOLETS", "LAVENDER"],
    "chemical": ["CHEMICAL"],
    # "edible": no Dravnieks counterpart -> unobserved
}

LEFFINGWELL_MAP = {
    "bakery": ["bread", "malty", "popcorn"],
    "sweet": ["sweet"],
    "fruit": ["fruity", "apple", "apricot", "banana", "berry", "black currant",
              "cherry", "citrus", "grape", "grapefruit", "lemon", "melon",
              "orange", "peach", "pear", "pineapple", "plum", "strawberry",
              "tropical"],
    "fish": ["fishy"],
    "garlic": ["garlic", "alliaceous", "onion"],
    "spices": ["spicy", "cinnamon", "anisic"],
    "sour": ["sour"],
    "burnt": ["burnt", "smoky", "roasted"],
    "warm": ["warm"],
    "musky": ["musk"],
    "sweaty": ["cheesy"],            # proxy: cheesy = butyric, sweaty-feet note
    "ammonia_urinous": ["catty"],    # proxy: catty = the cat-urine note
    "decayed": ["fermented", "musty"],
    "wood": ["woody", "pine"],
    "grass": ["grassy", "green", "leafy", "hay"],
    "flower": ["floral", "jasmine", "rose", "violet", "chamomile", "orris"],
    # "edible", "cold", "acid", "chemical": no Leffingwell counterpart
}


def canon_key(smiles):
    """Canonical merge key: RDKit canonical SMILES with stereochemistry
    stripped (Keller's SMILES carry no stereo; v1 treated it as invisible)."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def empty_record():
    return {"labels": {}, "observed": set(), "sources": [], "name": None}


def add_molecule(store, key, name, labels01, observed, source):
    rec = store.setdefault(key, empty_record())
    if rec["name"] is None and name:
        rec["name"] = name
    if source not in rec["sources"]:
        rec["sources"].append(source)
    for lab in observed:
        rec["observed"].add(lab)
        if labels01.get(lab, 0):
            rec["labels"][lab] = 1
        else:
            rec["labels"].setdefault(lab, 0)


# --------------------------------------------------------------------------- #
# source loaders: each returns {canon_key: (name, {label: 0/1}, observed)}    #
# --------------------------------------------------------------------------- #
def load_keller():
    df = pd.read_csv("data/molecule_labels.csv")
    out = {}
    for _, row in df.iterrows():
        key = canon_key(row["SMILES"])
        labels = {lab: int(row[lab]) for lab in LABELS20}
        out[key] = (row["Name"], labels, set(LABELS20))
    return out


def load_dravnieks():
    beh = pd.read_csv("data_v2/dravnieks_1985/behavior_1.csv", index_col=0)
    stim = pd.read_csv("data_v2/dravnieks_1985/stimuli.csv", index_col=0)
    mols = pd.read_csv("data_v2/dravnieks_1985/molecules.csv").set_index("CID")
    # 16 of the 160 stimuli have no CID in stimuli.csv (and hence no SMILES
    # anywhere in the source files) - they cannot enter a structure dataset.
    beh["CID"] = beh.index.map(stim["CID"])
    n_no_cid = int(beh["CID"].isna().sum())
    beh = beh.dropna(subset=["CID"])
    beh["CID"] = beh["CID"].astype(int)
    print(f"  (dropped {n_no_cid} stimuli with no CID/SMILES in the source)")
    # pool a molecule's stimuli (high/low concentration) like Keller pooled
    # its dilutions: average applicability, then threshold
    pooled = beh.groupby("CID")[list(beh.columns[:-1])].mean()
    out = {}
    for cid, row in pooled.iterrows():
        if cid not in mols.index:
            continue
        key = canon_key(mols.loc[cid, "IsomericSMILES"])
        if key is None:
            continue
        labels = {}
        for lab, words in DRAVNIEKS_MAP.items():
            labels[lab] = int(row[words].max() >= APPLICABILITY_THRESHOLD)
        out[key] = (mols.loc[cid, "name"], labels, set(DRAVNIEKS_MAP))
    return out


def load_leffingwell():
    beh = pd.read_csv("data_v2/leffingwell/behavior.csv", index_col=0)
    stim = pd.read_csv("data_v2/leffingwell/stimuli.csv", index_col=0)
    smiles = stim["IsomericSMILES"]
    names = stim["name"]
    out = {}
    for stim_id, row in beh.iterrows():
        if stim_id not in smiles.index:
            continue
        key = canon_key(smiles.loc[stim_id])
        if key is None:
            continue
        labels = {}
        for lab, words in LEFFINGWELL_MAP.items():
            labels[lab] = int(row[words].max() >= 1)
        name = names.loc[stim_id]
        # a few SMILES repeat under different stimulus ids: union their tags
        if key in out:
            prev_name, prev_labels, prev_obs = out[key]
            labels = {lab: max(labels[lab], prev_labels[lab]) for lab in labels}
            name = prev_name or name
        out[key] = (name, labels, set(LEFFINGWELL_MAP))
    return out


def v1_split_by_key():
    """Recover v1's exact split (train.py: RandomState(42) permutation of the
    rows of data/molecule_labels.csv, 20% test / 16% val / 64% train)."""
    df = pd.read_csv("data/molecule_labels.csv")
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(df))
    n_test = int(round(0.20 * len(df)))
    n_val = int(round(0.16 * len(df)))
    split = {}
    for name, idxs in [("test", perm[:n_test]),
                       ("val", perm[n_test:n_test + n_val]),
                       ("train", perm[n_test + n_val:])]:
        for i in idxs:
            split[canon_key(df["SMILES"][i])] = name
    return split


def agreement(a, b, shared_labels):
    """Descriptive cross-source check on molecules present in both sources:
    overall agreement and recall of a's positives by b, per shared label set."""
    keys = set(a) & set(b)
    tot = agree = pos_a = pos_caught = 0
    for k in keys:
        for lab in shared_labels:
            va, vb = a[k][1][lab], b[k][1][lab]
            tot += 1
            agree += int(va == vb)
            pos_a += va
            pos_caught += int(va == 1 and vb == 1)
    return {"n_molecules": len(keys), "overall_agreement": agree / max(tot, 1),
            "positive_recall": pos_caught / max(pos_a, 1)}


def main():
    keller = load_keller()
    drav = load_dravnieks()
    leff = load_leffingwell()
    print(f"Keller 2016:    {len(keller)} molecules, all {len(LABELS20)} labels observed")
    print(f"Dravnieks 1985: {len(drav)} molecules, {len(DRAVNIEKS_MAP)} labels observed")
    print(f"Leffingwell:    {len(leff)} molecules, {len(LEFFINGWELL_MAP)} labels observed")

    print("\noverlap (shared canonical SMILES):")
    print(f"  Keller & Dravnieks:  {len(set(keller) & set(drav))}")
    print(f"  Keller & Leffingwell:{len(set(keller) & set(leff))}")
    print(f"  Dravnieks & Leffingwell: {len(set(drav) & set(leff))}")

    print("\ncross-source agreement on shared molecules (descriptive):")
    print("  Keller vs Dravnieks:",
          agreement(keller, drav, sorted(DRAVNIEKS_MAP)))
    print("  Keller vs Leffingwell:",
          agreement(keller, leff, sorted(LEFFINGWELL_MAP)))

    store = {}
    for source, loaded in [("keller_2016", keller), ("dravnieks_1985", drav),
                           ("leffingwell", leff)]:
        for key, (name, labels, observed) in loaded.items():
            add_molecule(store, key, name, labels, observed, source)

    split = v1_split_by_key()
    rows, mask_rows = [], []
    for key, rec in store.items():
        row = {"SMILES": key, "Name": rec["name"],
               "Sources": ";".join(rec["sources"]),
               "Split": split.get(key, "train")}  # new molecules: train only
        mask = {"SMILES": key}
        for lab in LABELS20:
            obs = lab in rec["observed"]
            row[lab] = rec["labels"].get(lab, 0) if obs else np.nan
            mask[lab] = int(obs)
        rows.append(row)
        mask_rows.append(mask)

    out = pd.DataFrame(rows).sort_values(["Split", "SMILES"]).reset_index(drop=True)
    msk = pd.DataFrame(mask_rows).set_index("SMILES").loc[out["SMILES"]].reset_index()
    out.to_csv("data/molecule_labels_v2.csv", index=False)
    msk.to_csv("data/molecule_mask_v2.csv", index=False)

    print(f"\nmerged dataset: {len(out)} unique molecules")
    print(out["Split"].value_counts().to_string())
    print("\nlabel coverage (molecules observing the label / positives):")
    for lab in LABELS20:
        obs = int(msk[lab].sum())
        pos = int(out.loc[msk[lab] == 1, lab].sum())
        print(f"  {lab:16s} observed {obs:5d}  positives {pos:5d}")
    print("\nsaved data/molecule_labels_v2.csv and data/molecule_mask_v2.csv")


if __name__ == "__main__":
    main()
