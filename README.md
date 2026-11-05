# Smell Sense — a toy "AI that predicts smell"

A small, honest reproduction of the *idea* behind Osmo's Principal Odor Map:
given a molecule's structure, predict how it smells to humans — as a
multi-label classification over odor descriptors ("sweet", "musky", "woody", …).

Built with open data and classic ML only: RDKit fingerprints + logistic
regression / random forest / a small MLP. It is **not** Osmo-scale — see
*Limitations* — but every number below is real, from actual runs of the code
in this folder.

## Quick start

```bash
python3 -m venv venv
./venv/bin/pip install rdkit scikit-learn pandas numpy joblib
./venv/bin/python predict.py "COc1cc(C=O)ccc1O"     # vanillin
./venv/bin/python predict.py "CC1=CCC(CC1)C(=C)C"   # limonene
./venv/bin/python predict.py "C1=CC=C(C=C1)C=O"     # benzaldehyde
```

To rebuild everything from raw data:

```bash
./venv/bin/python prepare_data.py   # raw CSVs -> data/molecule_labels.csv
./venv/bin/python train.py          # trains, selects on validation, writes results.json + smell_model.joblib
./venv/bin/python similarity_check.py
```

## Dataset

**Keller & Vosshall (2016), "Olfactory perception of chemically diverse
molecules"** (the DREAM Olfaction Prediction Challenge data),
DOI [10.1186/s12868-016-0287-2](https://doi.org/10.1186/s12868-016-0287-2).

* Source: the official **pyrfume-data** GitHub repository,
  https://github.com/pyrfume/pyrfume-data/tree/main/keller_2016
  (files `molecules.csv`, `stimuli.csv`, `behavior.csv`, downloaded raw —
  see the pyrfume note under Limitations).
* 55 human subjects each rated 480 molecules (960 stimuli: dilutions and
  replicates) on 20 odor descriptors, plus intensity/pleasantness/familiarity.
  `behavior.csv` holds 1,430,000 rating rows.
* In this file's long format, a descriptor counts as *applied* by a subject
  when the `Value` cell is non-empty (verified by inspection; the numeric
  content is an artefact of the original spreadsheet and is ignored).
* Aggregation (see `prepare_data.py`): for each molecule × descriptor, pool
  all subject ratings of all its stimuli and take the fraction applying the
  descriptor. Binarise at **≥ 15%** application rate.
* 478 molecules had both ratings and a SMILES; after RDKit canonicalisation
  and merging molecules that share an identical SMILES: **474 unique
  molecules**.
* Labels kept: **18 of 20** descriptors with ≥ 30 positive examples.
  Dropped for too few positives: `fish` (17), `grass` (23).
  Positive counts range from `chemical` (395 of 474) down to `wood` (30).
* Source-file quirks handled: 4 stimulus rows key a molecule by CAS number
  instead of CID (isobutyl acetate; 6,10-dimethyl-5,9-undecadien-2-one) —
  resolved through the molecules table; 2 molecules have no SMILES and were
  dropped; 1 CID appears twice in `molecules.csv`.

## Method

* **Features** (`features.py`): RDKit Morgan fingerprint, radius 2, 2048 bits,
  plus 8 standard descriptors (MolWt, MolLogP, HBD, HBA, TPSA, rotatable
  bonds, aromatic rings, FractionCSP3), the latter standardised on the
  training split.
* **Split**: one fixed random split, seed 42 — **303 train / 76 validation /
  95 test**.
* **Candidates** (all one-vs-rest, fitted on train only): logistic regression
  (C ∈ {0.1, 1, 10}, `class_weight="balanced"`), random forest (400 trees),
  MLP (one hidden layer of 256, α ∈ {0.1, 1.0}, early stopping), and a
  Tanimoto-similarity kNN baseline (k = 20).
* **Selection**: winner chosen by macro-AUROC on the **validation** split.
  The test split was touched exactly once, afterwards, to report final
  numbers for all candidates. The saved model is the validation winner,
  fitted on the train split.

## Results (real, from `results.json`)

Macro-AUROC (mean of per-label AUROCs):

| model | validation | test |
|---|---|---|
| logreg C=0.1 | 0.688 | **0.750** |
| logreg C=1.0 | 0.682 | 0.733 |
| logreg C=10 | 0.665 | 0.713 |
| **random forest** | **0.699** | 0.741 |
| mlp α=0.1 | 0.599 | 0.683 |
| mlp α=1.0 | 0.599 | 0.680 |
| tanimoto kNN | 0.683 | 0.701 |

The **random forest won on validation (0.699)** and is the saved model
(`smell_model.joblib`). Logistic regression (C = 0.1) edged it out on the
test split (0.750 vs 0.741) — reported openly: model selection used the
validation split only, and with 76 validation molecules the top few models
are within noise of each other. The MLP came last — 303 training molecules
is far too little data for a neural net on 2056 features, which is itself a
useful lesson. Even the "just find the 20 most similar molecules" baseline
reaches 0.70, showing how much of smell prediction is local structure
similarity.

Per-label test AUROC of the saved random forest — best and worst:

| label | AUROC | | label | AUROC |
|---|---|---|---|---|
| garlic | 0.897 | | warm | 0.668 |
| fruit | 0.856 | | musky | 0.652 |
| flower | 0.833 | | chemical | 0.645 |
| sweet | 0.830 | | sour | 0.607 |
| bakery | 0.816 | | **cold** | **0.485** |

Distinctive, chemistry-linked smells (garlic = sulfur compounds) are easiest;
frequent-but-vague ones are hardest. `chemical` applies to 83% of molecules,
so there is little signal to rank; `cold` (60 positives, a fuzzy descriptor)
lands at chance level.

## Example predictions (actual `predict.py` output)

**Vanillin** — `COc1cc(C=O)ccc1O` (this molecule sits in the held-out *test*
split, so the model never trained on it):

```
sweet 0.90 | warm 0.87 | spices 0.78 | edible 0.75 | chemical 0.70 | burnt 0.49 | musky 0.45 | bakery 0.43
ground truth: edible, bakery, sweet, fruit, spices, warm, flower
```

**Limonene** — `CC1=CCC(CC1)C(=C)C` (train split):

```
chemical 0.98 | sweet 0.91 | musky 0.81 | acid 0.23 | spices 0.20 | edible 0.18 | sour 0.16 | flower 0.14
ground truth: sweet, musky, chemical   <- the model's top 3, exactly
```

**Benzaldehyde** — `C1=CC=C(C=C1)C=O` (validation split):

```
chemical 0.83 | sweet 0.68 | sour 0.54 | musky 0.46 | warm 0.37 | flower 0.33 | spices 0.32 | edible 0.31
ground truth: edible, bakery, sweet, fruit, warm, flower, chemical
```

Two honest caveats about expectations: the Keller descriptor vocabulary has
**no "citrus" and no "almond" label**, so "limonene smells citrusy" and
"benzaldehyde smells like almonds" cannot be expressed — or checked — in
this label set at all. What can be checked agrees: limonene's predicted top 3
are exactly its human ground-truth labels.

(A SMILES correction made along the way: the string `CC1=C(C=C(C=C1)C=O)O`
sometimes quoted for vanillin is C₈H₈O₂ — it is 3-hydroxy-4-methylbenzaldehyde.
True vanillin is C₈H₈O₃, `COc1cc(C=O)ccc1O`. The model scores the wrong
molecule chemical 0.86 / sweet 0.74 — plausible for a benzaldehyde cousin.)

## Similarity sanity check

The POM paper's famous finding: molecules with very different structures can
smell alike. Searching all 112,101 molecule pairs for low structural
similarity (Morgan-fingerprint Tanimoto ≤ 0.25) but near-identical predicted
smell profiles (cosine ≥ 0.90) finds 9,620 pairs — that count is inflated by
base rates (most profiles are dominated by `chemical`/`sweet`/`musky`), so
the meaningful check is whether the top pairs also share *human* labels.
The best one does, emphatically:

* **Whiskey lactone ↔ Acetanisole** — Tanimoto **0.07** (almost no shared
  structure), predicted-profile cosine 0.997, and they share **8 ground-truth
  labels**: edible, bakery, sweet, fruit, spices, warm, flower, chemical.
* Fumaric acid ↔ Benzoic acid — Tanimoto 0.21; share sweet, sour, musky,
  chemical.
* Butyl formate ↔ Acetal — Tanimoto 0.16; share sweet, chemical.

Run `similarity_check.py` to reproduce.

## Limitations — read before believing any of it

* **Toy scale.** 474 molecules and fingerprints + classic ML vs Osmo's POM:
  a graph neural network trained on ~5,000 odorants curated for the task.
  Our macro-AUROC ≈ 0.74; treat predictions as plausible guesses, not truth.
* **Labels are panel aggregates.** Human perception varies by person and by
  concentration; the dataset pools dilutions from 1/10 to 1/10,000,000 and
  replicate trials into one rate, and the ≥ 15% binarisation threshold is a
  modelling choice, not a fact of nature.
* **The vocabulary is Keller's 20 words.** No citrus, almond, vanilla,
  or most other specific notes exist as labels.
* **Stereochemistry is invisible here** — the dataset's SMILES are
  non-isomeric, yet enantiomers can smell different (the classic example is
  carvone: spearmint vs caraway).
* **Small validation set** (76 molecules): the ranking of the top models is
  within noise; don't read meaning into the third decimal.
* **Probabilities are model scores**, not calibrated chances, and random
  forest probabilities are especially coarse.
* **pyrfume note:** the `pyrfume` Python package could not be used in this
  environment — pyrfume 0.19.1 depends on `quantities`, which breaks under
  NumPy ≥ 2 (`ndarray.ptp` was removed), while its pinned RDKit (< 2024)
  conflicts the other way. The data therefore comes from the *same official
  source* — the pyrfume-data GitHub repository — as raw CSVs, which has the
  side benefit that every processing step is visible in `prepare_data.py`.

## Files

| file | what |
|---|---|
| `prepare_data.py` | raw CSVs → per-molecule rates and binarised labels |
| `features.py` | shared SMILES → fingerprint + descriptors featurisation |
| `train.py` | split, train all candidates, validation selection, one test pass |
| `predict.py` | CLI: SMILES in → top predicted descriptors with bars |
| `similarity_check.py` | the structurally-different-but-similar-smell pair search |
| `results.json` | all validation/test metrics, incl. per-label AUROCs |
| `smell_model.joblib` | the saved random-forest bundle used by `predict.py` (~91 MB) |
| `data/` | raw CSVs from pyrfume-data + processed `molecule_labels.csv` / `molecule_rates.csv` |

Tested versions: Python 3.12, RDKit 2026.09.1, scikit-learn 1.9.1,
pandas 3.1.1, NumPy 2.5.4.
