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

## v2 — bigger data + a graph neural network

v2 asks the two obvious follow-ups to v1: does **more data** help, and does a
**graph neural network** (the architecture Osmo's POM actually uses) beat
fingerprints + random forest? Everything from v1 is untouched; v2 lives in
`prepare_data_v2.py`, `graphs.py`, `train_gnn.py`, `predict_v2.py`,
`results_v2.json` and the checkpoint `smell_gnn.pt` (0.52 MB).

```bash
./venv/bin/pip install -r requirements_v2.txt   # torch (CPU) + torch_geometric
./venv/bin/python prepare_data_v2.py            # -> data/molecule_labels_v2.csv (+ mask)
./venv/bin/python train_gnn.py                  # -> smell_gnn.pt, results_v2.json
./venv/bin/python predict_v2.py "COc1cc(C=O)ccc1O"
```

### The enlarged dataset (3,465 molecules)

Two more descriptor-labelled sets from the same official pyrfume-data repo
were merged with Keller 2016 (all merges by RDKit-canonical SMILES with
stereochemistry stripped, matching v1's treatment; a molecule in several
sources keeps the **union** of its labels):

| source | molecules | what it is | labels it can express |
|---|---|---|---|
| Keller & Vosshall 2016 | 474 | panel ratings (v1's data) | all 20 |
| Dravnieks 1985 | 138 | panel *applicability* (%) on 146 words | 19 of 20 (no `edible`) |
| Leffingwell (PMP 2001) | 3,400 | binary expert-catalogue tags on 113 words | 16 of 20 (no `edible`, `cold`, `acid`, `chemical`) |

* **Dravnieks**: its applicability measure is the same quantity Keller
  pooled, so it is binarised at the same ≥ 15% rule; its 146 words are
  mapped onto Keller's 20 with explicit family tables (family score = max
  of its words — e.g. Keller `fruit` ← any of 14 Dravnieks fruit words).
  16 of its 160 stimuli have no CID/SMILES in the source and were dropped.
* **Leffingwell**: a listed word counts as positive, an unlisted word as
  negative *within the mapped families* (the conventional use of this
  dataset, but weaker evidence than panel ratings — see the caveat below).
  Two mappings are flagged proxies: Keller `sweaty` ← "cheesy" and
  `ammonia_urinous` ← "catty" (the perfumery terms for those notes).
* **Excluded — Keller 2012**: its descriptor table covers only 3 odorants
  (the rest of that dataset is intensity/pleasantness), so it cannot
  enlarge a descriptor dataset.
* Labels a source cannot express are **not** faked as 0: a per-label
  observed-mask (`data/molecule_mask_v2.csv`) travels with the data, the
  GNN trains with a masked loss, and each control forest trains only on
  molecules that observe its label. `edible` is learned from Keller
  molecules only; `cold`/`acid`/`chemical` from Keller + Dravnieks (543
  molecules); the other 14 labels from all 3,465.
* Overlaps: 69 molecules are in both Keller and Dravnieks, 430 in both
  Keller and Leffingwell, 112 in both Dravnieks and Leffingwell.
* **Merge caveat, measured**: on molecules shared with Keller, the other
  sources agree with Keller's labels 74% (Dravnieks) / 73% (Leffingwell)
  of the time overall, but recover only 34% / 19% of Keller's *positives*.
  The added sources are far more conservative about calling a smell
  present — "not listed" often means "not characteristic", not "absent".
  Expect label noise that mostly manufactures false negatives.

### Fair-comparison design

Validation (76) and test (95) are **exactly v1's Keller molecules** — same
seed-42 split, recovered from `train.py` and verified (vanillin ∈ test,
limonene ∈ train, benzaldehyde ∈ val, as in v1's README). Every non-Keller
molecule was assigned to train (train pool: 3,294). The GNN's checkpoint
was selected on validation only; test was touched once. As a control, v1's
winning recipe (400-tree RF on the same fingerprints) was retrained on the
enlarged pool — one forest per label, each on the molecules observing that
label — so "more data" and "new architecture" can be told apart.

### The GNN

Atoms are nodes (42-dim one-hot features: element, degree, charge,
hybridisation, aromaticity, H count, ring, chirality), bonds are bidirectional
edges with a one-hot bond type. Architecture: **3 × GINEConv (hidden 128) +
BatchNorm + ReLU, mean+max global pooling, MLP head → 18 logits**, trained
with masked BCE (per-label `pos_weight`), Adam (lr 1e-3), batch 64, early
stopping on validation (best checkpoint at epoch 7; stopped at 67).

### Results (real, from `results_v2.json`)

Macro-AUROC over the 18 descriptors, on v1's exact evaluation molecules:

| model | train data | validation | test |
|---|---|---|---|
| random forest (v1) | 303 Keller | 0.699 | **0.741** |
| random forest (v2 control) | 3,294 merged | 0.689 | 0.692 |
| **GNN (v2)** | 3,294 merged | **0.706** | 0.709 |

Honest reading:

* **The GNN is the best model on validation** and beats the RF control
  trained on the *same* enlarged data on both splits (0.706/0.689 val,
  0.709/0.692 test) — given identical training molecules, the graph
  architecture genuinely helps here.
* **But v1's little RF still wins on test** (0.741). The enlarged data
  *hurt* the RF overall (0.741 → 0.692): the added sources' conservative
  positives clash with Keller's panel semantics on exactly the molecules
  we test on. More data only helps if it speaks the same language.
* The aggregate hides big per-label swings. The GNN fixed v1's hopeless
  `cold` (0.485 → 0.664, thanks to Dravnieks actually having a "cool,
  cooling" word) and is strong on `flower` (0.923), `garlic` (0.908),
  `bakery` (0.874), `sweet` (0.828). It is *worse* on `chemical` (0.481,
  chance), `musky` (0.539) and `decayed` (0.557) — labels where the added
  data is sparsest or most idiosyncratic (Leffingwell tags "musk" only
  for true musks, vs 278/474 Keller positives).
* With 76 validation and 95 test molecules, differences of ±0.03 are
  within noise. Treat the ranking as suggestive, not settled.

### Example predictions — GNN (`predict_v2.py`, actual output)

**Vanillin** (test molecule; merged ground truth: edible, bakery, sweet,
fruit, spices, warm, flower):

```
spices 0.87 | warm 0.80 | sweet 0.77 | chemical 0.72 | wood 0.67 | edible 0.66 | cold 0.65 | burnt 0.65
```

**Limonene** (train molecule; merged ground truth: sweet, fruit, cold,
musky, chemical — the union added `fruit` and `cold` to v1's Keller-only
truth):

```
wood 0.88 | cold 0.75 | spices 0.71 | flower 0.63 | warm 0.62 | sweet 0.62 | musky 0.47 | chemical 0.47
```

Limonene is the GNN's visibly worst showcase: v1's RF had its top 3 exactly
right; the GNN leads with `wood` (likely — this is interpretation, not
measurement — because Leffingwell's wood family includes "pine", a note
carried by many terpenes like limonene). Its true labels sit at ranks 6–8
instead.

**Benzaldehyde** (validation molecule; merged ground truth: edible,
bakery, sweet, fruit, spices, warm, flower, chemical):

```
spices 0.80 | warm 0.72 | sweet 0.69 | chemical 0.67 | cold 0.65 | flower 0.65 | musky 0.58 | wood 0.57
```

### v2 limitations (in addition to v1's)

* **Mixed label semantics** is the big one, quantified above: panel
  application rates, atlas applicability and catalogue tags are three
  different answers to three slightly different questions.
* The GNN's best validation epoch was 7 of 400 — on this data it starts
  memorising almost immediately; the win over the RF control is real but
  modest, and checkpoint choice rides on 76 molecules.
* `fish` (73 positives) and `grass` (935) now have more merged data but
  remain unpredicted, to stay comparable with v1's 18 targets.
* Added in v2: `graphs.py` (SMILES → PyG graphs), `prepare_data_v2.py`,
  `train_gnn.py`, `predict_v2.py`, `results_v2.json`, `smell_gnn.pt`,
  `requirements_v2.txt`, `data_v2/` (raw source CSVs for Dravnieks and
  Leffingwell; the inspected-but-excluded Keller 2012 download is
  git-ignored), `data/molecule_labels_v2.csv`, `data/molecule_mask_v2.csv`.

Tested versions for v2: PyTorch 2.15.0+cpu, torch_geometric 2.8.1,
on top of v1's stack.
