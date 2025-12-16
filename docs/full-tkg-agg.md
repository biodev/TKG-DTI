# Data Dictionary 


| Column | Type | Description |
|--------|------|-------------|
| `drug` | int | Drug node index used in model prediction (maps to `data.node_name_dict['drug']`) |
| `protein` | int | Protein node index used in model prediction (maps to `data.node_name_dict['gene']`) |
| `score` | list[float] | Predicted link scores aggregated across folds meeting inclusion criteria. Higher scores indicate stronger predicted drug-target interaction. |
| `fnr_est` | list[float] | Estimated False Negative Rate per fold. `fnr_est` = proportion of known DTIs (test set) with scores **below** this observation. **Higher `fnr_est` → observation scores higher than most known DTIs → stronger DTI evidence.** |
| `fpr_est` | list[float] | Estimated False Positive Rate per fold. `fpr_est` = proportion of negatives with scores **at or above** this observation. **Lower `fpr_est` → observation scores higher than most negatives → stronger DTI evidence.** |
| `tnr_est` | list[float] | Estimated True Negative Rate (Specificity) per fold. TNR = TN/(TN+FP) = proportion of negatives correctly identified. |
| `tpr_est` | list[float] | Estimated True Positive Rate (Sensitivity/Recall) per fold. TPR = TP/(TP+FN) = proportion of positives correctly identified. |
| `fnr_est_mean` | float | Mean of `fnr_est` across included folds |
| `fpr_est_mean` | float | Mean of `fpr_est` across included folds |
| `score_mean` | float | Mean of `score` across included folds |
| `in_n_folds` | int | Number of folds in which observation met inclusion criteria (max=10) |
| `inchikey` | str | Drug InChIKey identifier (standardized molecular identifier) |
| `gene_symbol` | str | Protein gene symbol (HGNC) |
| `inhibitor` | str | Drug common name |
| `uniprot_id` | str | Protein UniProt accession ID |

## Raw Predictions (df from read_preds)

| Column | Type | Description |
|--------|------|-------------|
| `drug` | int | Drug node index |
| `protein` | int | Protein node index |
| `score` | float | Raw predicted link score |
| `test` | bool | True if this drug-protein pair is a known DTI (positive validation sample) |
| `negatives` | bool | True if this is a candidate pair not in training/test (unknown interaction status) |
| `uid` | str | Unique identifier for the model run |
| `fold` | int | Cross-validation fold number (0-9) |

---

# Overview of Aggregation

Predictions are aggregated across 10 cross-validation folds:

1. **Per-fold processing**: For each fold, load predictions and estimate FNR/FPR at each score using Gaussian Process regression fitted on threshold-metric curves.
2. **Inclusion filtering**: Retain only `negatives` (candidate DTIs) meeting criteria: `fnr_est <= max_fnr` AND `fpr_est >= min_fpr`
3. **Cross-fold aggregation**: Group by (drug, protein) and collect scores/metrics as lists from all folds where the pair passed filtering.
4. **Summary statistics**: Compute means and count how many folds each pair appeared in.

---

# Overview of Inclusion Criteria

Fold-level filtering selects high-confidence DTI predictions using two thresholds:

### Filter Logic
```python
df = df[lambda x: x.negatives & (x.fnr_est >= min_fnr)]  # Score above min_fnr fraction of known DTIs
df = df[lambda x: x.negatives & (x.fpr_est <= max_fpr)]  # Score above (1-max_fpr) fraction of negatives
```

### Parameter Interpretation

| Parameter | Default | Effect | Example |
|-----------|---------|--------|---------|
| `min_fnr` | 0.0 | Keep observations scoring above `min_fnr` fraction of known DTIs | `min_fnr=0.9` → top 10% vs known DTIs |
| `max_fpr` | 1.0 | Keep observations scoring above `(1-max_fpr)` fraction of negatives | `max_fpr=0.1` → top 10% vs negatives |

### Primary Filter: `min_fnr`

**`min_fnr` is the preferred filtering criterion** because:
- Known DTIs (test set) have high confidence — they are validated interactions
- "Negatives" are merely assumed negatives — many may be undiscovered true DTIs
- Filtering relative to known DTIs provides a more reliable quality threshold

With `min_fnr=0.9`: Keep only predictions that score higher than 90% of known DTIs.

### Secondary Filter: `max_fpr`

Use `max_fpr` for additional stringency when needed:
- With `max_fpr=0.1`: Keep only predictions scoring higher than 90% of negatives

### Class Imbalance Caveat
Drug-target interactions are sparse. Many "negatives" may be undiscovered true DTIs, which is why `min_fnr` (based on confident positives) is preferred over `max_fpr` (based on uncertain negatives).

### Tertiary Filter: `in_n_folds` 

While we don't apply any filtering using `in_n_folds` we do provide it as a confidence metric. If a DTI appears in only 1 or a few folds it is likely lower confidence than if it appears in every fold (max: 10). 

---

# Metric Formulas Reference

## Standard Definitions (at threshold T)

| Metric | Formula | Standard Interpretation |
|--------|---------|-------------------------|
| FPR | FP / (TN + FP) | Proportion of negatives with score ≥ T |
| FNR | FN / (TP + FN) | Proportion of positives with score < T |
| TPR | TP / (TP + FN) | Proportion of positives with score ≥ T (= 1 - FNR) |
| TNR | TN / (TN + FP) | Proportion of negatives with score < T (= 1 - FPR) |

## Per-Observation Estimates (threshold = observation's score)

| Estimate | Meaning | High Value Indicates |
|----------|---------|---------------------|
| `fpr_est` | Fraction of negatives scoring ≥ this observation | **Low score** (many negatives score as high or higher) |
| `fnr_est` | Fraction of positives scoring < this observation | **High score** (most positives score lower) |
| `tpr_est` | Fraction of positives scoring ≥ this observation | **Low score** (most positives score as high or higher) |
| `tnr_est` | Fraction of negatives scoring < this observation | **High score** (most negatives score lower) |

## Quick Reference: What Makes a Strong DTI Prediction?

A high-scoring observation (strong DTI evidence) will have:
- ✅ **HIGH `fnr_est`** — observation scores higher than most known DTIs → use `min_fnr` to filter
- ✅ **LOW `fpr_est`** — observation scores higher than most negatives → use `max_fpr` to filter
- ✅ **HIGH `tnr_est`** — most negatives score lower (equivalent to low `fpr_est`)
- ✅ **LOW `tpr_est`** — few known DTIs score this high (equivalent to high `fnr_est`) 
