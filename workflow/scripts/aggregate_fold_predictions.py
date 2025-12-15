#!/usr/bin/env python3
"""
Aggregate predictions across K-fold cross-validation folds.

This script follows the logic of notebooks/00_AGG_FULL_TKG_PREDS.ipynb:
1. For each fold, read predictions and compute FNR/FPR estimates using linear interpolation
2. Filter predictions based on min_fnr and max_fpr thresholds
3. Aggregate across folds, computing mean metrics and fold counts
4. Add metadata (inchikey, gene_symbol, inhibitor, uniprot_id)
5. Save the final aggregated predictions

Author: Generated from notebook logic
"""

import argparse
import os
import pandas as pd
import numpy as np
import torch


def get_args():
    parser = argparse.ArgumentParser(description="Aggregate predictions across K folds")
    
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory containing FOLD_X subdirectories with predictions")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to Data.pt file containing node_name_dict")
    parser.add_argument("--targetome_path", type=str, required=True,
                        help="Path to targetome__drug_targets_gene.csv for metadata")
    parser.add_argument("--out", type=str, required=True,
                        help="Output directory for aggregated predictions")
    parser.add_argument("--k_folds", type=int, default=10,
                        help="Number of folds to aggregate")
    parser.add_argument("--min_fnr", type=float, default=0.5,
                        help="Minimum FNR threshold (keep predictions scoring above this fraction of known DTIs)")
    parser.add_argument("--max_fpr", type=float, default=1.0,
                        help="Maximum FPR threshold (keep predictions scoring above (1-max_fpr) fraction of negatives)")
    parser.add_argument("--n_thresholds", type=int, default=250,
                        help="Number of thresholds to use for metric interpolation")
    parser.add_argument("--min_folds", type=int, default=1,
                        help="Minimum number of folds a prediction must appear in to be included")
    
    return parser.parse_args()


def read_preds(root: str, fold: int) -> pd.DataFrame:
    """
    Read predictions from a fold directory.
    
    The training script creates a uid subdirectory, so we search for predictions.csv
    within subdirectories of the fold directory.
    
    Args:
        root: Root directory containing FOLD_X subdirectories
        fold: Fold index (0-based)
        
    Returns:
        DataFrame with predictions and uid column
    """
    path = os.path.join(root, f'FOLD_{fold}')
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fold directory not found: {path}")
    
    uid_dirs = os.listdir(path)
    
    df = None
    for u in uid_dirs:
        pred_path = os.path.join(path, u, 'predictions.csv')
        if os.path.exists(pred_path):
            if df is not None:
                raise ValueError(f'Multiple predictions files found in fold {fold}')
            df = pd.read_csv(pred_path)
            df = df.assign(uid=u)
    
    if df is None:
        raise ValueError(f'No predictions file found in fold {fold}')
    
    df = df.assign(fold=fold)
    
    return df


def compute_classification_metrics(pos_scores: np.ndarray, 
                                   neg_scores: np.ndarray, 
                                   threshold: float) -> dict:
    """
    Compute classification metrics given positive scores, negative scores, and a threshold.
    
    Args:
        pos_scores: Array of scores for true positive samples
        neg_scores: Array of scores for true negative samples  
        threshold: Threshold to classify scores as positive (>= threshold) or negative (< threshold)
    
    Returns:
        dict with keys: TP, TN, FP, FN, FPR, FNR, TNR, TPR
    """
    threshold = np.array([threshold])
    
    # Compute confusion matrix components
    TP = np.sum(pos_scores >= threshold)  # True positives predicted as positive
    FN = np.sum(pos_scores < threshold)   # True positives predicted as negative
    FP = np.sum(neg_scores >= threshold)  # True negatives predicted as positive
    TN = np.sum(neg_scores < threshold)   # True negatives predicted as negative
    
    # False negative rate
    if (TP + FN) > 0:
        FNR = FN / (TP + FN)
    else:
        FNR = 0.0

    # True Positive Rate (Sensitivity/Recall)
    if (TP + FN) > 0:
        TPR = TP / (TP + FN)
    else:
        TPR = 0.0

    # False Positive Rate
    if (TN + FP) > 0:
        FPR = FP / (TN + FP)
    else:
        FPR = 0.0

    # True Negative Rate (Specificity)
    if (TN + FP) > 0:
        TNR = TN / (TN + FP)
    else:
        TNR = 0.0
    
    return {
        'TP': int(TP),
        'TN': int(TN),
        'FP': int(FP),
        'FN': int(FN),
        'FPR': float(FPR),
        'FNR': float(FNR),
        'TNR': float(TNR),
        'TPR': float(TPR),
    }


def compute_fdr_est(df: pd.DataFrame, 
                    min_fnr: float = 0.0, 
                    max_fpr: float = 1.0, 
                    n_thresholds: int = 250) -> pd.DataFrame:
    """
    Estimate FNR/FPR for each observation and filter for high-confidence DTI predictions.
    
    Uses linear interpolation on monotonic metric curves computed at discrete thresholds.
    
    Args:
        df: DataFrame with columns 'score', 'test', 'negatives'
        min_fnr: Minimum required FNR. Observations must score higher than min_fnr fraction of known DTIs.
                 Higher values = stricter filter (e.g., 0.9 keeps only predictions in top 10% vs known DTIs).
        max_fpr: Maximum allowed FPR. Observations must score higher than (1 - max_fpr) fraction of negatives.
                 Lower values = stricter filter (e.g., 0.1 keeps only predictions in top 10% vs negatives).
        n_thresholds: Number of thresholds to compute metrics at
                 
    Returns:
        Filtered DataFrame with additional columns: fnr_est, fpr_est, tnr_est, tpr_est
    """
    print('Computing metric estimates...')
    test_scores = df[lambda x: x.test].score.values
    neg_scores = df[lambda x: x.negatives].score.values
    all_scores = df.score.values

    # Cover the full range of all scores
    score_min = min(test_scores.min(), neg_scores.min(), all_scores.min())
    score_max = max(test_scores.max(), neg_scores.max(), all_scores.max())
    
    ts = np.linspace(score_min, score_max, n_thresholds)

    metrics = []
    for ii, threshold in enumerate(ts):
        print(f'{ii}/{len(ts)}', end='\r')

        dict_ = compute_classification_metrics(test_scores, neg_scores, threshold)
        df_ = pd.DataFrame(dict_, index=[0]).assign(threshold=threshold)
        metrics.append(df_)
    print()

    metrics = pd.concat(metrics).reset_index(drop=True)
    thresholds = metrics.threshold.values

    # Use linear interpolation for monotonic metric curves
    print('Interpolating FNR estimates...')
    fnr_est = np.interp(all_scores, thresholds, metrics.FNR.values)
    df = df.assign(fnr_est=np.clip(fnr_est, 0, 1))

    print('Interpolating FPR estimates...')
    fpr_est = np.interp(all_scores, thresholds, metrics.FPR.values)
    df = df.assign(fpr_est=np.clip(fpr_est, 0, 1))

    print('Interpolating TNR estimates...')
    tnr_est = np.interp(all_scores, thresholds, metrics.TNR.values)
    df = df.assign(tnr_est=np.clip(tnr_est, 0, 1))

    print('Interpolating TPR estimates...')
    tpr_est = np.interp(all_scores, thresholds, metrics.TPR.values)
    df = df.assign(tpr_est=np.clip(tpr_est, 0, 1))

    # Filter for high-scoring observations
    print(f'Filtering negatives with fnr_est >= {min_fnr} (scores above {min_fnr*100:.0f}% of known DTIs)...')
    df = df[lambda x: x.negatives & (x.fnr_est >= min_fnr)]

    print(f'Filtering negatives with fpr_est <= {max_fpr} (scores above {(1-max_fpr)*100:.0f}% of negatives)...')
    df = df[df.fpr_est <= max_fpr]

    print(f'Final # predicted DTIs: {df.shape[0]}')

    return df


def main():
    args = get_args()
    
    print("=" * 80)
    print("Aggregating predictions across folds")
    print("=" * 80)
    print(f"Root directory: {args.root}")
    print(f"Data path: {args.data_path}")
    print(f"Targetome path: {args.targetome_path}")
    print(f"Output directory: {args.out}")
    print(f"K folds: {args.k_folds}")
    print(f"Min FNR: {args.min_fnr}")
    print(f"Max FPR: {args.max_fpr}")
    print(f"N thresholds: {args.n_thresholds}")
    print(f"Min folds: {args.min_folds}")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Load data for node name mapping
    print("\nLoading Data.pt for node name mappings...")
    data = torch.load(args.data_path, weights_only=False)
    
    # Load targetome for metadata
    print("Loading targetome for metadata...")
    targ = pd.read_csv(args.targetome_path)
    inchi2name = targ[['inchikey', 'inhibitor']].drop_duplicates()
    gene2uniprot = targ[['gene_symbol', 'uniprot_id']].drop_duplicates()
    
    # Process each fold
    dfs = []
    for fold in range(args.k_folds):
        print('\n' + '-' * 80)
        print(f'Processing fold {fold}/{args.k_folds}')
        
        try:
            df = read_preds(args.root, fold)
            df = compute_fdr_est(df, 
                                min_fnr=args.min_fnr, 
                                max_fpr=args.max_fpr, 
                                n_thresholds=args.n_thresholds)
            dfs.append(df)
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Skipping fold {fold}: {e}")
            continue
    
    if len(dfs) == 0:
        raise ValueError("No predictions found in any fold!")
    
    print(f"\n{'=' * 80}")
    print(f"Successfully processed {len(dfs)} folds")
    print(f"{'=' * 80}")
    
    # Concatenate all fold predictions
    preds = pd.concat(dfs)
    print(f"\nTotal predictions before aggregation: {preds.shape[0]}")
    
    # Aggregate across folds
    print("\nAggregating across folds...")
    preds2 = preds.groupby(['drug', 'protein'])[['score', 'fnr_est', 'fpr_est', 'tnr_est', 'tpr_est']].agg(lambda x: list(x))
    preds2 = preds2.reset_index()
    
    # Compute summary statistics
    preds2 = preds2.assign(fnr_est_mean=[np.mean(x) for x in preds2.fnr_est])
    preds2 = preds2.assign(fpr_est_mean=[np.mean(x) for x in preds2.fpr_est])
    preds2 = preds2.assign(tnr_est_mean=[np.mean(x) for x in preds2.tnr_est])
    preds2 = preds2.assign(tpr_est_mean=[np.mean(x) for x in preds2.tpr_est])
    preds2 = preds2.assign(score_mean=[np.mean(x) for x in preds2.score])
    preds2 = preds2.assign(in_n_folds=[len(x) for x in preds2.fnr_est])
    
    # Filter by minimum fold count
    if args.min_folds > 1:
        print(f"\nFiltering predictions appearing in >= {args.min_folds} folds...")
        preds2 = preds2[preds2.in_n_folds >= args.min_folds]
    
    print(f"Predictions after aggregation: {preds2.shape[0]}")
    
    # Add metadata
    print("\nAdding metadata...")
    preds2 = preds2.assign(
        inchikey=data.node_name_dict['drug'][preds2.drug.values],
        gene_symbol=data.node_name_dict['gene'][preds2.protein.values]
    )
    
    preds2 = preds2.merge(inchi2name, on='inchikey', how='left')
    preds2 = preds2.merge(gene2uniprot, on='gene_symbol', how='left')
    
    # Sort by fnr_est_mean descending (higher = stronger evidence)
    preds2 = preds2.sort_values('fnr_est_mean', ascending=False)
    
    # Save aggregated predictions
    out_path = os.path.join(args.out, 'aggregated_predictions.csv')
    print(f"\nSaving aggregated predictions to: {out_path}")
    
    # Convert list columns to string for CSV compatibility
    preds2_save = preds2.copy()
    for col in ['score', 'fnr_est', 'fpr_est', 'tnr_est', 'tpr_est']:
        preds2_save[col] = preds2_save[col].apply(lambda x: str(x))
    
    preds2_save.to_csv(out_path, index=False)
    
    # Also save a summary
    summary = {
        'n_folds_processed': len(dfs),
        'k_folds': args.k_folds,
        'min_fnr': args.min_fnr,
        'max_fpr': args.max_fpr,
        'n_thresholds': args.n_thresholds,
        'min_folds': args.min_folds,
        'n_predictions': preds2.shape[0],
        'n_unique_drugs': preds2.drug.nunique(),
        'n_unique_proteins': preds2.protein.nunique(),
    }
    
    summary_path = os.path.join(args.out, 'aggregation_summary.csv')
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"Saved aggregation summary to: {summary_path}")
    
    print("\n" + "=" * 80)
    print("Aggregation complete!")
    print(f"Total unique drug-protein predictions: {preds2.shape[0]}")
    print(f"Unique drugs: {preds2.drug.nunique()}")
    print(f"Unique proteins: {preds2.protein.nunique()}")
    print("=" * 80)


if __name__ == "__main__":
    main()

