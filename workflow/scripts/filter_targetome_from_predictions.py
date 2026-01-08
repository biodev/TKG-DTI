#!/usr/bin/env python3
"""
Filter targetome annotations from aggregated predictions.

This script removes predictions that have a targetome annotation, keeping only
novel DTI predictions that are not already in targetome.

Author: Generated from notebook logic (00_PREDICTED_NEGATIVES_VS_TARGETOME.ipynb)
"""

import argparse
import os
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(
        description="Filter predictions to remove those with targetome annotations"
    )
    
    parser.add_argument("--aggregated_predictions", type=str, required=True,
                        help="Path to aggregated_predictions.csv")
    parser.add_argument("--targetome_path", type=str, required=True,
                        help="Path to targetome_expanded CSV file")
    parser.add_argument("--out", type=str, required=True,
                        help="Output directory for filtered predictions")
    
    return parser.parse_args()


def main():
    args = get_args()
    
    print("=" * 80)
    print("Filtering targetome annotations from predictions")
    print("=" * 80)
    print(f"Aggregated predictions: {args.aggregated_predictions}")
    print(f"Targetome path: {args.targetome_path}")
    print(f"Output directory: {args.out}")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Load aggregated predictions
    print("\nLoading aggregated predictions...")
    preds = pd.read_csv(args.aggregated_predictions)
    print(f"Total predictions: {preds.shape[0]}")
    
    # Load targetome
    print("\nLoading targetome...")
    targetome = pd.read_csv(args.targetome_path)
    print(f"Total targetome entries: {targetome.shape[0]}")
    
    # Process targetome to get unique (inchi_key, uniprot_id) pairs
    # Keep only entries with assay_relation '=' and valid assay values
    targetome_filtered = targetome[['inchi_key', 'uniprot_id', 'assay_relation', 'assay_value']].drop_duplicates()
    
    # Filter all targetome entries
    #if 'assay_relation' in targetome_filtered.columns:
    #    targetome_filtered = targetome_filtered[targetome_filtered.assay_relation == '=']
    
    # Get unique (inchi_key, uniprot_id) pairs with valid assay values
    targetome_filtered = targetome_filtered[targetome_filtered.assay_value.notna()]
    targetome_pairs = targetome_filtered[['inchi_key', 'uniprot_id']].drop_duplicates()
    targetome_pairs = targetome_pairs.assign(_has_targetome=True)
    print(f"Unique targetome (drug, protein) pairs: {targetome_pairs.shape[0]}")
    
    # Merge predictions with targetome to identify matches
    # The aggregated predictions use 'inchikey' and 'uniprot_id' columns
    print("\nMerging predictions with targetome...")
    preds_merged = preds.merge(
        targetome_pairs, 
        left_on=['inchikey', 'uniprot_id'], 
        right_on=['inchi_key', 'uniprot_id'], 
        how='left'
    )
    
    # Count how many predictions have targetome annotations
    n_with_targetome = preds_merged._has_targetome.notna().sum()
    print(f"Predictions with targetome annotation: {n_with_targetome} ({n_with_targetome/preds.shape[0]*100:.2f}%)")
    
    # Filter out predictions with targetome annotations
    preds_filtered = preds_merged[preds_merged._has_targetome.isna()].copy()
    
    # Drop the helper columns
    preds_filtered = preds_filtered.drop(columns=['_has_targetome', 'inchi_key'], errors='ignore')
    
    print(f"Predictions after filtering: {preds_filtered.shape[0]} ({preds_filtered.shape[0]/preds.shape[0]*100:.2f}% retained)")
    
    # Save filtered predictions
    out_path = os.path.join(args.out, 'aggregated_and_filtered_predictions.csv')
    print(f"\nSaving filtered predictions to: {out_path}")
    preds_filtered.to_csv(out_path, index=False)
    
    print("\n" + "=" * 80)
    print("Filtering complete!")
    print(f"Original predictions: {preds.shape[0]}")
    print(f"Removed (with targetome): {n_with_targetome}")
    print(f"Remaining (novel): {preds_filtered.shape[0]}")
    print("=" * 80)


if __name__ == "__main__":
    main()
