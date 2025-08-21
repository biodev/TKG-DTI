#!/usr/bin/env python3
"""
KG construction step 10: Build targetome knowledge Graph (TKG) from relations

This script creates a knowledge graph by building PyTorch Geometric HeteroData
objects from all relation CSV files. It performs K-fold cross-validation splits on the
target drug-target interactions while keeping all other relations in the training set.

The script follows the logic from the 00_MAKE_TKG.ipynb notebook, creating cross-validation
folds for link prediction tasks. Each fold contains train/validation/test splits where:
- Train: Contains most target DTIs plus all auxiliary relations  
- Validation: Contains a subset of target DTIs for hyperparameter tuning
- Test: Contains held-out target DTIs for final evaluation

The output includes PyTorch tensors for each fold containing the structured graph data
and train/val/test splits suitable for knowledge graph embedding models.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch_geometric as pyg
from sklearn.model_selection import KFold

from tkgdti.data.GraphBuilder import GraphBuilder


def get_args():
    """Parse command line arguments for TKG construction."""
    parser = argparse.ArgumentParser(
        description="KG construction step 10: Build Temporal Knowledge Graph with cross-validation folds."
    )
    parser.add_argument("--data", type=str, default="../../../data/", help="Path to the input data dir")
    parser.add_argument("--extdata", type=str, default="../../extdata/", help="Path to the extra data dir")
    parser.add_argument("--out", type=str, default="../../output/", help="Path to the output data dir")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--k_folds", type=int, default=10, help="Number of K-fold cross-validation splits")
    parser.add_argument("--val_prop", type=float, default=0.075, help="Proportion of training data to use for validation")
    parser.add_argument("--tkg_output_dir", type=str, default="tkge_no_patient", help="Output directory name for TKG data")
    parser.add_argument("--relations_root", type=str, default="extdata/relations", help="Root directory containing relation CSV files")
    parser.add_argument("--exclude_patient_relations", action="store_true", default=True, help="Exclude relations involving patient data")
    parser.add_argument("--no_rev", action="store_true", default=False, help="Exclude reverse relations (those ending with _rev)")
    
    return parser.parse_args()


def get_relation_files(relations_root, exclude_patient_relations=True, no_rev=False):
    """
    Get list of relation CSV files to include in the knowledge graph.
    
    Parameters
    ----------
    relations_root : str
        Path to directory containing relation CSV files
    exclude_patient_relations : bool
        Whether to exclude relations involving patient/subject data
    no_rev : bool
        Whether to exclude reverse relations (_rev files)
        
    Returns
    -------
    list
        List of relation filenames to include
    """
    if not os.path.exists(relations_root):
        raise FileNotFoundError(f"Relations directory not found: {relations_root}")
    
    relnames = [f for f in os.listdir(relations_root) if f.endswith('.csv')]
    
    if no_rev:
        relnames = [r for r in relnames if '_rev' not in r]
    
    # Relations to exclude based on ablation study results
    DO_NOT_INCLUDE_RELATIONS = [
        'gene->mut_missense_variant_deleterious_rev->dbgap_subject',
        'gene->isin_fwd->pathway', 
        'gene->associates_fwd->disease',
        'gene->A549_lincs_perturbation_rev->drug'
    ]
    
    excluded_count = 0
    valid_relnames = []
    
    for rname in relnames:
        try:
            rdf = pd.read_csv(os.path.join(relations_root, rname))
            
            # Skip empty relation files
            if rdf.shape[0] == 0:
                print(f"no relations in df: {rname}")
                excluded_count += 1
                continue
            
            # Exclude patient relations if requested
            if exclude_patient_relations:
                if 'dbgap_subject' in rdf.src_type.values or 'dbgap_subject' in rdf.dst_type.values:
                    print(f"excluding patient relation: {rname}")
                    excluded_count += 1
                    continue
            
            # Check against ablation study exclusions
            rel_type = f'{rdf.src_type.values[0]}->{rdf.relation.values[0]}->{rdf.dst_type.values[0]}'
            if rel_type in DO_NOT_INCLUDE_RELATIONS:
                print(f"excluding based on ablation study: {rname}")
                excluded_count += 1
                continue
                
            valid_relnames.append(rname)
            
        except Exception as e:
            print(f"Error processing {rname}: {e}")
            excluded_count += 1
            continue
    
    print(f"# excluded: {excluded_count}")
    print(f"# of relation types: {len(valid_relnames)}")
    
    return valid_relnames


def build_fold_data(relations_root, relnames, train_idxs, val_idxs, test_idxs, 
                   target_key=('drug', 'targets', 'gene')):
    """
    Build graph data for a single cross-validation fold.
    
    Parameters
    ---------- 
    relations_root : str
        Path to relations directory
    relnames : list
        List of relation filenames to include
    train_idxs : array-like
        Indices of target relations for training
    val_idxs : array-like  
        Indices of target relations for validation
    test_idxs : array-like
        Indices of target relations for testing
    target_key : tuple
        (src_type, relation, dst_type) tuple defining the target relation
        
    Returns
    -------
    tuple
        (train_dict, valid_dict, test_dict, data) containing fold data
    """
    print('init...')
    GB = GraphBuilder(root=relations_root, relnames=relnames, 
                     val_idxs=val_idxs, test_idxs=test_idxs)
    print('building...')
    GB.build()
    print('generating triples...')
    train, valid, test, data = GB.get_triples()
    
    return train, valid, test, data


def compute_graph_metrics(data):
    """
    Compute and return graph statistics.
    
    Parameters
    ----------
    data : HeteroData
        PyTorch Geometric heterogeneous graph data
        
    Returns
    -------
    dict
        Dictionary containing graph metrics
    """
    metrics = {
        'num_nodes': sum(data['num_nodes_dict'].values()),
        'num_edges': sum(v.size(1) for v in data['edge_index_dict'].values()),
        'num_node_types': len(data['node_name_dict']),
        'num_relations': len(data['edge_index_dict']),
        'node_counts': dict(data['num_nodes_dict']),
        'edge_counts': {str(k): v.size(1) for k, v in data['edge_index_dict'].items()}
    }
    return metrics


def print_graph_summary(metrics, fold=None):
    """Print summary of graph statistics."""
    fold_str = f" (Fold {fold})" if fold is not None else ""
    print(f"\nGraph Summary{fold_str}:")
    print("=" * 60)
    print(f"Total nodes: {metrics['num_nodes']:,}")
    print(f"Total edges: {metrics['num_edges']:,}")
    print(f"Node types: {metrics['num_node_types']}")
    print(f"Relation types: {metrics['num_relations']}")
    
    print(f"\nNode counts by type:")
    for node_type, count in metrics['node_counts'].items():
        print(f"  {node_type}: {count:,}")
    
    print(f"\nEdge counts by relation:")
    for relation, count in metrics['edge_counts'].items():
        print(f"  {relation}: {count:,}")
    print("=" * 60)


def main():
    """Main execution function."""
    args = get_args()
    
    print("------------------------------------------------------------------")
    print("kg_construction_10_make_tkg.py")
    print("------------------------------------------------------------------")
    print()
    print("-------------------------------------------------------------------")
    print(args)
    print("-------------------------------------------------------------------")
    print()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup paths
    relations_root = os.path.join(args.extdata, args.relations_root)
    tkg_output_path = os.path.join(args.out, args.tkg_output_dir)
    os.makedirs(tkg_output_path, exist_ok=True)
    os.makedirs(os.path.join(tkg_output_path, 'processed'), exist_ok=True)
    
    print(f"Reading relations from: {relations_root}")
    print(f"Output directory: {tkg_output_path}")
    
    # Get relation files to include
    relnames = get_relation_files(
        relations_root, 
        exclude_patient_relations=args.exclude_patient_relations,
        no_rev=args.no_rev
    )
    
    if not relnames:
        raise ValueError("No valid relation files found!")
    
    # Get number of target DTIs for cross-validation 
    target_file = os.path.join(relations_root, 'targetome_drug_targets_gene.csv')
    if not os.path.exists(target_file):
        # Try alternative naming
        target_file = os.path.join(relations_root, 'targetome_drug_targets_gene_fwd.csv')
    
    if not os.path.exists(target_file):
        raise FileNotFoundError(f"Target relation file not found: {target_file}")
        
    # Read target file to get actual number of valid DTIs after processing
    target_df = pd.read_csv(target_file)
    target_df = target_df.dropna(subset=['src', 'dst', 'src_type', 'dst_type', 'relation'])
    N_dtis = target_df.shape[0]
    print(f"Number of target DTIs (after processing): {N_dtis}")
    
    # Perform K-fold cross-validation
    kfold = KFold(n_splits=args.k_folds, random_state=args.seed, shuffle=True)
    
    # Store metrics for final summary
    all_metrics = []
    
    for fold, (train_idxs, test_idxs) in enumerate(kfold.split(range(N_dtis))):
        print(f"\nProcessing Fold {fold}...")
        
        # Sample validation indices from training set
        val_size = int(args.val_prop * len(train_idxs))  # Use training set size, not total DTI count
        val_idxs = np.random.choice(train_idxs, val_size, replace=False)
        train_idxs = np.array([i for i in train_idxs if i not in val_idxs])
        
        # Validate no overlap between splits
        assert len(set(train_idxs).intersection(set(val_idxs))) == 0, 'train and val overlap'
        assert len(set(train_idxs).intersection(set(test_idxs))) == 0, 'train and test overlap'
        assert len(set(val_idxs).intersection(set(test_idxs))) == 0, 'val and test overlap'
        
        # Build graph data for this fold
        train, valid, test, data = build_fold_data(
            relations_root, relnames, train_idxs, val_idxs, test_idxs
        )
        
        # Create fold output directory
        fold_dir = os.path.join(tkg_output_path, 'processed', f'FOLD_{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # Save fold data
        torch.save(train, os.path.join(fold_dir, 'pos_train.pt'))
        torch.save(valid, os.path.join(fold_dir, 'pos_valid.pt'))
        torch.save(test, os.path.join(fold_dir, 'pos_test.pt'))
        torch.save(data, os.path.join(fold_dir, 'Data.pt'))
        
        # Save deprecated negative files (may be needed for compatibility)
        torch.save(None, os.path.join(fold_dir, 'neg_train.pt'))
        torch.save(None, os.path.join(fold_dir, 'neg_valid.pt'))
        torch.save(None, os.path.join(fold_dir, 'neg_test.pt'))
        
        # Compute and store metrics
        metrics = compute_graph_metrics(data)
        all_metrics.append(metrics)
        
        print(f'Fold {fold} -> # train: {len(train_idxs)}, # val: {len(val_idxs)}, # test: {len(test_idxs)}')
    
    # Print final summary using the last fold's data as representative
    print_graph_summary(all_metrics[-1])
    
    # Print relation types included
    if 'relations' in locals():
        print(f"\nIncluded relation types:")
        unique_relations = set()
        for relname in relnames:
            try:
                rdf = pd.read_csv(os.path.join(relations_root, relname))
                if not rdf.empty:
                    rel_type = f"{rdf.src_type.iloc[0]}->{rdf.relation.iloc[0]}->{rdf.dst_type.iloc[0]}"
                    unique_relations.add(rel_type)
            except:
                continue
        
        for rel_type in sorted(unique_relations):
            print(f"  {rel_type}")
    
    print(f"\nTKG construction completed successfully!")
    print(f"Output saved to: {tkg_output_path}")
    print(f"Generated {args.k_folds} cross-validation folds")


if __name__ == "__main__":
    main()
