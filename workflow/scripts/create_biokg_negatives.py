#!/usr/bin/env python3
"""
Generate negative samples for BioKG evaluation following OGB protocol.

This script creates negative samples by corrupting head and tail entities
of positive edges while ensuring type consistency (e.g., only corrupt 
drug nodes with other drug nodes).
"""

import argparse
import os
import numpy as np
import torch
import pandas as pd
from collections import defaultdict
import random
from typing import Dict, Set, Tuple, List


def get_args():
    parser = argparse.ArgumentParser(description="Generate negative samples for BioKG evaluation")
    
    parser.add_argument("--data_root", type=str, required=True,
                       help="Path to the BioKG data root directory")
    parser.add_argument("--target_relation", type=str, default="drug,drug-protein,protein",
                       help="Target relation for negative sampling")
    parser.add_argument("--n_neg_per_pos", type=int, default=500,
                       help="Number of negative samples per positive edge")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (defaults to data_root/processed)")
    parser.add_argument("--corruption_mode", type=str, default="both", 
                       choices=["head", "tail", "both"],
                       help="Which entities to corrupt: head, tail, or both")
    
    return parser.parse_args()


def parse_relation(target_relation: str) -> Tuple[str, str, str]:
    """Parse target relation string into (head_type, relation, tail_type)"""
    parts = target_relation.split(',')
    if len(parts) != 3:
        raise ValueError(f"Target relation must have format 'head,relation,tail', got: {target_relation}")
    return tuple(parts)


def load_biokg_data(data_root: str) -> Tuple[Dict, Dict, Dict, Dict]:
    """Load BioKG data and splits"""
    processed_dir = os.path.join(data_root, "processed")
    
    # Load data
    data = torch.load(os.path.join(processed_dir, "Data.pt"))
    pos_train = torch.load(os.path.join(processed_dir, "pos_train.pt"))
    pos_valid = torch.load(os.path.join(processed_dir, "pos_valid.pt"))
    pos_test = torch.load(os.path.join(processed_dir, "pos_test.pt"))
    
    return data, pos_train, pos_valid, pos_test


def get_existing_edges(pos_train: Dict, pos_valid: Dict, pos_test: Dict, 
                      target_relint: int) -> Set[Tuple[int, int]]:
    """Get all existing positive edges for the target relation"""
    existing_edges = set()
    
    for split_data in [pos_train, pos_valid, pos_test]:
        # Filter for target relation
        mask = split_data['relation'] == target_relint
        heads = split_data['head'][mask]
        tails = split_data['tail'][mask]
        
        for h, t in zip(heads, tails):
            existing_edges.add((h.item(), t.item()))
    
    return existing_edges


def sample_negative_edges(positive_edges: List[Tuple[int, int]], 
                         head_nodes: List[int], tail_nodes: List[int],
                         existing_edges: Set[Tuple[int, int]],
                         n_neg_per_pos: int, corruption_mode: str,
                         seed: int) -> List[Tuple[int, int]]:
    """
    Efficiently sample negative edges by corrupting positive edges using vectorized operations.
    
    Args:
        positive_edges: List of (head, tail) positive edges
        head_nodes: List of all possible head nodes (for corruption)
        tail_nodes: List of all possible tail nodes (for corruption)
        existing_edges: Set of all existing positive edges to avoid
        n_neg_per_pos: Number of negative samples per positive edge
        corruption_mode: 'head', 'tail', or 'both'
        seed: Random seed
    
    Returns:
        List of negative edges as (head, tail) tuples
    """
    np.random.seed(seed)
    random.seed(seed)
    
    head_nodes = np.array(head_nodes)
    tail_nodes = np.array(tail_nodes)
    
    negative_edges = []
    
    # Process edges in batches for better efficiency
    batch_size = min(100, len(positive_edges))  # Process in batches to avoid memory issues
    
    for i in range(0, len(positive_edges), batch_size):
        batch_edges = positive_edges[i:i + batch_size]
        batch_negatives = _sample_batch_negatives(
            batch_edges, head_nodes, tail_nodes, existing_edges,
            n_neg_per_pos, corruption_mode
        )
        negative_edges.extend(batch_negatives)
    
    return negative_edges


def _sample_batch_negatives(batch_edges: List[Tuple[int, int]],
                           head_nodes: np.ndarray, tail_nodes: np.ndarray,
                           existing_edges: Set[Tuple[int, int]],
                           n_neg_per_pos: int, corruption_mode: str) -> List[Tuple[int, int]]:
    """
    Efficiently sample negatives for a batch of positive edges using vectorized operations.
    """
    batch_negatives = []
    
    for pos_head, pos_tail in batch_edges:
        # Generate candidate negative edges efficiently
        if corruption_mode == "head":
            candidates = _generate_head_corrupted_candidates(pos_head, pos_tail, head_nodes, n_neg_per_pos)
        elif corruption_mode == "tail":
            candidates = _generate_tail_corrupted_candidates(pos_head, pos_tail, tail_nodes, n_neg_per_pos)
        else:  # corruption_mode == "both"
            # Split negatives between head and tail corruption
            n_head = n_neg_per_pos // 2
            n_tail = n_neg_per_pos - n_head
            
            head_candidates = _generate_head_corrupted_candidates(pos_head, pos_tail, head_nodes, n_head)
            tail_candidates = _generate_tail_corrupted_candidates(pos_head, pos_tail, tail_nodes, n_tail)
            candidates = head_candidates + tail_candidates
        
        # Filter out existing positive edges efficiently
        valid_negatives = [edge for edge in candidates if edge not in existing_edges]
        
        # If we don't have enough negatives, generate more
        if len(valid_negatives) < n_neg_per_pos:
            additional_needed = n_neg_per_pos - len(valid_negatives)
            additional_candidates = _generate_additional_candidates(
                pos_head, pos_tail, head_nodes, tail_nodes, existing_edges,
                additional_needed, corruption_mode
            )
            valid_negatives.extend(additional_candidates)
        
        # Take exactly n_neg_per_pos negatives
        batch_negatives.extend(valid_negatives[:n_neg_per_pos])
        
        if len(valid_negatives) < n_neg_per_pos:
            print(f"Warning: Could only generate {len(valid_negatives)}/{n_neg_per_pos} negatives for edge ({pos_head}, {pos_tail})")
    
    return batch_negatives


def _generate_head_corrupted_candidates(pos_head: int, pos_tail: int, 
                                       head_nodes: np.ndarray, n_samples: int) -> List[Tuple[int, int]]:
    """Generate candidates by corrupting the head node."""
    if n_samples == 0:
        return []
    
    # Sample with replacement initially, then remove duplicates
    sample_size = min(n_samples * 2, len(head_nodes))  # Oversample to account for duplicates
    corrupted_heads = np.random.choice(head_nodes, size=sample_size, replace=True)
    
    # Remove duplicates and the original head
    unique_heads = np.unique(corrupted_heads)
    unique_heads = unique_heads[unique_heads != pos_head]
    
    # If we need more samples and haven't exhausted all possibilities
    if len(unique_heads) < n_samples and len(unique_heads) < len(head_nodes) - 1:
        # Sample without replacement from remaining nodes
        remaining_heads = head_nodes[head_nodes != pos_head]
        if len(remaining_heads) > 0:
            additional_size = min(n_samples, len(remaining_heads))
            unique_heads = np.random.choice(remaining_heads, size=additional_size, replace=False)
    
    # Create candidate edges
    candidates = [(int(head), pos_tail) for head in unique_heads[:n_samples]]
    return candidates


def _generate_tail_corrupted_candidates(pos_head: int, pos_tail: int,
                                       tail_nodes: np.ndarray, n_samples: int) -> List[Tuple[int, int]]:
    """Generate candidates by corrupting the tail node."""
    if n_samples == 0:
        return []
    
    # Sample with replacement initially, then remove duplicates
    sample_size = min(n_samples * 2, len(tail_nodes))  # Oversample to account for duplicates
    corrupted_tails = np.random.choice(tail_nodes, size=sample_size, replace=True)
    
    # Remove duplicates and the original tail
    unique_tails = np.unique(corrupted_tails)
    unique_tails = unique_tails[unique_tails != pos_tail]
    
    # If we need more samples and haven't exhausted all possibilities
    if len(unique_tails) < n_samples and len(unique_tails) < len(tail_nodes) - 1:
        # Sample without replacement from remaining nodes
        remaining_tails = tail_nodes[tail_nodes != pos_tail]
        if len(remaining_tails) > 0:
            additional_size = min(n_samples, len(remaining_tails))
            unique_tails = np.random.choice(remaining_tails, size=additional_size, replace=False)
    
    # Create candidate edges
    candidates = [(pos_head, int(tail)) for tail in unique_tails[:n_samples]]
    return candidates


def _generate_additional_candidates(pos_head: int, pos_tail: int,
                                   head_nodes: np.ndarray, tail_nodes: np.ndarray,
                                   existing_edges: Set[Tuple[int, int]],
                                   n_needed: int, corruption_mode: str) -> List[Tuple[int, int]]:
    """Generate additional candidates when initial sampling didn't produce enough valid negatives."""
    additional_candidates = []
    max_attempts = n_needed * 5  # Limit attempts to avoid infinite loops
    attempts = 0
    
    while len(additional_candidates) < n_needed and attempts < max_attempts:
        attempts += 1
        
        if corruption_mode == "head" or (corruption_mode == "both" and np.random.random() < 0.5):
            # Corrupt head
            candidate_head = np.random.choice(head_nodes)
            if candidate_head != pos_head:
                candidate = (int(candidate_head), pos_tail)
                if candidate not in existing_edges and candidate not in additional_candidates:
                    additional_candidates.append(candidate)
        else:
            # Corrupt tail
            candidate_tail = np.random.choice(tail_nodes)
            if candidate_tail != pos_tail:
                candidate = (pos_head, int(candidate_tail))
                if candidate not in existing_edges and candidate not in additional_candidates:
                    additional_candidates.append(candidate)
    
    return additional_candidates


def create_negative_samples(data: Dict, pos_valid: Dict, pos_test: Dict,
                          pos_train: Dict, target_relation: str, 
                          n_neg_per_pos: int, corruption_mode: str, 
                          seed: int) -> Tuple[Dict, Dict]:
    """Create negative samples for validation and test sets"""
    
    head_type, relation, tail_type = parse_relation(target_relation)
    
    # Get relation integer mapping
    rel2int = {k: v[0] for k, v in data.edge_reltype.items()}
    # Convert target_relation string to tuple format that matches edge_reltype keys
    target_relation_tuple = (head_type, relation, tail_type)
    target_relint = rel2int[target_relation_tuple]
    if not isinstance(target_relint, int):
        target_relint = target_relint.item()
    
    print(f"Creating negative samples for relation: {target_relation} -> {target_relation_tuple} (id: {target_relint})")
    print(f"Head type: {head_type}, Tail type: {tail_type}")
    
    # Get all nodes of the relevant types
    head_nodes = list(range(data.num_nodes_dict[head_type]))
    tail_nodes = list(range(data.num_nodes_dict[tail_type]))
    
    print(f"Number of {head_type} nodes: {len(head_nodes)}")
    print(f"Number of {tail_type} nodes: {len(tail_nodes)}")
    
    # Get all existing positive edges to avoid sampling them as negatives
    existing_edges = get_existing_edges(pos_train, pos_valid, pos_test, target_relint)
    print(f"Total existing positive edges: {len(existing_edges)}")
    
    # Create negative samples for validation set
    print("Creating negative samples for validation set...")
    valid_mask = pos_valid['relation'] == target_relint
    valid_positive_edges = [(h.item(), t.item()) for h, t in 
                           zip(pos_valid['head'][valid_mask], pos_valid['tail'][valid_mask])]
    
    print(f"Number of positive validation edges: {len(valid_positive_edges)}")
    
    valid_negative_edges = sample_negative_edges(
        valid_positive_edges, head_nodes, tail_nodes, existing_edges,
        n_neg_per_pos, corruption_mode, seed
    )
    
    # Create negative samples for test set
    print("Creating negative samples for test set...")
    test_mask = pos_test['relation'] == target_relint
    test_positive_edges = [(h.item(), t.item()) for h, t in 
                          zip(pos_test['head'][test_mask], pos_test['tail'][test_mask])]
    
    print(f"Number of positive test edges: {len(test_positive_edges)}")
    
    test_negative_edges = sample_negative_edges(
        test_positive_edges, head_nodes, tail_nodes, existing_edges,
        n_neg_per_pos, corruption_mode, seed + 1  # Different seed for test
    )
    
    print(f"Generated {len(valid_negative_edges)} validation negatives")
    print(f"Generated {len(test_negative_edges)} test negatives")
    
    # Convert to the same format as positive edges - ensure proper tensor types
    if len(valid_negative_edges) > 0:
        valid_heads = [int(edge[0]) for edge in valid_negative_edges]
        valid_tails = [int(edge[1]) for edge in valid_negative_edges]
    else:
        valid_heads = []
        valid_tails = []
    
    if len(test_negative_edges) > 0:
        test_heads = [int(edge[0]) for edge in test_negative_edges]
        test_tails = [int(edge[1]) for edge in test_negative_edges]
    else:
        test_heads = []
        test_tails = []
    
    neg_valid = {
        'head': torch.tensor(valid_heads, dtype=torch.long),
        'tail': torch.tensor(valid_tails, dtype=torch.long),
        'relation': torch.tensor([int(target_relint)] * len(valid_negative_edges), dtype=torch.long)
    }
    
    neg_test = {
        'head': torch.tensor(test_heads, dtype=torch.long),
        'tail': torch.tensor(test_tails, dtype=torch.long),
        'relation': torch.tensor([int(target_relint)] * len(test_negative_edges), dtype=torch.long)
    }
    
    return neg_valid, neg_test


def save_negative_samples(neg_valid: Dict, neg_test: Dict, output_dir: str):
    """Save negative samples to disk"""
    os.makedirs(output_dir, exist_ok=True)
    
    valid_path = os.path.join(output_dir, "neg_valid.pt")
    test_path = os.path.join(output_dir, "neg_test.pt")
    
    torch.save(neg_valid, valid_path)
    torch.save(neg_test, test_path)
    
    print(f"Saved validation negatives to: {valid_path}")
    print(f"Saved test negatives to: {test_path}")
    
    # Also save in extdata for convenience
    extdata_dir = os.path.join(os.path.dirname(output_dir), "..", "..", "extdata")
    if os.path.exists(extdata_dir):
        extdata_valid_path = os.path.join(extdata_dir, "biokg_negative_valid.pt")
        extdata_test_path = os.path.join(extdata_dir, "biokg_negative_test.pt")
        
        torch.save(neg_valid, extdata_valid_path)
        torch.save(neg_test, extdata_test_path)
        
        print(f"Also saved to extdata: {extdata_valid_path}")
        print(f"Also saved to extdata: {extdata_test_path}")


def main():
    args = get_args()
    
    print(f"Loading BioKG data from: {args.data_root}")
    data, pos_train, pos_valid, pos_test = load_biokg_data(args.data_root)
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else os.path.join(args.data_root, "processed")
    
    print(f"Generating negative samples with the following parameters:")
    print(f"  Target relation: {args.target_relation}")
    print(f"  Negatives per positive: {args.n_neg_per_pos}")
    print(f"  Corruption mode: {args.corruption_mode}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output directory: {output_dir}")
    
    # Create negative samples
    neg_valid, neg_test = create_negative_samples(
        data, pos_valid, pos_test, pos_train,
        args.target_relation, args.n_neg_per_pos, 
        args.corruption_mode, args.seed
    )
    
    # Save negative samples
    save_negative_samples(neg_valid, neg_test, output_dir)
    
    print("Negative sampling completed successfully!")


if __name__ == "__main__":
    main()
