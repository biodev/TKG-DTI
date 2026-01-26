import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.stats import rankdata

def evaluate(df, ece_bins=10, partition='test', verbose=True, method='all'):
    """
    Evaluate model performance metrics on the given dataframe.

    Parameters:
    df (pd.DataFrame): DataFrame containing the predictions and ground truth.
    ece_bins (int): Number of bins for Expected Calibration Error calculation.
    partition (str): Which partition to evaluate ('test', 'valid', etc.).
    verbose (bool): Whether to print progress information.
    method (str): Evaluation method - 'all' for all-vs-all evaluation, 
                 'negatives' for evaluation using pre-sampled negatives.

    Returns:
    dict: Dictionary containing evaluation metrics.
    """
    if method == 'all':
        return _evaluate_all(df, ece_bins, partition, verbose)
    elif method == 'negatives':
        return _evaluate_with_negatives(df, ece_bins, partition, verbose)
    else:
        raise ValueError(f"Unknown evaluation method: {method}. Use 'all' or 'negatives'.")


def _evaluate_all(df, ece_bins=10, partition='test', verbose=True):
    """
    Original evaluation method that compares against all possible negatives.
    This method becomes intractable for large datasets.
    """
    ranks = []
    aurocs = []
    auprcs = []
    brier_scores = []

    # Unique drugs in the dataframe
    unique_drugs = df['drug'].unique()

    for i,drug in enumerate(unique_drugs):
        if verbose: print(f'[evaluating {partition} set, progress: {i}/{len(unique_drugs)}]', end='\r')
        # Filter dataframe for the current drug and relevant partitions
        tmp = df[(df['drug'] == drug) & (df[partition] | df['negatives'])]

        # Skip if there are no positive test samples
        if tmp[partition].astype(float).sum() < 1:
            continue

        y_true = tmp[partition].astype(float).values
        y_prob = tmp['prob'].values

        # Calculate AUROC and AUPRC
        aurocs.append(roc_auc_score(y_true, y_prob))
        auprcs.append(average_precision_score(y_true, y_prob))

        # Calculate Brier score
        brier_scores.append(brier_score_loss(y_true, y_prob))

        # Calculate rank of the each positive sample; only compare to negatives not to other positives
        # probably a more clever way to do this...
        for i,row in tmp[lambda x: x[partition]].iterrows():
            neg_scores = tmp[lambda x: x.negatives].score
            rank = (neg_scores >= row.score).sum() + 1
            ranks.append(rank)

    ranks = np.array(ranks)
    metrics = {
        'MRR': np.mean(1 / ranks),
        'Top1': np.mean(ranks == 1),
        'Top3': np.mean(ranks <= 3),
        'Top10': np.mean(ranks <= 10),
        'Top100': np.mean(ranks <= 100),
        'avg_AUC': np.mean(aurocs),
        'avg_AP': np.mean(auprcs),
        'avg_Brier': np.mean(brier_scores)
    }

    # Calculate Expected Calibration Error (ECE)
    y_true_all = df.loc[df[partition] | df['negatives'], partition].astype(float).values
    y_prob_all = df.loc[df[partition] | df['negatives'], 'prob'].values

    bin_edges = np.linspace(0.0, 1.0, ece_bins + 1)
    bin_indices = np.digitize(y_prob_all, bins=bin_edges, right=False)

    ece = 0.0
    for i in range(ece_bins):
        bin_mask = bin_indices == i + 1
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            bin_accuracy = np.mean(y_true_all[bin_mask])
            bin_confidence = np.mean(y_prob_all[bin_mask])
            ece += (bin_count / len(y_prob_all)) * np.abs(bin_confidence - bin_accuracy)
    metrics['ECE'] = ece

    return metrics


def _evaluate_with_negatives(df, ece_bins=10, partition='test', verbose=True):
    """
    Scalable evaluation method using pre-sampled negative edges.
    
    This method expects the DataFrame to contain only:
    1. Positive edges for the specified partition
    2. Pre-sampled negative edges
    
    This is much more tractable for large datasets like BioKG.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing positive edges and pre-sampled negatives.
                      Expected columns: 'drug', 'protein', 'score', 'prob', partition column,
                      and a column indicating if edge is a negative sample.
    """
    if 'negative_sample' not in df.columns:
        raise ValueError("DataFrame must contain 'negative_sample' column for negatives evaluation method")
    
    ranks = []
    aurocs = []
    auprcs = []
    brier_scores = []
    
    # Get positive edges for the specified partition
    positive_edges = df[df[partition] == True]
    negative_edges = df[df['negative_sample'] == True]
    
    if len(positive_edges) == 0:
        raise ValueError(f"No positive edges found for partition '{partition}'")
    
    if len(negative_edges) == 0:
        raise ValueError("No negative samples found in DataFrame")
    
    # Group by drug to evaluate each drug separately (following OGB protocol)
    unique_drugs = positive_edges['drug'].unique()
    
    for i, drug in enumerate(unique_drugs):
        if verbose: 
            print(f'[evaluating {partition} set with negatives, progress: {i}/{len(unique_drugs)}]', end='\r')
        
        # Get positive edges for this drug
        drug_positives = positive_edges[positive_edges['drug'] == drug]
        
        # Get negative edges for this drug (could be all negatives or drug-specific)
        drug_negatives = negative_edges[negative_edges['drug'] == drug]
        
        # If no drug-specific negatives, use all negatives (head corruption)
        if len(drug_negatives) == 0:
            drug_negatives = negative_edges
        
        if len(drug_positives) == 0:
            continue
            
        # Combine positives and negatives for this drug
        drug_data = pd.concat([drug_positives, drug_negatives], ignore_index=True)
        
        # Create labels: 1 for positives, 0 for negatives
        y_true = drug_data[partition].astype(float).fillna(0).values
        y_prob = drug_data['prob'].values
        
        # Calculate AUROC and AUPRC if we have both positive and negative samples
        if len(np.unique(y_true)) > 1:
            aurocs.append(roc_auc_score(y_true, y_prob))
            auprcs.append(average_precision_score(y_true, y_prob))
            brier_scores.append(brier_score_loss(y_true, y_prob))
        
        # Calculate ranking metrics for each positive edge
        for _, pos_edge in drug_positives.iterrows():
            pos_score = pos_edge['score']
            neg_scores = drug_negatives['score'].values
            
            # Rank = number of negatives with score >= positive score + 1
            rank = (neg_scores >= pos_score).sum() + 1
            ranks.append(rank)
    
    ranks = np.array(ranks)
    
    # Calculate metrics
    metrics = {
        'MRR': np.mean(1 / ranks) if len(ranks) > 0 else 0.0,
        'Top1': np.mean(ranks == 1) if len(ranks) > 0 else 0.0,
        'Top3': np.mean(ranks <= 3) if len(ranks) > 0 else 0.0,
        'Top10': np.mean(ranks <= 10) if len(ranks) > 0 else 0.0,
        'Top100': np.mean(ranks <= 100) if len(ranks) > 0 else 0.0,
        'avg_AUC': np.mean(aurocs) if len(aurocs) > 0 else 0.0,
        'avg_AP': np.mean(auprcs) if len(auprcs) > 0 else 0.0,
        'avg_Brier': np.mean(brier_scores) if len(brier_scores) > 0 else 1.0
    }
    
    # Calculate Expected Calibration Error (ECE) on all data
    all_data = df[df[partition] | df['negative_sample']]
    y_true_all = all_data[partition].astype(float).fillna(0).values
    y_prob_all = all_data['prob'].values
    
    bin_edges = np.linspace(0.0, 1.0, ece_bins + 1)
    bin_indices = np.digitize(y_prob_all, bins=bin_edges, right=False)
    
    ece = 0.0
    for i in range(ece_bins):
        bin_mask = bin_indices == i + 1
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            bin_accuracy = np.mean(y_true_all[bin_mask])
            bin_confidence = np.mean(y_prob_all[bin_mask])
            ece += (bin_count / len(y_prob_all)) * np.abs(bin_confidence - bin_accuracy)
    
    metrics['ECE'] = ece
    
    return metrics


def evaluate_with_negatives(df, ece_bins=10, partition='test', verbose=True):
    """
    Wrapper function for backwards compatibility.
    Use evaluate(df, method='negatives') instead.
    """
    return _evaluate_with_negatives(df, ece_bins, partition, verbose)