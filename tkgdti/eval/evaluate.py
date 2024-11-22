import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.stats import rankdata

def evaluate(df, ece_bins=10, partition='test'):
    """
    Evaluate model performance metrics on the given dataframe.

    Parameters:
    df (pd.DataFrame): DataFrame containing the predictions and ground truth.

    Returns:
    dict: Dictionary containing evaluation metrics.
    """
    ranks = []
    aurocs = []
    auprcs = []
    brier_scores = []

    # Unique drugs in the dataframe
    unique_drugs = df['drug_name'].unique()

    for i,drug in enumerate(unique_drugs):
        print(f'evaluating {partition} set, progress: {i}/{len(unique_drugs)}', end='\r')
        # Filter dataframe for the current drug and relevant partitions
        tmp = df[(df['drug_name'] == drug) & (df[partition] | df['negatives'])]

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

        # Calculate ranks using scipy's rankdata function
        # Higher probabilities get lower ranks (rank 1 is the highest probability)
        tmp_ranks = rankdata(-y_prob, method='min')

        # Extract ranks of positive samples
        pos_indices = np.where(y_true == 1)[0]
        ranks.extend(tmp_ranks[pos_indices])

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