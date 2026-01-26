import os
import argparse
import pandas as pd
import numpy as np
import torch


def get_args():
    parser = argparse.ArgumentParser(description='pre-processing for relations between proteins and diseases (CTD)')
    parser.add_argument('--data', type=str, default='../../../data/', help='Path to the input data dir')
    parser.add_argument('--extdata', type=str, default='../../extdata/', help='Path to the extra data dir')
    parser.add_argument('--out', type=str, default='../../output/', help='Path to the output data dir')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    return parser.parse_args()


def load_ctd_genes_diseases(path):
    df = pd.read_csv(path, comment='#', header=None, low_memory=False)
    # From file: GeneSymbol,GeneID,DiseaseName,DiseaseID,DirectEvidence,OmimIDs,PubMedIDs
    df.columns = ['GeneSymbol', 'GeneID', 'DiseaseName', 'DiseaseID', 'DirectEvidence', 'OmimIDs', 'PubMedIDs']
    return df


if __name__ == '__main__':
    print('------------------------------------------------------------------')
    print('kg_construction_04__protein_associates_disease.py')
    print('------------------------------------------------------------------')
    print()

    args = get_args()
    print('-------------------------------------------------------------------')
    print(args)
    print('-------------------------------------------------------------------')
    print()

    os.makedirs(f'{args.out}/meta', exist_ok=True)
    os.makedirs(f'{args.out}/relations', exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ctd_path = os.path.join(args.data, 'CTD_curated_genes_diseases.csv')
    protdis = load_ctd_genes_diseases(ctd_path)

    disease_space_path = f'{args.out}/meta/disease_space.txt'
    if os.path.exists(disease_space_path):
        disease_space = np.loadtxt(disease_space_path, dtype=str)
        protdis = protdis[lambda x: x.DiseaseID.isin(disease_space)]

    protdis.to_csv(f'{args.out}/meta/CTD__genes_diseases.csv', index=False)

    protdis_fwd = (
        protdis[['GeneSymbol', 'DiseaseID']]
        .drop_duplicates()
        .rename(columns={'GeneSymbol': 'src', 'DiseaseID': 'dst'})
        .assign(src_type='gene', dst_type='disease', relation='associates_fwd')
    )
    protdis_rev = (
        protdis[['DiseaseID', 'GeneSymbol']]
        .drop_duplicates()
        .rename(columns={'DiseaseID': 'src', 'GeneSymbol': 'dst'})
        .assign(src_type='disease', dst_type='gene', relation='associates_rev')
    )

    protdis_fwd.to_csv(f'{args.out}/relations/ctd_genes_diseases_fwd.csv', index=False)
    protdis_rev.to_csv(f'{args.out}/relations/ctd_genes_diseases_rev.csv', index=False)

    print('summary:')
    print('-' * 100)
    print(f"number of unique diseases: {protdis.DiseaseID.nunique()}")
    print(f"number of unique genes: {protdis.GeneID.nunique()}")
    print('example rows:')
    print(protdis.head(3))
    print('-' * 100)

    print(f'saved to: {args.out}')
    print()


