import os
import argparse
import pandas as pd
import numpy as np
import torch


def get_args():
    parser = argparse.ArgumentParser(description='pre-processing for relations: protein isin pathway (CTD)')
    parser.add_argument('--data', type=str, default='../../../data/', help='Path to the input data dir')
    parser.add_argument('--extdata', type=str, default='../../extdata/', help='Path to the extra data dir')
    parser.add_argument('--out', type=str, default='../../output/', help='Path to the output data dir')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    return parser.parse_args()


def load_ctd_genes_pathways(path):
    df = pd.read_csv(path, comment='#', header=None, low_memory=False)
    df.columns = ['GeneSymbol', 'GeneID', 'PathwayName', 'PathwayID']
    return df


if __name__ == '__main__':
    print('------------------------------------------------------------------')
    print('kg_construction_06__protein_isin_pathway.py')
    print('------------------------------------------------------------------')
    print()

    args = get_args()
    print('-------------------------------------------------------------------')
    print(args)
    print('-------------------------------------------------------------------')
    print()

    os.makedirs(f'{args.out}/relations', exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ctd_path = os.path.join(args.data, 'CTD_genes_pathways.csv')
    genepath = load_ctd_genes_pathways(ctd_path)

    print('# unique genes:', genepath.GeneSymbol.nunique())
    print('# unique pathways:', genepath.PathwayName.nunique())
    print('# edges:', genepath.shape[0])

    gp_fwd = genepath[['GeneSymbol', 'PathwayID']].rename({'GeneSymbol': 'src', 'PathwayID': 'dst'}, axis=1).assign(src_type='gene', dst_type='pathway', relation='isin_fwd').drop_duplicates()
    gp_rev = genepath[['PathwayID', 'GeneSymbol']].rename({'PathwayID': 'src', 'GeneSymbol': 'dst'}, axis=1).assign(src_type='pathway', dst_type='gene', relation='isin_rev').drop_duplicates()

    gp_fwd.to_csv(f'{args.out}/relations/ctd_gene_isin_pathway_fwd.csv', index=False)
    gp_rev.to_csv(f'{args.out}/relations/ctd_pathway_isin_gene_rev.csv', index=False)

    print('summary:')
    print('-' * 100)
    print(f"number of unique genes: {genepath.GeneSymbol.nunique()}")
    print(f"number of unique pathways: {genepath.PathwayName.nunique()}")
    print(f"number of edges: {genepath.shape[0]}")
    print('example rows:')
    print(genepath.head(3))
    print('-' * 100)

    print(f'saved to: {args.out}')
    print()


