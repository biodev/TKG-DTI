import os
import argparse
import numpy as np
import torch
import omnipath as op
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description='pre-processing for relations between proteins (PPI via OmniPath)')
    parser.add_argument('--data', type=str, default='../../../data/', help='Path to the input data dir')
    parser.add_argument('--extdata', type=str, default='../../extdata/', help='Path to the extra data dir')
    parser.add_argument('--out', type=str, default='../../output/', help='Path to the output data dir')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    return parser.parse_args()


if __name__ == '__main__':
    print('------------------------------------------------------------------')
    print('kg_construction_05__protein_interacts_protein.py')
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

    ppi = op.interactions.OmniPath().get(genesymbols=True)

    # Map to gene symbols directly from OmniPath output
    cols = ppi.columns.tolist()
    if 'source_genesymbol' in cols and 'target_genesymbol' in cols:
        ppi['source_gene'] = ppi['source_genesymbol']
        ppi['target_gene'] = ppi['target_genesymbol']
    elif 'source_genesymbols' in cols and 'target_genesymbols' in cols:
        # take first symbol if list-like; otherwise passthrough
        def first_symbol(val):
            if isinstance(val, (list, tuple)) and len(val) > 0:
                return val[0]
            if isinstance(val, str) and '|' in val:
                return val.split('|')[0]
            return val
        ppi['source_gene'] = ppi['source_genesymbols'].apply(first_symbol)
        ppi['target_gene'] = ppi['target_genesymbols'].apply(first_symbol)
    else:
        # fallback: assume source/target are already symbols
        ppi['source_gene'] = ppi['source']
        ppi['target_gene'] = ppi['target']

    protspace = np.unique(ppi['source_gene'].astype(str).values.tolist() + ppi['target_gene'].astype(str).values.tolist())
    ppi.to_csv(f'{args.out}/meta/omnipath__protein_interacts_protein.csv', index=False)

    ppi_fwd = ppi[['source_gene', 'target_gene', 'consensus_inhibition', 'consensus_stimulation']].rename({'source_gene': 'src', 'target_gene': 'dst'}, axis=1).assign(src_type='gene', dst_type='gene')
    ppi_stim_fwd = ppi_fwd[ppi_fwd['consensus_stimulation']].assign(relation='stimulates_fwd').drop(['consensus_inhibition', 'consensus_stimulation'], axis=1)
    ppi_inhib_fwd = ppi_fwd[ppi_fwd['consensus_inhibition']].assign(relation='inhibits_fwd').drop(['consensus_inhibition', 'consensus_stimulation'], axis=1)
    ppi_other_fwd = ppi_fwd[~ppi_fwd['consensus_inhibition'] & ~ppi_fwd['consensus_stimulation']].assign(relation='interacts_fwd').drop(['consensus_inhibition', 'consensus_stimulation'], axis=1)

    ppi_rev = ppi_fwd.rename({'src': 'dst', 'dst': 'src'}, axis=1)
    ppi_stim_rev = ppi_rev[ppi_rev['consensus_stimulation']].assign(relation='stimulates_rev').drop(['consensus_inhibition', 'consensus_stimulation'], axis=1)
    ppi_inhib_rev = ppi_rev[ppi_rev['consensus_inhibition']].assign(relation='inhibits_rev').drop(['consensus_inhibition', 'consensus_stimulation'], axis=1)
    ppi_other_rev = ppi_rev[~ppi_rev['consensus_inhibition'] & ~ppi_rev['consensus_stimulation']].assign(relation='interacts_rev').drop(['consensus_inhibition', 'consensus_stimulation'], axis=1)

    ppi_stim_fwd.to_csv(f'{args.out}/relations/omnipath_ppi_stim_fwd.csv', index=False)
    ppi_inhib_fwd.to_csv(f'{args.out}/relations/omnipath_ppi_inhib_fwd.csv', index=False)
    ppi_stim_rev.to_csv(f'{args.out}/relations/omnipath_ppi_stim_rev.csv', index=False)
    ppi_inhib_rev.to_csv(f'{args.out}/relations/omnipath_ppi_inhib_rev.csv', index=False)
    ppi_other_fwd.to_csv(f'{args.out}/relations/omnipath_ppi_other_fwd.csv', index=False)
    ppi_other_rev.to_csv(f'{args.out}/relations/omnipath_ppi_other_rev.csv', index=False)

    print('summary:')
    print('-' * 100)
    print(f"# of unique proteins (by symbol): {len(protspace)}")
    print(f"# of edges (all): {ppi.shape[0]}")
    print('example rows:')
    print(ppi.head(3))
    print('-' * 100)

    print(f'saved to: {args.out}')
    print()


