import os
import argparse
import pandas as pd
import numpy as np
import h5py
import gc


VALID_CELL_LINES = ['MCF7', 'SKB', 'PC3', 'VCAP', 'PHH', 'HT29', 'HA1E', 'A375'] # removed A549 [see ablation study results]


def get_args():
    parser = argparse.ArgumentParser(description='pre-processing for relations: LINCS drug perturbed expression')
    parser.add_argument('--data', type=str, default='../../../data/', help='Path to the input data dir')
    parser.add_argument('--extdata', type=str, default='../../extdata/', help='Path to the extra data dir')
    parser.add_argument('--out', type=str, default='../../output/', help='Path to the output data dir')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--score_threshold', type=int, default=8, help='Score threshold for perturbation')
    parser.add_argument('--sig_sign', type=str, default='-', help='Sign of perturbation score [options: "-", "+", "+/-"]')
    return parser.parse_args()


if __name__ == '__main__':
    print('------------------------------------------------------------------')
    print('kg_construction_07__lincs_drug_perturbed_expression.py')
    print('------------------------------------------------------------------')
    print()

    args = get_args()
    print('-------------------------------------------------------------------')
    print(args)
    print('-------------------------------------------------------------------')
    print()

    os.makedirs(f'{args.out}/relations', exist_ok=True)
    os.makedirs(f'{args.out}/meta', exist_ok=True)

    print('loading siginfo...')
    siginfo = pd.read_csv(f'{args.data}/siginfo_beta.txt', sep='\t', low_memory=False)
    print(f'  siginfo rows: {siginfo.shape[0]}')

    print('loading compound metadata...')
    # CRITICAL FIX: Always filter to original targetome drugs (89 drugs)
    targetome_meta_path = f'{args.out}/meta/targetome__drug_targets_gene.csv'
    if os.path.exists(targetome_meta_path):
        targetome_drugs = pd.read_csv(targetome_meta_path)
        valid_inchikeys = set(targetome_drugs['inchikey'].dropna())
        print(f'Original targetome drugs: {len(valid_inchikeys)}')
    else:
        # Fallback to drugspace.txt if available
        drugspace = np.loadtxt(f'{args.out}/meta/drugspace.txt', dtype=str) if os.path.exists(f'{args.out}/meta/drugspace.txt') else None
        if drugspace is not None:
            valid_inchikeys = set(drugspace)
            print(f'Using drugspace.txt: {len(valid_inchikeys)} drugs')
        else:
            raise FileNotFoundError(f"Neither {targetome_meta_path} nor drugspace.txt found. Cannot filter LINCS compounds.")
    
    druginfo = pd.read_csv(f'{args.data}/compoundinfo_beta.txt', sep='\t', low_memory=False)
    before = druginfo.shape[0]
    druginfo = druginfo[druginfo.inchi_key.isin(valid_inchikeys)]
    print(f'  filtered LINCS compounds to targetome drugs: {before} -> {druginfo.shape[0]}')
    pertid_space = druginfo.pert_id.unique()

    print('filtering siginfo to 10uM/24h exemplar signatures in valid cell lines...')
    before = siginfo.shape[0]
    siginfo = siginfo[lambda x: x.pert_id.isin(pertid_space) & (x.pert_dose == 10.0) & (x.pert_time == 24.0)]
    siginfo = siginfo[lambda x: x.is_exemplar_sig == 1]
    siginfo = siginfo[lambda x: x.cell_iname.isin(VALID_CELL_LINES)]
    siginfo = siginfo.groupby(['cell_iname', 'pert_id']).first().reset_index()
    print(f'  siginfo rows after filtering/grouping: {before} -> {siginfo.shape[0]}')

    print('opening GCTX file (level5) and reading metadata...')
    hdf = h5py.File(f'{args.data}/level5_beta_trt_cp_n720216x12328.gctx')
    hdf_ids = hdf['0']['META']['COL']['id'][...].astype(str)
    hdf_genes = hdf['0']['META']['ROW']['id'][...].astype(str)
    print(f'  signatures in GCTX: {len(hdf_ids)} ; genes: {len(hdf_genes)}')

    print('loading gene metadata...')
    geneinfo = pd.read_csv(f'{args.data}/geneinfo_beta.txt', sep='\t', low_memory=False)
    geneinfo.gene_id = geneinfo.gene_id.astype(str)

    print('indexing signatures against GCTX...')
    sig2hdfix = {sig: i for i, sig in enumerate(hdf_ids)}
    idxs = [sig2hdfix[sig] for sig in siginfo.sig_id.values if sig in sig2hdfix]
    missing = len(siginfo) - len(idxs)
    if missing:
        print(f'  missing signatures not found in GCTX: {missing}')
    sorting_ixs = np.argsort(idxs)
    sorted_idxs = np.array(idxs)[sorting_ixs]
    sorted_sigids = siginfo.sig_id.values[sorting_ixs]

    # Prefilter features to reduce memory before touching the matrix
    print('prefiltering feature space to protein-coding and landmark/best-inferred...')
    feature_filtered = geneinfo[lambda x: x.feature_space.isin(['best inferred', 'landmark']) & (x.gene_type == 'protein-coding')]

    kg_geneinfo = pd.read_csv(f'{args.out}/meta/gene2aa.csv') if os.path.exists(f'{args.out}/meta/gene2aa.csv') else None
    if kg_geneinfo is not None:
        genespace = set(kg_geneinfo.gene_name.unique())
        feature_filtered = feature_filtered[lambda x: x.gene_symbol.isin(genespace)]

    print(f"  feature-filtered genes: {feature_filtered.shape[0]}")

    # Map gene_id -> matrix column index and symbols; keep only present
    gene_id_to_idx = {gid: i for i, gid in enumerate(hdf_genes)}
    valid_gene_ids = [gid for gid in feature_filtered.gene_id.astype(str).tolist() if gid in gene_id_to_idx]
    valid_gene_idxs = np.array([gene_id_to_idx[gid] for gid in valid_gene_ids], dtype=np.int64)
    valid_gene_symbols = feature_filtered.set_index('gene_id').loc[valid_gene_ids, 'gene_symbol'].astype(str).values
    # h5py advanced indexing requires strictly increasing indices
    col_order = np.argsort(valid_gene_idxs)
    valid_gene_idxs = valid_gene_idxs[col_order]
    valid_gene_symbols = valid_gene_symbols[col_order]
    print(f'  valid gene columns after filters: {len(valid_gene_idxs)}')

    # Map sig->(cell, pert) and pert->inchi
    sig_to_cell = dict(zip(siginfo.sig_id.values, siginfo.cell_iname.values))
    sig_to_pert = dict(zip(siginfo.sig_id.values, siginfo.pert_id.values))
    pert_to_inchi = dict(zip(druginfo.pert_id.values, druginfo.inchi_key.values))

    # Free large intermediate metadata
    del geneinfo, feature_filtered
    gc.collect()

    total_edges = 0
    print('thresholding and writing relations per cell...')
    for cell in VALID_CELL_LINES:
        # signatures for this cell
        cell_sigs = siginfo[lambda x: x.cell_iname == cell].sig_id.values
        cell_sig_idxs = np.array([sig2hdfix[s] for s in cell_sigs if s in sig2hdfix], dtype=np.int64)
        if len(cell_sig_idxs) == 0:
            print(cell, 0)
            continue

        # read submatrix: signatures x valid genes only
        # h5py requires indices be in strictly increasing order; sort rows and carry aligned sig ids
        row_order = np.argsort(cell_sig_idxs)
        cell_sig_idxs_sorted = cell_sig_idxs[row_order]
        cell_sigs_sorted = np.array([s for s in cell_sigs if s in sig2hdfix])[row_order]
        X = hdf['0']['DATA']['0']['matrix'][cell_sig_idxs_sorted][:, valid_gene_idxs]

        # apply threshold
        if args.sig_sign == '-':
            mask = X < -args.score_threshold
        elif args.sig_sign == '+':
            mask = X > args.score_threshold
        elif args.sig_sign == '+/-':
            mask = (X < -args.score_threshold) | (X > args.score_threshold)
        else:
            raise ValueError(f'Invalid sign: {args.sig_sign}')

        si, gi = np.where(mask)
        if si.size == 0:
            print(cell, 0)
            del X, mask
            gc.collect()
            continue

        # map to identifiers
        src_inchi = [pert_to_inchi.get(sig_to_pert.get(cell_sigs_sorted[i], ''), '') for i in si]
        dst_gene = valid_gene_symbols[gi]

        edges = pd.DataFrame({'src': src_inchi, 'dst': dst_gene})
        edges = edges[lambda x: (x.src != '') & (x.dst != '')].drop_duplicates()

        rel_fwd = edges.assign(src_type='drug', dst_type='gene', relation=f'{cell}_lincs_perturbation_fwd')
        rel_rev = edges.rename(columns={'src': 'dst', 'dst': 'src'}).assign(dst_type='drug', src_type='gene', relation=f'{cell}_lincs_perturbation_rev')

        rel_fwd.to_csv(f'{args.out}/relations/{cell}_lincs_perturbation_fwd.csv', index=False)
        rel_rev.to_csv(f'{args.out}/relations/{cell}_lincs_perturbation_rev.csv', index=False)

        print(cell, rel_fwd.shape[0])
        total_edges += rel_fwd.shape[0]

        # free per-cell arrays
        del X, mask, si, gi, edges, rel_fwd, rel_rev
        gc.collect()

    print('summary:')
    print('-' * 100)
    print(f'# cells: {len(VALID_CELL_LINES)}')
    print(f'# total edges (fwd): {total_edges}')
    print('-' * 100)

    print(f'saved to: {args.out}')
    print()


