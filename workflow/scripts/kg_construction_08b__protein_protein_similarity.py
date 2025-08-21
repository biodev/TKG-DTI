import os
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity


def get_args():
    parser = argparse.ArgumentParser(description='protein-protein similarity using ProtBert embeddings')
    parser.add_argument('--data', type=str, default='../../../data/', help='Path to the input data dir')
    parser.add_argument('--extdata', type=str, default='../../extdata/', help='Path to the extra data dir')
    parser.add_argument('--out', type=str, default='../../output/', help='Path to the output data dir')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--sim_quantile', type=float, default=0.999, help='Quantile for cosine similarity threshold')
    return parser.parse_args()


def compute_upper_triangle_links(z: np.ndarray):
    res = {'drug_i': [], 'drug_j': [], 'cos_sim': []}
    N = z.shape[0]
    for i in range(N - 1):
        if i % 100 == 0:
            print(f'progress: {i}/{N}', end='\r')
        a = z[[i]]
        b = z[i + 1:].reshape(-1, z.shape[1])
        sims = cosine_similarity(a, b).ravel()
        j_vals = list(range(i + 1, N))
        res['drug_i'].extend([i] * len(j_vals))
        res['drug_j'].extend(j_vals)
        res['cos_sim'].extend(sims.tolist())
    print()
    return pd.DataFrame(res)


if __name__ == '__main__':
    print('------------------------------------------------------------------')
    print('kg_construction_08b__protein_protein_similarity.py')
    print('------------------------------------------------------------------')
    print()

    args = get_args()
    print('-------------------------------------------------------------------')
    print(args)
    print('-------------------------------------------------------------------')
    print()

    os.makedirs(f'{args.out}/relations', exist_ok=True)
    os.makedirs(f'{args.out}/meta', exist_ok=True)

    # load embeddings dict from previous step
    embed_dict = torch.load(f'{args.out}/meta/aas_dict.pt', weights_only=False)
    aas = embed_dict['amino_acids']
    gene2aa = embed_dict['meta_df']
    z_prot = embed_dict['embeddings']

    print(f'z_prot shape: {z_prot.shape}')

    # compute pairwise similarities (upper triangle)
    res = compute_upper_triangle_links(z_prot)

    # map indices to gene symbols
    aa2gene = {aa: g for aa, g in zip(aas, gene2aa.gene_name)}
    res = res.assign(
        gene_i=[aa2gene[aas[i]] for i in res['drug_i']],
        gene_j=[aa2gene[aas[j]] for j in res['drug_j']],
    )
    res = res[lambda x: x.gene_i != x.gene_j]

    # threshold by quantile
    cos_sim_thresh = np.quantile(res['cos_sim'], args.sim_quantile)
    print(f'Cosine similarity threshold: {cos_sim_thresh:.4f}')
    res = res.assign(is_similar=res['cos_sim'] > cos_sim_thresh)

    sim_relations = res[res['is_similar']]
    sim_relations = sim_relations[['gene_i', 'gene_j']].rename({'gene_i': 'src', 'gene_j': 'dst'}, axis=1)
    sim_relations = sim_relations.assign(src_type='gene', dst_type='gene', relation='protbert_similarity')
    sim_relations = pd.concat([sim_relations, sim_relations.rename({'src': 'dst', 'dst': 'src'}, axis=1)])

    sim_relations.to_csv(f'{args.out}/relations/protbert__gene_gene_similarity.csv', index=False)

    print('summary:')
    print('-' * 100)
    print(f'# pairs above threshold (directed): {sim_relations.shape[0]}')
    print('example rows:')
    print(sim_relations.head(5)[['src', 'dst']])
    print('-' * 100)

    print(f'saved to: {args.out}')
    print()



