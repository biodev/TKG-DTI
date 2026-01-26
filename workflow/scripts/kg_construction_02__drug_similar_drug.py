
import torch
import numpy as np
from rdkit import Chem
import requests
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import umap
from sklearn.cluster import DBSCAN
import torchvision
from tkgdti.embed.SMILES2EMB import SMILES2EMB

import os
import argparse 


# NOTE: seyonec/ChemBERTa-zinc-base-v1 does not have support for "safetensors"
# switch to: ChemBERTa-77M-MLM-safetensors or yzimmermann/ChemBERTa-zinc-base-v1-safetensors 
# Higly recommend: ChemBERTa-77M-MLM-safetensor 


def get_args(): 

    parser = argparse.ArgumentParser(description="pre-processing for relations between drugs and drugs (similarity)")
    parser.add_argument("--data", type=str, default="../../../data/", help="Path to the input data dir")
    parser.add_argument("--extdata", type=str, default="../../extdata/", help="Path to the extra data dir")
    parser.add_argument("--out", type=str, default="../../output/", help="Path to the output data dir")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--repr", type=str, default='cls', help="Representation to use for drug embeddings")
    parser.add_argument("--model_name", type=str, default="yzimmermann/ChemBERTa-77M-MLM-safetensors", help="Model name")
    parser.add_argument("--method", type=str, default='knn', help="How to threshold similarity [options: threshold, knn]")
    parser.add_argument("--q_threshold", type=float, default=0.95, help="Quantile threshold for cosine similarity")
    parser.add_argument("--knn_k", type=int, default=3, help="Number of nearest neighbors for knn thresholding")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for embedding")


    return parser.parse_args()

def get_dds(z_drug, smiles): 

    # Initialize an empty list to store the combined embeddings
    res = {'drug_i': [], 'drug_j': [], 'cos_sim': [],  'mse': []}

    print('computing similarity between drugs...') 
    # Iterate over the upper triangle indices
    ii = 1
    for i in range(len(smiles)):
        for j in range(len(smiles)):
            print(f'\tprogress: {ii/(len(smiles)**2)*100:.1f} %', end='\r'); ii += 1

            if i == j:
                continue

            res['cos_sim'].append(cosine_similarity([z_drug[i]], [z_drug[j]])[0][0])
            # Convert PyTorch tensors to numpy for MSE calculation
            z_i = z_drug[i].numpy() if hasattr(z_drug[i], 'numpy') else z_drug[i]
            z_j = z_drug[j].numpy() if hasattr(z_drug[j], 'numpy') else z_drug[j]
            res['mse'].append(np.mean((z_i - z_j)**2))
            res['drug_i'].append(i)
            res['drug_j'].append(j)
    
    print() 

    # Convert the list to a numpy array
    res = pd.DataFrame(res)
    res = res.assign(smiles_i = [smiles[i] for i in res['drug_i']],
                     smiles_j = [smiles[j] for j in res['drug_j']])

    return res 



if __name__ == "__main__": 

    print('------------------------------------------------------------------')
    print('kg_construction_02__drug_similar_drug.py')
    print('------------------------------------------------------------------')
    print() 

    args = get_args() 
    print('-------------------------------------------------------------------')
    print(args)
    print('-------------------------------------------------------------------') 
    print() 

    os.makedirs(f'{args.out}/meta', exist_ok=True)
    os.makedirs(f'{args.out}/relations', exist_ok=True)

    ######### set seed ########## 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ##############################

    sm2emb = SMILES2EMB(model_name=args.model_name, batch_size=args.batch_size, repr=args.repr)

    drug_info = pd.read_csv(f'{args.out}/meta/targetome__drug_targets_gene.csv')

    inchi2inhibitor = drug_info[['inchikey', 'inhibitor']].drop_duplicates().set_index('inchikey')['inhibitor'].to_dict()

    smiles = drug_info['smiles'].unique().astype(str).tolist()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    z_drug = sm2emb.embed(smiles, device=device, verbose=True)

    sim_res = get_dds(z_drug, smiles)

    if args.method == 'threshold': 

        threshold = np.quantile(sim_res['cos_sim'], args.q_threshold) 
        sim_res = sim_res.assign(is_similar=sim_res['cos_sim'] > threshold)

    elif args.method == 'knn': 
        
        # get k nearest neighbors for each drug 
        sim_res2 = sim_res.groupby('drug_i').apply(lambda x: x.nlargest(args.knn_k, 'cos_sim')).reset_index(drop=True)
        sim_res2 = sim_res2.assign(is_similar=True)[['drug_i', 'drug_j', 'is_similar']]
        sim_res = sim_res.merge(sim_res2, on=['drug_i', 'drug_j'], how='left')
        sim_res.is_similar = sim_res.is_similar.fillna(False)

    s2i = drug_info[['smiles', 'inchikey']].drop_duplicates()
    sim_res = sim_res.merge(s2i, left_on='smiles_i', right_on='smiles', how='left').drop(columns='smiles')
    sim_res = sim_res.merge(s2i, left_on='smiles_j', right_on='smiles', how='left').drop(columns='smiles')

    sim_res = sim_res.assign(inhibitor_x = sim_res['inchikey_x'].map(inchi2inhibitor),
                             inhibitor_y = sim_res['inchikey_y'].map(inchi2inhibitor))

    print('summary:')
    print('-'*100)
    if args.method == 'threshold': 
        print(f'Cosine similarity threshold: {threshold:.2f}')
    elif args.method == 'knn': 
        print(f'KNN thresholding with k={args.knn_k}')
    print(f'# of similar drug pairs: {sim_res["is_similar"].sum()} [p={sim_res["is_similar"].mean():.2f}]')
    print(f'# of non-similar drug pairs: {(~sim_res["is_similar"]).sum()} [p={(~sim_res["is_similar"]).mean():.2f}]') 
    print('examples [similar]:')
    print(sim_res[lambda x: x['is_similar']].head(5)[['inhibitor_x', 'inhibitor_y', 'cos_sim']])
    print('examples [non-similar]:')
    print(sim_res[lambda x: ~x['is_similar']].head(5)[['inhibitor_x', 'inhibitor_y', 'cos_sim']])
    print('-'*100)

    sim_res.to_csv(f'{args.out}/meta/chemberta_drug_drug_similarity.csv', index=False)

    sim_relations = sim_res[sim_res['is_similar']]
    sim_relations = sim_relations[['inchikey_x', 'inchikey_y']].rename({'inchikey_x': 'src', 'inchikey_y': 'dst'}, axis=1)
    sim_relations = sim_relations.assign(src_type = 'drug', dst_type = 'drug', relation = 'chemberta_cosine_similarity')
    sim_relations.to_csv(f'{args.out}/relations/chemberta_drug_cosine_similarity.csv', index=False)
