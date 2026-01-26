import pandas as pd
from tqdm import tqdm
import torch 
import numpy as np 
from gsnn.proc.bio import uniprot2symbol, symbol2uniprot
import os

from tkgdti.data.utils import get_smiles_inchikey, get_protein_sequence_uniprot
import argparse 
import pubchempy as pcp
from concurrent.futures import ThreadPoolExecutor, as_completed
import rdkit.Chem as Chem


def get_args(): 

    parser = argparse.ArgumentParser(description="pre-processing for relations between drugs and proteins")
    parser.add_argument("--data", type=str, default="../../../data/", help="Path to the input data dir")
    parser.add_argument("--extdata", type=str, default="../../extdata/", help="Path to the extra data dir")
    parser.add_argument("--out", type=str, default="../../output/", help="Path to the output data dir")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--assay_types", nargs='+', default=['Kd', 'Ki'], help="Assay types to include; options: [Kd, Ki, IC50, EC50]")
    parser.add_argument("--max_assay_value", type=float, default=1000, help="Maximum assay value to include (nM)")
    parser.add_argument("--TIERS", nargs='+', default=['TIER_1'], help="Tiers to include; options: [TIER_1, TIER_2, TIER_3, TIER_4, TIER_5*]")
    return parser.parse_args()

def load_tge(args): 
    
    tge_meta = pd.read_csv(f'{args.data}/targetome_extended_drugs-01-23-25.csv', low_memory=False)
    tge_meta = tge_meta[['inchi_key', 'drug_name', 'molecule_type', 'clinical_phase', 'inchi']].rename({'drug_name':'inhibitor'}, axis=1).drop_duplicates()

    tge = pd.read_csv(f'{args.data}/targetome_extended-01-23-25.csv')
    tge = tge.assign(targetome_adj_tier='TIER_1') # Future focused, if we implement multiple tiers 

    # add drug name 
    tge = tge.merge(tge_meta, on='inchi_key', how='inner')

    uni2symb = uniprot2symbol(tge.uniprot_id.unique())
    tge = tge.merge(uni2symb, on='uniprot_id', how='inner')

    tge = tge.rename({'inchi_key': 'inchikey'}, axis=1) 

    tge = tge.drop_duplicates()

    drug_names = tge_meta[lambda x: (x.molecule_type == 'Small molecule') 
                                  & (x.clinical_phase != 'preclinical compounds with bioactivity data')].inhibitor.unique().tolist()

    tge = tge[lambda x: x.inhibitor.isin(drug_names)] # filter to beataml drugs 

    inchi2smiles = get_smiles_from_inchi(tge.inchi.values)

    tge = tge.merge(inchi2smiles, on='inchi', how='left')

    #tge = tge.drop(columns='inhibitor_x').rename(columns={'inhibitor_y': 'inhibitor'}) # use TG inhibitor names 

    return tge 

def get_smiles_from_inchi(inchis, verbose=True):

    inchis = np.unique(inchis).tolist() 

    res = {'inchi': [], 'smiles': []}
    for i, inchi in enumerate(inchis):    
        if verbose:
            print(f"progress: {i}/{len(inchis)}", end="\r")
        mol = Chem.MolFromInchi(inchi)
        if mol is not None:
            smiles_string = Chem.MolToSmiles(mol)
            res['inchi'].append(inchi)
            res['smiles'].append(smiles_string)
        else:
            res['inchi'].append(inchi)
            res['smiles'].append(None)

    res = pd.DataFrame(res)

    return res


def filter_dtis(dtis, args): 

    dtis = dtis[lambda x: x.targetome_adj_tier.isin(args.TIERS)]

    # filter to assay relation 
    dtis = dtis[lambda x: x.assay_relation.isin(['=', '<', '<='])]

    # filter to assay type 
    dtis = dtis[lambda x: x.assay_type.isin(args.assay_types)]

    # filter to binding affinity less than XX nM
    dtis = dtis[lambda x: x.assay_value <= args.max_assay_value]

    return dtis 
    

if __name__ == "__main__": 

    print('------------------------------------------------------------------')
    print('kg_construction_01__drug_interacts_protein.py')
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

    #tg = load_tg(args).assign(file_source='beataml_drug_families.xlsx')
    #tg = tg[['inhibitor', 'inchikey', 'gene_symbol', 'uniprot_id', 'targetome_adj_tier', 'file_source', \
    #    'assay_relation', 'assay_value', 'assay_type', 'smiles']]

    tge = load_tge(args).assign(file_source='targetome_extended-01-23-25.csv')
    tge = tge[['inhibitor', 'inchikey', 'gene_symbol', 'uniprot_id', 'targetome_adj_tier', 'file_source', \
        'assay_relation', 'assay_value', 'assay_type', 'smiles']]

    drug_info = filter_dtis(tge, args)
    drug_info = drug_info[['inhibitor', 'inchikey', 'gene_symbol', 'uniprot_id', 'file_source', 'smiles']].drop_duplicates()

    # bug fix: was getting more unique inchikeys than unique drugs 
    drug_info = drug_info.dropna(subset=['inchikey'])
    drug_info = drug_info.dropna(subset=['smiles'])
    drug_info = drug_info.dropna(subset=['gene_symbol'])
    drug_info = drug_info.dropna(subset=['inhibitor']) # this fixed it I think 

    
    print('summary:')
    print('-'*100)
    print(f'number of unique drugs: {drug_info.inhibitor.nunique()}') # multiple names map to same smiles 
    print(f'number of unique gene symbols: {drug_info.gene_symbol.nunique()}')
    print(f'number of unique uniprot ids: {drug_info.uniprot_id.nunique()}')
    print(f'number of unique inchikeys: {drug_info.inchikey.nunique()}')
    print(f'number of DTIs: {drug_info.shape[0]}')
    print(drug_info.groupby('file_source').count()[['inhibitor']].rename({'inhibitor': 'count'}, axis=1))
    print() 
    print('example rows:')
    print(drug_info.head(3))
    print('-'*100)

    drug_info.to_csv(f'{args.out}/meta/targetome__drug_targets_gene.csv', index=False)

    rel_fwd = drug_info[['inchikey', 'gene_symbol']].rename({'inchikey': 'src', 'gene_symbol': 'dst'}, axis=1).assign(src_type='drug', dst_type='gene', relation='targets')
    rel_fwd.to_csv(f'{args.out}/relations/targetome_drug_targets_gene_fwd.csv', index=False)

    print(f'saved to: {args.out}')
    print() 



