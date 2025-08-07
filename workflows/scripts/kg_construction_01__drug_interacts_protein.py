import pandas as pd
import requests
from tqdm import tqdm
import torch 
import numpy as np 
from Bio import Entrez
from io import StringIO
from Bio import SeqIO

from tkgdti.data.utils import get_smiles_inchikey, get_protein_sequence_uniprot
import argparse 


def get_args(): 

    parser = argparse.ArgumentParser(description="pre-processing for relations between drugs and proteins")
    parser.add_argument("--data", type=str, default="../data/", help="Path to the input data dir")
    parser.add_argument("--out", type=str, default="../output/", help="Path to the output data dir")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    return parser.parse_args()

def load_targetome_data(args): 
    
    tge_meta = pd.read_csv(f'{args.data}/targetome_extended_drugs-01-23-25.csv', low_memory=False)
    tge_meta = tge_meta[['inchi_key', 'drug_name']].rename({'drug_name':'inhibitor'}, axis=1).drop_duplicates()

    tge = pd.read_csv(f'{args.data}/targetome_extended-01-23-25.csv')
    tge = tge.assign(targetome_adj_tier='TIER_1')

    # filter to assay relation 
    tge = tge[lambda x: x.assay_relation == '=']

    # filter to binding affinity less than 10uM 
    tge = tge[lambda x: x.assay_value <= args.max_assay_value]

    # add drug name 
    tge = tge.merge(tge_meta, on='inchi_key', how='inner')

    # add gene symbol
    uni2symb = pd.read_csv('/home/teddy/local/TKG-DTI/extdata/meta/omnipath_uniprot2symbol.csv')
    uni2symb = uni2symb.set_index('From')[['FirstGene']].to_dict()['FirstGene']

    tge = tge.assign(Symbol=tge.uniprot_id.map(uni2symb))

    tge = tge.rename({'inchi_key': 'inchikey'}, axis=1) 

    return tge 

def get_drug_info(args, drug_names): 

    drug_names = drug_info.inhibitor.unique()
    results = {'drug': [], 'can_smiles': [], 'inchikey': [], 'iso_smiles': []}
    for drug in tqdm(drug_names):
        can_smiles, iso_smiles, inchikey = get_smiles_inchikey(drug)
        results['drug'].append(drug)
        results['can_smiles'].append(can_smiles)
        results['iso_smiles'].append(iso_smiles)
        results['inchikey'].append(inchikey)

    results = pd.DataFrame(results)
    return results 

if __name__ == "__main__": 

    print('------------------------------------------------------------------')
    print('kg_construction_01__drug_interacts_protein.py')
    print('------------------------------------------------------------------')
    print() 

    args = get_args() 
    print('-------------------------------------------------------------------')
    print(args)
    print('-------------------------------------------------------------------') 

    ######### set seed ########## 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ##############################

    drug_info = pd.read_excel(f'{args.data}/beataml_drug_families.xlsx', sheet_name='drug_gene')
    drug_info = drug_info[['inhibitor', 'Symbol', 'GeneID', 'targetome_adj_tier']]
    drug_info.targetome_adj_tier = drug_info.targetome_adj_tier.fillna('TIER_5*')
    drug_info = drug_info[lambda x: x.targetome_adj_tier == 'TIER_1']

    tge = load_targetome_data(args) 

    print(f'number of unique drugs: {drug_info.inhibitor.nunique()}')
    print(f'number of unique genes: {drug_info.Symbol.nunique()}')




# set seed 
torch.manual_seed(0)
np.random.seed(0)