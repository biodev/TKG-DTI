import os
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch

from tkgdti.data.utils import get_smiles_inchikey


def get_args():
    parser = argparse.ArgumentParser(description="pre-processing for relations between drugs and diseases (CTD)")
    parser.add_argument("--data", type=str, default="../../../data/", help="Path to the input data dir")
    parser.add_argument("--extdata", type=str, default="../../extdata/", help="Path to the extra data dir")
    parser.add_argument("--out", type=str, default="../../output/", help="Path to the output data dir")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    return parser.parse_args()


def load_ctd_chem_disease(ctd_path):
    # CTD format per docs; file is CSV with headerless rows and '#' comments
    df = pd.read_csv(ctd_path, sep=',', comment='#', header=None, low_memory=False)
    df.columns = [
        'ChemicalName',
        'ChemicalID',
        'CasRN',
        'DiseaseName',
        'DiseaseID',
        'DirectEvidence',
        'InferenceGeneSymbol',
        'InferenceScore',
        'OmimIDs',
        'PubMedIDs',
    ]
    return df


def build_ctd_targetome_overlap(ctd, targetome_meta_path, cache_path):
    if os.path.exists(cache_path):
        print('NOTE: using cached CTD overlap [delete `extdata/meta/ctd_targetome_drug_overlap.csv` if this is not desired]')
        return pd.read_csv(cache_path)

    print('resolving CTD ChemicalName to InChIKey via PubChem (first result heuristic)...')
    drug_names = ctd.ChemicalName.unique().tolist()
    results = {'drug': [], 'can_smiles': [], 'inchikey': []}
    for drug in tqdm(drug_names, desc='CTD name to InChIKey'):
        can_smiles, _, inchikey = get_smiles_inchikey(drug)
        results['drug'].append(drug)
        results['can_smiles'].append(can_smiles)
        results['inchikey'].append(inchikey)
    results = pd.DataFrame(results)

    druginfo = pd.read_csv(targetome_meta_path)
    # map only those CTD names whose InChIKey appears in targetome output
    overlap = (
        druginfo[['inhibitor', 'inchikey']]
        .drop_duplicates()
        .merge(results, on='inchikey', how='inner')
        .rename({'drug': 'CTD_ChemicalName', 'inhibitor': 'Targetome_inhibitor'}, axis=1)
        .drop(columns=['can_smiles'])
    )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    overlap.to_csv(cache_path, index=False)
    return overlap


if __name__ == '__main__':
    print('------------------------------------------------------------------')
    print('kg_construction_03__drug_associates_disease.py')
    print('------------------------------------------------------------------')
    print()

    args = get_args()
    print('-------------------------------------------------------------------')
    print(args)
    print('-------------------------------------------------------------------')
    print()

    os.makedirs(f"{args.out}/meta", exist_ok=True)
    os.makedirs(f"{args.out}/relations", exist_ok=True)

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ctd_path = os.path.join(args.data, 'CTD_chemicals_diseases.csv')
    ctd = load_ctd_chem_disease(ctd_path)

    targetome_meta_path = f"{args.out}/meta/targetome__drug_targets_gene.csv"
    if not os.path.exists(targetome_meta_path):
        raise FileNotFoundError(
            f"Required file not found: {targetome_meta_path}. Run kg_construction_01__drug_interacts_protein.py first."
        )

    cache_overlap_path = f"{args.extdata}/meta/ctd_targetome_drug_overlap.csv"
    ctd_overlap = build_ctd_targetome_overlap(ctd, targetome_meta_path, cache_overlap_path)
    print(f"num matching drugs (targetome, ctd): {ctd_overlap.shape[0]}")

    # Join CTD rows to only those chemicals present in Targetome via InChIKey
    ctd = ctd.merge(ctd_overlap, left_on='ChemicalName', right_on='CTD_ChemicalName', how='inner')

    chemdis_meta = ctd[['DiseaseName', 'DiseaseID', 'ChemicalName', 'inchikey']].drop_duplicates()

    # Relations: drug (src) associates_fwd disease (dst)
    chemdis_rel_fwd = (
        chemdis_meta[['DiseaseID', 'inchikey']]
        .rename({'DiseaseID': 'dst', 'inchikey': 'src'}, axis=1)
        .assign(src_type='drug', dst_type='disease', relation='associates_fwd')
        .drop_duplicates()[['src', 'dst', 'src_type', 'dst_type', 'relation']]
    )

    chemdis_rel_rev = (
        chemdis_meta[['DiseaseID', 'inchikey']]
        .rename({'DiseaseID': 'src', 'inchikey': 'dst'}, axis=1)
        .assign(dst_type='drug', src_type='disease', relation='associates_rev')
        .drop_duplicates()[['src', 'dst', 'src_type', 'dst_type', 'relation']]
    )
    chemdis_rel_fwd.to_csv(f"{args.out}/relations/ctd__drug_disease_association_fwd.csv", index=False)
    chemdis_rel_rev.to_csv(f"{args.out}/relations/ctd__drug_disease_association_rev.csv", index=False)

    diseasespace = chemdis_meta.DiseaseID.unique()
    # summary status
    print('summary:')
    print('-' * 100)
    print(f"number of unique drugs: {chemdis_meta.ChemicalName.nunique()}")
    print(f"number of unique diseases: {chemdis_meta.DiseaseName.nunique()}")
    print(f"number of drug–disease pairs: {chemdis_meta.shape[0]}")
    print('example rows:')
    print(chemdis_meta.head(3))
    print('-' * 100)

    # save outputs
    chemdis_meta.to_csv(f"{args.out}/meta/CTD___drug_associates_disease.csv", index=False)
    chemdis_rel_fwd.to_csv(f"{args.out}/relations/ctd__drug_disease_association_fwd.csv", index=False)
    chemdis_rel_rev.to_csv(f"{args.out}/relations/ctd__drug_disease_association_rev.csv", index=False)
    np.savetxt(f"{args.out}/meta/disease_space.txt", diseasespace, fmt="%s")

    print(f'saved to: {args.out}')
    print()


