import os
import argparse
import numpy as np
import pandas as pd
import torch

from datasets import load_dataset

from tkgdti.embed.AA2EMB import AA2EMB
from tkgdti.embed.SMILES2EMB import SMILES2EMB


def get_args():
    parser = argparse.ArgumentParser(
        description="KG construction step 09a: make binding affinity training data and embeddings (jglaser/binding_affinity)."
    )
    parser.add_argument("--data", type=str, default="../../../data/", help="Input data dir (unused; kept for interface parity)")
    parser.add_argument("--extdata", type=str, default="../../extdata/", help="Extra data dir (artifacts saved here)")
    parser.add_argument("--out", type=str, default="../../output/", help="Output dir (unused here)")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--train_split", type=str, default="train[:90%]", help="HF split string for training")
    parser.add_argument("--test_split", type=str, default="train[90%:]", help="HF split string for test")
    parser.add_argument("--prot_model_name", type=str, default="Rostlab/prot_bert", help="Model name")  # this should be matched to 05
    parser.add_argument("--prot_repr", type=str, default="cls", help="Representation")
    parser.add_argument("--prot_batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--prot_max_len", type=int, default=2048, help="Max length for protein")
    parser.add_argument("--smiles_model_name", type=str, default="yzimmermann/ChemBERTa-77M-MLM-safetensors", help="Model name") # this should be matched to 02
    parser.add_argument("--smiles_repr", type=str, default="cls", help="Representation")
    parser.add_argument("--smiles_batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--smiles_max_len", type=int, default=2048, help="Max length for SMILES")
    return parser.parse_args()


def main():
    args = get_args()

    print("------------------------------------------------------------------")
    print("kg_construction_09a__binding_affinity_make.py")
    print("------------------------------------------------------------------")
    print()
    print("-------------------------------------------------------------------")
    print(args)
    print("-------------------------------------------------------------------")
    print()

    # seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ensure dirs
    jg_dir = os.path.join(args.extdata, "jglaser")
    os.makedirs(jg_dir, exist_ok=True)

    # load dataset
    train = load_dataset("jglaser/binding_affinity", split=args.train_split)
    test = load_dataset("jglaser/binding_affinity", split=args.test_split)

    train_df = train.to_pandas()

    # ------------------------------------------------------------------------------------------------
    print() 
    print('-' * 100)
    print('removing TKG DTIS from jglaser training set to prevent data leak')

    # remove any of the known DTIS from `targetome_drug_targets_gene_fwd.csv` from the training set to prevent data leak
    drug_info = pd.read_csv(f'{args.out}/meta/targetome__drug_targets_gene.csv')
    gene2aa = pd.read_csv(f'{args.out}/meta/gene2aa.csv')
    drug_info = drug_info.merge(gene2aa, left_on='gene_symbol', right_on='gene_name', how='inner')[['smiles', 'sequence']]
    drug_info = drug_info.assign(kg_dti = True) 

    train_df = train_df.merge(drug_info, left_on=['smiles_can', 'seq'], right_on=['smiles', 'sequence'], how='left')

    jglaser_smiles = train_df.smiles_can.unique()
    jglaser_aas = train_df.seq.unique()

    kg_smiles = drug_info.smiles.unique()
    kg_aas = drug_info.sequence.unique()

    overlap_smiles = set(jglaser_smiles).intersection(set(kg_smiles))
    overlap_aas = set(jglaser_aas).intersection(set(kg_aas))
    print(f'# of overlap smiles (jglaser + tkg) / tkg: {len(overlap_smiles)} / {len(kg_smiles)}')
    print(f'# of overlap aas (jglaser + tkg) / tkg: {len(overlap_aas)} / {len(kg_aas)}')

    n_kg_dtis = train_df.kg_dti.sum() 
    print('# of TKG DTIS (will be removed from jglaser training set):', n_kg_dtis)

    train_df = train_df[train_df.kg_dti.isna()] # remove KG DTIS
    train_df = train_df.drop(columns=['kg_dti'])

    print('-' * 100)
    print()
    # ------------------------------------------------------------------------------------------------

    test_df = test.to_pandas()

    # build vocabularies
    train_aas = train_df.seq.unique()
    train_smiles = train_df.smiles_can.unique()
    test_aas = test_df.seq.unique()
    test_smiles = test_df.smiles_can.unique()

    smiles = list(set(train_smiles.tolist()).union(test_smiles.tolist()))
    aas = list(set(train_aas.tolist()).union(test_aas.tolist()))

    print(f"# unique sequences: {len(aas)}")
    print(f"# unique smiles: {len(smiles)}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # embed
    AA2E = AA2EMB(model_name        = args.prot_model_name, 
                  repr              = args.prot_repr, 
                  batch_size        = args.prot_batch_size, 
                  max_len           = args.prot_max_len)
                  
    print(AA2E.tokenizer.get_vocab())
    z_prot = AA2E.embed(aas, device=device)

    S2E = SMILES2EMB(model_name         = args.smiles_model_name, 
                     batch_size         = args.smiles_batch_size, 
                     repr              = args.smiles_repr,
                     max_len            = args.smiles_max_len)

    print(S2E.tokenizer.get_vocab())
    z_drug = S2E.embed(smiles, device=device)

    # save artifacts
    torch.save(z_prot, os.path.join(jg_dir, "z_prot.pt"))
    with open(os.path.join(jg_dir, "amino_acids.txt"), "w") as f:
        f.write("\n".join(aas))

    torch.save(z_drug, os.path.join(jg_dir, "z_drug.pt"))
    with open(os.path.join(jg_dir, "smiles.txt"), "w") as f:
        f.write("\n".join(smiles))

    # indices
    smiles2idx = {smile: i for i, smile in enumerate(smiles)}
    aas2idx = {aa: i for i, aa in enumerate(aas)}

    train_df = train_df.assign(partition="train")
    test_df = test_df.assign(partition="test")
    df = pd.concat([train_df, test_df])
    df = df.assign(drug_idx=df.smiles_can.map(smiles2idx))
    df = df.assign(prot_idx=df.seq.map(aas2idx))
    df = df[["drug_idx", "prot_idx", "affinity", "partition"]]

    out_csv = os.path.join(jg_dir, "jglaser_affinity_data.csv")
    df.to_csv(out_csv, index=False)

    print("summary:")
    print("-" * 100)
    print(df.head(5))
    print("saved:")
    print(f"- {out_csv}")
    print(f"- {os.path.join(jg_dir, 'z_prot.pt')}")
    print(f"- {os.path.join(jg_dir, 'z_drug.pt')}")
    print(f"- {os.path.join(jg_dir, 'amino_acids.txt')}")
    print(f"- {os.path.join(jg_dir, 'smiles.txt')}")


if __name__ == "__main__":
    main()


