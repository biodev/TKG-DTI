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

    # embed
    AA2E = AA2EMB()
    z_prot = AA2E.embed(aas)

    S2E = SMILES2EMB()
    z_drug = S2E.embed(smiles)

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


