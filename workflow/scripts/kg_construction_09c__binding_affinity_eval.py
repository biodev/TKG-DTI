import os
import argparse
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score

from tkgdti.embed.SMILES2EMB import SMILES2EMB


def get_args():
    parser = argparse.ArgumentParser(
        description="KG construction step 09c: evaluate trained model and generate predictions over (drug, protein) grid for targetome universe."
    )
    parser.add_argument("--extdata", type=str, default="../../extdata/", help="Extra data dir (expects jglaser + meta)")
    parser.add_argument("--out", type=str, default="../../output/", help="Output dir")
    parser.add_argument("--samples", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main():
    args = get_args()

    print("------------------------------------------------------------------")
    print("kg_construction_09c__binding_affinity_eval.py")
    print("------------------------------------------------------------------")
    print()
    print("-------------------------------------------------------------------")
    print(args)
    print("-------------------------------------------------------------------")
    print()

    os.makedirs(args.out, exist_ok=True)

    # inputs
    meta_dir = os.path.join(args.extdata, "meta")
    jg_dir = os.path.join(args.extdata, "jglaser")

    tdf = pd.read_csv(os.path.join(meta_dir, "targetome__drug_targets_gene.csv"))
    aadf = pd.read_csv(os.path.join(meta_dir, "gene2aa.csv"))

    smiles = tdf["can_smiles"].astype(str).unique().tolist()
    smiles2idx = {s: i for i, s in enumerate(smiles)}

    # embed drugs (use same embedder as data make step)
    z_drug = SMILES2EMB().embed(smiles)

    aas_dict = torch.load(os.path.join(meta_dir, "aas_dict.pt"), weights_only=False)
    aas = aas_dict["amino_acids"]
    z_prot = torch.tensor(aas_dict["embeddings"], dtype=torch.float32)

    gene2aas = {g: a for g, a in zip(aadf.gene_name, aadf.sequence)}
    aas2idx = {aa: i for i, aa in enumerate(aas)}
    gene2idx = lambda gene: aas2idx[gene2aas[gene]]

    tdf = tdf.assign(drug_idx=tdf.can_smiles.map(smiles2idx))
    tdf = tdf.assign(prot_idx=tdf.Symbol.map(gene2idx))

    # load model
    model_path = os.path.join(jg_dir, "model.pt")
    model = torch.load(model_path, weights_only=False)
    device = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu")
    )
    model = model.to(device)

    # build full grid
    res = {"drug_idx": [], "prot_idx": [], "inchikey": [], "gene_symbol": []}
    smiles2inchi = {s: i for s, i in zip(tdf.can_smiles, tdf.inchikey)}
    aa2gene = {a: g for a, g in zip(aadf.sequence, aadf.gene_name)}

    for i in range(len(smiles)):
        for j in range(len(aas)):
            res["drug_idx"].append(i)
            res["prot_idx"].append(j)
            res["inchikey"].append(smiles2inchi[smiles[i]])
            res["gene_symbol"].append(aa2gene[aas[j]])

    res = pd.DataFrame(res).merge(
        tdf[["prot_idx", "drug_idx"]].assign(in_targetome=True), on=["prot_idx", "drug_idx"], how="left"
    ).fillna(False)

    # predict
    y_pred, lcb_pred, ucb_pred = [], [], []
    for i in range(0, len(res), args.batch_size):
        print(f"progress: {i}/{len(res)}", end="\r")
        z_drug_batch = z_drug[res.drug_idx.values[i : i + args.batch_size]]
        z_prot_batch = z_prot[res.prot_idx.values[i : i + args.batch_size]]
        zz = torch.cat([z_drug_batch, z_prot_batch], dim=1).to(device)
        with torch.no_grad():
            preds = model(zz, samples=args.samples).detach().cpu().numpy()
        y_pred.append(preds.mean(0))
        lcb_pred.append(np.quantile(preds, q=0.05, axis=0))
        ucb_pred.append(np.quantile(preds, q=0.95, axis=0))

    y_pred = np.concatenate(y_pred)
    lcb_pred = np.concatenate(lcb_pred)
    ucb_pred = np.concatenate(ucb_pred)

    res = res.assign(kd_score=y_pred, kd_lcb=lcb_pred, kd_ucb=ucb_pred)

    # basic eval signal (optional; requires in_targetome indicator)
    try:
        auroc = roc_auc_score(res.in_targetome, res.kd_score)
        print(f"Targetome AUROC: {auroc:.4f}")
    except Exception:
        pass

    pred_csv = os.path.join(jg_dir, "predictions.csv")
    res.to_csv(pred_csv, index=False)
    print(f"saved predictions -> {pred_csv}")


if __name__ == "__main__":
    main()


