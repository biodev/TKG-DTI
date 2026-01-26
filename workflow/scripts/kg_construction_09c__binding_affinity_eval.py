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
    parser.add_argument("--data", type=str, default="../../output/", help="Experiment data dir (expects jglaser + meta)")
    parser.add_argument("--out", type=str, default="../../output/", help="Output dir")
    parser.add_argument("--samples", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    # Embedding parameters (should match training parameters)
    parser.add_argument("--smiles_model_name", type=str, default="yzimmermann/ChemBERTa-zinc-base-v1-safetensors", help="SMILES model name")
    parser.add_argument("--smiles_repr", type=str, default="mean", help="SMILES representation")
    parser.add_argument("--smiles_batch_size", type=int, default=128, help="SMILES batch size")
    parser.add_argument("--smiles_max_len", type=int, default=1024, help="SMILES max length")
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
    meta_dir = os.path.join(args.data, "meta")
    jg_dir = os.path.join(args.data, "jglaser")

    tdf = pd.read_csv(os.path.join(meta_dir, "targetome__drug_targets_gene.csv"))
    aadf = pd.read_csv(os.path.join(meta_dir, "gene2aa.csv"))

    # CRITICAL FIX: Only use drugs from the original targetome (89 drugs)
    # Filter to ensure we only get smiles from the original targetome drugs
    smiles = tdf["smiles"].astype(str).unique().tolist()
    smiles2idx = {s: i for i, s in enumerate(smiles)}
    
    print(f"Making predictions for {len(smiles)} targetome drugs (should be 89 or close)")
    print(f"First few SMILES: {smiles[:3]}")
    
    # TODO: the drug embeddings should be cached from step 02 and loaded here; for now, we re-embed
    print(f"Computing drug embeddings for {len(smiles)} targetome drugs...")
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu")
    
    sm2emb = SMILES2EMB(
        model_name=args.smiles_model_name,
        batch_size=args.smiles_batch_size,
        repr=args.smiles_repr,
        max_len=args.smiles_max_len
    )
    z_drug = sm2emb.embed(smiles, device=device, verbose=True)
    
    aas_dict = torch.load(os.path.join(meta_dir, "aas_dict.pt"), weights_only=False)
    aas = aas_dict["amino_acids"]
    z_prot = torch.tensor(aas_dict["embeddings"], dtype=torch.float32)

    gene2aas = {g: a for g, a in zip(aadf.gene_name, aadf.sequence)}
    aas2idx = {aa: i for i, aa in enumerate(aas)}
    
    def gene2idx(gene):
        if pd.isna(gene) or gene not in gene2aas:
            return None
        return aas2idx.get(gene2aas[gene], None)

    tdf = tdf.assign(drug_idx=tdf.smiles.map(smiles2idx))
    tdf = tdf.assign(prot_idx=tdf.gene_symbol.map(gene2idx))
    
    # Remove rows with missing indices
    initial_rows = len(tdf)
    tdf = tdf.dropna(subset=['drug_idx', 'prot_idx'])
    final_rows = len(tdf)
    if initial_rows != final_rows:
        print(f"Warning: Removed {initial_rows - final_rows} rows with missing drug or protein indices")

    # load model
    model_path = os.path.join(jg_dir, "model.pt")
    model = torch.load(model_path, weights_only=False)
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu")
    model = model.to(device)

    # build full grid
    res = {"drug_idx": [], "prot_idx": [], "inchikey": [], "gene_symbol": []}
    smiles2inchi = {s: i for s, i in zip(tdf.smiles, tdf.inchikey)}
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
        
        # Calculate MRR (Mean Reciprocal Rank)
        # For each drug, rank proteins by kd_score and find reciprocal rank of true targets
        mrr_scores = []
        for drug_idx in res.drug_idx.unique():
            drug_data = res[res.drug_idx == drug_idx].copy()
            
            # Sort by kd_score in descending order (higher score = better rank)
            drug_data = drug_data.sort_values('kd_score', ascending=False).reset_index(drop=True)
            
            # Find ranks of true targets (1-indexed)
            true_target_ranks = drug_data[drug_data.in_targetome == True].index + 1
            
            if len(true_target_ranks) > 0:
                # Take reciprocal of the best (smallest) rank for this drug
                best_rank = min(true_target_ranks)
                mrr_scores.append(1.0 / best_rank)
        
        mrr = np.mean(mrr_scores) if mrr_scores else 0.0
        
        print(f"Targetome AUROC: {auroc:.4f}")
        print(f"Targetome MRR: {mrr:.4f}")
    except Exception:
        pass

    pred_csv = os.path.join(jg_dir, "predictions.csv")
    res.to_csv(pred_csv, index=False)
    print(f"saved predictions -> {pred_csv}")


if __name__ == "__main__":
    main()


