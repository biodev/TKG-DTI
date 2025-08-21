import os
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score


def get_args():
    parser = argparse.ArgumentParser(
        description="KG construction step 09d: post-process predicted binding affinities and export strong-binding relations."
    )
    parser.add_argument("--extdata", type=str, default="../../extdata/", help="Extra data dir (expects jglaser predictions)")
    parser.add_argument("--q_weak", type=float, default=0.05, help="Weak binding quantile")
    parser.add_argument("--q_strong", type=float, default=0.95, help="Strong binding quantile")
    parser.add_argument("--out", type=str, default="../../output/", help="Output dir")
    return parser.parse_args()


def main():
    args = get_args()

    print("------------------------------------------------------------------")
    print("kg_construction_09d__predicted_binding_affinity.py")
    print("------------------------------------------------------------------")
    print()
    print("-------------------------------------------------------------------")
    print(args)
    print("-------------------------------------------------------------------")
    print()

    jg_dir = os.path.join(args.extdata, "jglaser")
    os.makedirs(os.path.join(args.extdata, "relations"), exist_ok=True)

    res_path = os.path.join(jg_dir, "predictions.csv")
    res = pd.read_csv(res_path)

    # CRITICAL FIX: Filter to only original targetome drugs (89 drugs)
    # Load the original targetome drug set to ensure we only include those drugs
    targetome_meta_path = os.path.join(args.extdata, "meta", "targetome__drug_targets_gene.csv")
    if os.path.exists(targetome_meta_path):
        targetome_drugs = pd.read_csv(targetome_meta_path)
        valid_inchikeys = set(targetome_drugs['inchikey'].dropna())
        
        print(f"Original targetome drugs: {len(valid_inchikeys)}")
        print(f"Predictions before drug filtering: {res['inchikey'].nunique()} unique drugs")
        
        # Filter predictions to only include the original targetome drugs
        res = res[res['inchikey'].isin(valid_inchikeys)]
        
        print(f"Predictions after drug filtering: {res['inchikey'].nunique()} unique drugs")
    else:
        print(f"Warning: Could not find {targetome_meta_path} - proceeding without drug filtering")

    # filter out microRNAs
    res = res[lambda x: ~x.gene_symbol.str.startswith("MIR")]

    # quick metrics if available
    if {"in_targetome", "kd_score"}.issubset(res.columns):
        try:
            auroc = roc_auc_score(res.in_targetome, res.kd_score)
            aupr = average_precision_score(res.in_targetome, res.kd_score)
            print(f"Targetome AUROC: {auroc:.6f}")
            print(f"Targetome AUPR:  {aupr:.6f}")
        except Exception:
            pass

    # choose quantile thresholds
    q10, q90 = np.quantile(res.kd_score, [args.q_weak, args.q_strong])
    res = res.assign(
        weak_binders=lambda x: x.kd_score < q10,
        strong_binders=lambda x: x.kd_score > q90,
        conf_weak=lambda x: x.kd_ucb < np.quantile(res.kd_ucb, 0.025),
        conf_strong=lambda x: x.kd_lcb > np.quantile(res.kd_lcb, 0.975),
    )

    print(f"# weak binders: {int(res.weak_binders.sum())}")
    print(f"# strong binders: {int(res.strong_binders.sum())}")
    print(f"# confident weak binders: {int(res.conf_weak.sum())}")
    print(f"# confident strong binders: {int(res.conf_strong.sum())}")

    # export strong binding relations only (per ablation note)
    strong_fwd = (
        res[lambda x: x.strong_binders]
        .rename({"inchikey": "src", "gene_symbol": "dst"}, axis=1)
        .assign(src_type="drug", dst_type="gene", relation="predicted_strong_binding_fwd")
        [["src", "dst", "src_type", "dst_type", "relation"]]
    )
    
    strong_rev = (
        res[lambda x: x.strong_binders]
        .rename({"inchikey": "dst", "gene_symbol": "src"}, axis=1)
        .assign(src_type="gene", dst_type="drug", relation="predicted_strong_binding_rev")
        [["src", "dst", "src_type", "dst_type", "relation"]]
    )

    rel_dir = os.path.join(args.extdata, "relations")
    strong_fwd.to_csv(os.path.join(rel_dir, "jglaser__predicted_strong_binding_fwd.csv"), index=False)
    strong_rev.to_csv(os.path.join(rel_dir, "jglaser__predicted_strong_binding_rev.csv"), index=False)
    print(f"saved strong-binding relations -> {rel_dir}")


if __name__ == "__main__":
    main()


