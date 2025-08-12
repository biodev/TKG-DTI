import os
import argparse
import numpy as np
import pandas as pd
import torch

from hnet.models.MLP import MLP
from hnet.models.HyperNet import HyperNet
from hnet.train.hnet import EnergyDistanceLoss


def get_args():
    parser = argparse.ArgumentParser(
        description="KG construction step 09b: train binding affinity model (HyperNet) on jglaser data."
    )
    parser.add_argument("--extdata", type=str, default="../../extdata/", help="Extra data dir (expects jglaser artifacts)")
    parser.add_argument("--out", type=str, default="../../output/", help="Output dir")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--nsamples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--stochastic_channels", type=int, default=4)
    return parser.parse_args()


def main():
    args = get_args()

    print("------------------------------------------------------------------")
    print("kg_construction_09b__binding_affinity_train.py")
    print("------------------------------------------------------------------")
    print()
    print("-------------------------------------------------------------------")
    print(args)
    print("-------------------------------------------------------------------")
    print()

    # seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    jg_dir = os.path.join(args.extdata, "jglaser")
    os.makedirs(args.out, exist_ok=True)

    affinity = pd.read_csv(os.path.join(jg_dir, "jglaser_affinity_data.csv"))
    aas = np.loadtxt(os.path.join(jg_dir, "amino_acids.txt"), dtype=str)
    smiles = np.loadtxt(os.path.join(jg_dir, "smiles.txt"), dtype=str)

    z_drug = torch.load(os.path.join(jg_dir, "z_drug.pt"), weights_only=False)
    z_prot = torch.load(os.path.join(jg_dir, "z_prot.pt"), weights_only=False)

    # split
    train_df = affinity[lambda x: x.partition == "train"]
    drug_idx_train = torch.tensor(train_df.drug_idx.values, dtype=torch.long)
    prot_idx_train = torch.tensor(train_df.prot_idx.values, dtype=torch.long)
    y_train = torch.tensor(train_df.affinity.values, dtype=torch.float32)

    test_df = affinity[lambda x: x.partition == "test"]
    drug_idx2 = torch.tensor(test_df.drug_idx.values, dtype=torch.long)
    prot_idx2 = torch.tensor(test_df.prot_idx.values, dtype=torch.long)
    y_test = torch.tensor(test_df.affinity.values, dtype=torch.float32)

    n_val = max(1, int(len(y_test) * 0.05))
    val_idx = np.random.choice(len(y_test), n_val, replace=False)
    test_idx = np.setdiff1d(np.arange(len(y_test)), val_idx)

    y_val = y_test[val_idx]
    y_test = y_test[test_idx]
    drug_idx_val = drug_idx2[val_idx]
    prot_idx_val = prot_idx2[val_idx]
    drug_idx_test = drug_idx2[test_idx]
    prot_idx_test = prot_idx2[test_idx]

    device = "cuda" if (torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")

    mlp_kwargs = {
        "in_channels": z_drug.shape[1] + z_prot.shape[1],
        "hidden_channels": args.hidden_channels,
        "out_channels": 1,
        "layers": args.layers,
    }
    hnet_kwargs = {"stochastic_channels": args.stochastic_channels, "width": args.hidden_channels}

    model = HyperNet(MLP(**mlp_kwargs), **hnet_kwargs).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = EnergyDistanceLoss()

    def eval_val():
        splits = torch.split(torch.arange(y_val.size(0)), args.batch_size)
        yhats = []
        for idx in splits:
            with torch.no_grad():
                xx = torch.cat([z_drug[drug_idx_val[idx]], z_prot[prot_idx_val[idx]]], dim=1).to(device)
                yhats.append(model(xx, samples=args.nsamples).cpu())
        yhat_val = torch.cat(yhats, dim=1)
        yhat_mu = yhat_val.mean(dim=0).numpy().ravel()
        r = np.corrcoef(yhat_mu, y_val.numpy().ravel())[0, 1]
        # r2
        ss_res = ((y_val.numpy().ravel() - yhat_mu) ** 2).sum()
        ss_tot = ((y_val.numpy().ravel() - y_val.numpy().ravel().mean()) ** 2).sum() + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        return r, r2

    best_r2 = -np.inf
    best_state = None
    for epoch in range(args.num_epochs):
        splits = torch.split(torch.randperm(y_train.size(0)), args.batch_size)
        batch_losses = []
        for ii, idx in enumerate(splits):
            xx = torch.cat([z_drug[drug_idx_train[idx]], z_prot[prot_idx_train[idx]]], dim=1).to(device)
            yy = y_train[idx].to(device).unsqueeze(1)
            optim.zero_grad()
            yhat = model(xx, samples=args.nsamples)
            loss = crit(yhat, yy)
            loss.backward()
            optim.step()
            batch_losses.append(loss.item())
            print(f'[batch {ii+1}/{len(splits)}: loss: {loss.item():.4f}]', end='\r')

        r, r2 = eval_val()
        print(f"epoch {epoch+1}/{args.num_epochs} -> loss: {np.mean(batch_losses):.4f}, val r: {r:.3f}, val r2: {r2:.3f}")
        if r2 > best_r2:
            best_r2 = r2
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    out_model = os.path.join(jg_dir, "model.pt")
    torch.save(model, out_model)
    print(f"saved model -> {out_model}")


if __name__ == "__main__":
    main()


