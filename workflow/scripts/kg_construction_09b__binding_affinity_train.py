import os
import argparse
import numpy as np
import pandas as pd
import torch
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
    parser.add_argument("--p_val", type=float, default=0.05)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--norm", type=str, default="batch")
    parser.add_argument("--nonlin", type=str, default="elu")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learn_pz", type=bool, default=False)
    parser.add_argument("--hnet_width", type=int, default=256)

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

    if type(z_drug) == np.ndarray:
        z_drug = torch.from_numpy(z_drug).type(torch.float32)
    if type(z_prot) == np.ndarray:
        z_prot = torch.from_numpy(z_prot).type(torch.float32)

    # split
    train_df = affinity[lambda x: x.partition == "train"]
    drug_idx_train = torch.tensor(train_df.drug_idx.values, dtype=torch.long)
    prot_idx_train = torch.tensor(train_df.prot_idx.values, dtype=torch.long)
    y_train = torch.tensor(train_df.affinity.values, dtype=torch.float32)

    test_df = affinity[lambda x: x.partition == "test"]
    drug_idx2 = torch.tensor(test_df.drug_idx.values, dtype=torch.long)
    prot_idx2 = torch.tensor(test_df.prot_idx.values, dtype=torch.long)
    y_test = torch.tensor(test_df.affinity.values, dtype=torch.float32)

    n_val = max(1, int(len(y_test) * args.p_val))
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
        "norm": args.norm, 
        "nonlin": args.nonlin,
        "dropout": args.dropout,
    }
    hnet_kwargs = {"stochastic_channels": args.stochastic_channels, 
                   "width": args.hnet_width,
                   "learn_pz": args.learn_pz,
                   }

    model = HyperNet(MLP(**mlp_kwargs), **hnet_kwargs).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = EnergyDistanceLoss()

    # -------------------------------------------------------------------------
    print() 
    print('-'*100)
    print('pre-training summary: ')
    print(f'\t- train size: {len(y_train)}')
    print(f'\t- val size: {len(y_val)}')
    print(f'\t- test size: {len(y_test)}')
    print(f'\t- # mlp params: {sum(p.numel() for p in model.model.parameters())}')
    print(f'\t- # total params: {sum(p.numel() for p in model.parameters())}')
    print(f'\t- # input channels: {z_drug.shape[1] + z_prot.shape[1]}') 
    print('-'*100)
    print()

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

    def eval_test():
        """Evaluate model on test set and return comprehensive metrics"""
        model.eval()
        splits = torch.split(torch.arange(y_test.size(0)), args.batch_size)
        yhats = []
        for idx in splits:
            with torch.no_grad():
                xx = torch.cat([z_drug[drug_idx_test[idx]], z_prot[prot_idx_test[idx]]], dim=1).to(device)
                yhats.append(model(xx, samples=args.nsamples).cpu())
        yhat_test = torch.cat(yhats, dim=1)
        yhat_mu = yhat_test.mean(dim=0).numpy().ravel()
        y_true = y_test.numpy().ravel()
        
        # Calculate metrics
        r = np.corrcoef(yhat_mu, y_true)[0, 1]
        
        # R²
        ss_res = ((y_true - yhat_mu) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum() + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        
        # MSE and MAE
        mse = mean_squared_error(y_true, yhat_mu)
        mae = mean_absolute_error(y_true, yhat_mu)
        
        # RMSE
        rmse = np.sqrt(mse)
        
        return {
            'correlation': float(r),
            'r2': float(r2),
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'n_samples': len(y_true)
        }

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

    # Evaluate on test set
    print()
    print('-'*100)
    print('FINAL TEST EVALUATION:')
    print('-'*100)
    
    test_metrics = eval_test()
    
    # Print test metrics to console
    print(f"Test Set Performance (n={test_metrics['n_samples']}):")
    print(f"\t- Correlation (r): {test_metrics['correlation']:.4f}")
    print(f"\t- R²: {test_metrics['r2']:.4f}")
    print(f"\t- MSE: {test_metrics['mse']:.4f}")
    print(f"\t- MAE: {test_metrics['mae']:.4f}")
    print(f"\t- RMSE: {test_metrics['rmse']:.4f}")
    
    # Save metrics to disk
    metrics_file = os.path.join(args.out, "binding_affinity_test_metrics.json")
    
    # Add training configuration to metrics
    full_metrics = {
        'test_metrics': test_metrics,
        'training_config': {
            'seed': args.seed,
            'nsamples': args.nsamples,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'num_epochs': args.num_epochs,
            'hidden_channels': args.hidden_channels,
            'layers': args.layers,
            'stochastic_channels': args.stochastic_channels,
            'p_val': args.p_val,
            'best_val_r2': float(best_r2)
        },
        'data_info': {
            'train_size': len(y_train),
            'val_size': len(y_val),
            'test_size': len(y_test),
            'input_channels': z_drug.shape[1] + z_prot.shape[1]
        }
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(full_metrics, f, indent=2)
    
    print(f"\nMetrics saved to: {metrics_file}")
    print('-'*100)




if __name__ == "__main__":
    main()


