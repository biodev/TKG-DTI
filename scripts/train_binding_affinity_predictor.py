import torch 
from hnet.train.hnet import train_hnet
import pandas as pd 
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from hnet.models.MLP import MLP 
from hnet.models.HyperNet import HyperNet
from hnet.train.hnet import EnergyDistanceLoss 
from sklearn.metrics import roc_auc_score
from adan import Adan

from tkgdti.embed.AA2EMB import AA2EMB
from tkgdti.embed.SMILES2EMB import SMILES2EMB

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import os 


def get_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default='../extdata/jglaser/', help="path to the data dir")
    parser.add_argument("--extdata", type=str, default='../extdata/', help="path to the external data dir")
    parser.add_argument("--out", type=str, default='../output/kd/', help="path to output dir")
    parser.add_argument("--wd", type=float, default=0, help="weight decay")
    parser.add_argument("--channels", type=int, default=256, help="number of hidden channels")
    parser.add_argument("--layers", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--batch_size", type=int, default=10000, help="batch size")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--dropout", type=float, default=0., help="dropout rate")
    parser.add_argument("--nsamples", type=int, default=250, help="number of samples")
    parser.add_argument("--hnet_channels", type=int, default=4, help="number of hidden channels in hnet")
    parser.add_argument("--hnet_width", type=int, default=256, help="width of hnet")
    parser.add_argument("--q", type=float, default=0.2, help="quantile for strong/weak binding")
    return parser.parse_args()


def load_data(): 

    affinity = pd.read_csv(f'{args.data}/jglaser_affinity_data.csv')
    z_drug = torch.load(f'{args.data}/z_drug.pt', weights_only=True)
    z_prot = torch.load(f'{args.data}/z_prot.pt', weights_only=True)

    return z_drug, z_prot, affinity

def test_(drug_idx, prot_idx, y, args, model): 

    model.eval()

    with torch.no_grad():

        splits = torch.split(torch.arange(y.size(0)), args.batch_size)

        yhats = [] 
        for j,idx in enumerate(splits): 
            print(f'[batch:{j}/{len(splits)}]', end='\r')

            didx = drug_idx[idx]
            pidx = prot_idx[idx]
            xx = torch.cat([z_drug[didx], z_prot[pidx]], dim=1).to(device)
            yhats.append( model(xx, samples=args.nsamples).cpu().mean(0) ) 
            #yhats.append( model(z_drug[didx].to(device), z_prot[pidx].to(device), samples=100).cpu().mean(0) ) 

        yhat_test = torch.cat(yhats, dim=0)

        r2 = r2_score(y.detach().cpu().numpy().ravel(), yhat_test.detach().cpu().numpy().ravel())
        mse = torch.nn.MSELoss()(yhat_test.squeeze(), y.squeeze()).item()
        r = np.corrcoef(y.detach().cpu().numpy().ravel(), yhat_test.detach().cpu().numpy().ravel())[0,1]

    return mse, r2, r

def train_step(drug_idx, prot_idx, y, args, model, optim, crit, device): 

    model.train()

    batch_loss = []
    batch_r2 = []
    batch_auc = []

    splits = torch.split(torch.randperm(y.size(0)), args.batch_size)
    for j,idx in enumerate(splits): 

        didx = drug_idx[idx]
        pidx = prot_idx[idx]
        xx = torch.cat([z_drug[didx], z_prot[pidx]], dim=1).to(device)
        yy = y[idx].to(device).unsqueeze(1)

        optim.zero_grad()
        
        yhat = model(xx, samples=args.nsamples)
        #yhat = model(z_drug[didx].to(device), z_prot[pidx].to(device), samples=100)

        loss = crit(yhat, yy)
        loss.backward()
        optim.step()

        yhat = yhat.detach().mean(dim=0)
        
        batch_loss.append(loss.item())
        r2 = r2_score(yy.detach().cpu().numpy(), yhat.detach().cpu().numpy())
        batch_r2.append(r2)

        batch_auc.append(-1)

        print(f'[batch:{j}/{len(splits)}]->{loss.item():.2f},{r2:.2f}', end='\r')

        loss = np.mean(batch_loss)
        r2 = np.mean(batch_r2)
    
    return hnet, loss, r2



if __name__ == '__main__':

    args = get_args()
    print(args)
    os.makedirs(args.out, exist_ok=True)
    print() 

    device = 'cuda' if (torch.cuda.is_available()) else 'cpu'

    z_drug, z_prot, affinity = load_data()

    train = affinity[lambda x: x.partition == 'train'] 
    train_drug_idx = torch.tensor(train.drug_idx.values, dtype=torch.long)
    train_prot_idx = torch.tensor(train.prot_idx.values, dtype=torch.long)
    train_y = torch.tensor(train.affinity.values, dtype=torch.float32)

    test = affinity[lambda x: x.partition == 'test'] 
    test_drug_idx = torch.tensor(test.drug_idx.values, dtype=torch.long)
    test_prot_idx = torch.tensor(test.prot_idx.values, dtype=torch.long)
    test_y = torch.tensor(test.affinity.values, dtype=torch.float32)

    mlp_kwargs = {'in_channels': z_drug.shape[1] + z_prot.shape[1], 
                  'dropout': args.dropout,
                'hidden_channels': args.channels, 
                'out_channels': 1, 
                'layers': args.layers}
    
    hnet_kwargs = {'stochastic_channels': args.hnet_channels, 
                    'width':args.hnet_width} 
    
    mlp = MLP(**mlp_kwargs)

    # print # params 
    n_mlp = sum([p.numel() for p in mlp.parameters()])
    print(f'# params in MLP: {n_mlp}')

    hnet = HyperNet(mlp, **hnet_kwargs).to(device)
    n_hnet = sum([p.numel() for p in hnet.parameters()])
    print(f'# params in HNet: {n_hnet}')

    optim = torch.optim.Adam(hnet.parameters(), lr=args.lr, weight_decay=args.wd)
    
    crit = EnergyDistanceLoss()

    for i in range(args.n_epochs): 

        hnet, loss, r2 = train_step(train_drug_idx, train_prot_idx, train_y, args, hnet, optim, crit, device)
        test_mse, test_r2, test_r = test_(test_drug_idx, test_prot_idx, test_y, args, hnet) 

        print(f'epoch:{i}/{args.n_epochs} mse:{loss:.2f} r2: {r2:.2f} || >> TEST >> || mse: {test_mse:.2f} r2: {test_r2:.2f}, r (pearson): {test_r:.2f}')

    torch.save(hnet, f'{args.out}/model.pt')



    ####################################
    # make predictions and eval with targetome 
    ####################################
    print()
    print('making KG predictions and evaluating with targetome...')

    tdf = pd.read_csv(f'{args.extdata}/meta/targetome__drug_targets_gene.csv')
    aadf = pd.read_csv(f'{args.extdata}/meta/gene2aa.csv')

    smiles = tdf['can_smiles'].unique().astype(str).tolist()
    aas = aadf['sequence'].unique().astype(str).tolist()

    smiles2idx = {smile: i for i, smile in enumerate(smiles)}
    aas2idx = {aa: i for i, aa in enumerate(aas)}
    gene2aas = {gene: aas for gene, aas in zip(aadf.gene_name, aadf.sequence)}
    gene2idx = lambda gene: aas2idx[gene2aas[gene]]

    tdf = tdf.assign(drug_idx=tdf.can_smiles.map(smiles2idx))
    tdf = tdf.assign(prot_idx=tdf.Symbol.map(gene2idx))

    S2E = SMILES2EMB()
    z_drug = S2E.embed(smiles)

    AA2E = AA2EMB()
    z_prot = AA2E.embed(aas)

    smiles2inchi = {smile: inchi for smile, inchi in zip(tdf.can_smiles, tdf.inchikey)}
    aa2gene = {aa: gene for aa, gene in zip(aadf.sequence, aadf.gene_name)}

    res = {'drug_idx': [], 'prot_idx': [], 'inchikey': [], 'gene_symbol': []}
    for i in range(len(smiles)): 
        for j in range(len(aas)):
            res['drug_idx'].append(i)
            res['prot_idx'].append(j)
            gene = aa2gene[aas[j]]
            inchi = smiles2inchi[smiles[i]]
            res['inchikey'].append(inchi)
            res['gene_symbol'].append(gene)

    res = pd.DataFrame(res).merge(tdf[['prot_idx', 'drug_idx']].assign(in_targetome=True), 
                                  on=['prot_idx', 'drug_idx'], how='left').fillna(False)

    y_pred = []
    with torch.no_grad(): 
        for i in range(0, len(res), args.batch_size):
            print(f'progress: {i}/{len(res)}', end='\r')
            z_drug_batch = z_drug[res.drug_idx.values[i:i+args.batch_size]]
            z_prot_batch = z_prot[res.prot_idx.values[i:i+args.batch_size]]
            zz = torch.cat([z_drug_batch, z_prot_batch], dim=1)
            y_pred.append(  hnet(zz, samples=args.nsamples).detach().cpu().numpy().mean(0)  )
    y_pred = np.concatenate(y_pred)

    res = res.assign(kd_score=y_pred)

    res.to_csv(f'{args.out}/predictions.csv', index=False)

    roc = roc_auc_score(res.in_targetome, res.kd_score)
    print(f'Targetome ROC AUC: {roc:.3f}')



