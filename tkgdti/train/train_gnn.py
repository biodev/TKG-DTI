
import torch 
import torch_geometric as pyg
import numpy as np
import time 
from sklearn.metrics import roc_auc_score
from tkgdti.data.TriplesDatasetGNN import TriplesDatasetGNN
from tkgdti.data.process_graph import process_graph
from tkgdti.train.utils import device_and_data_loading

from tkgdti.models.GNN import GNN
from tkgdti.train.EarlyStopper import EarlyStopper
import pandas as pd 


def predict_all(data, tdata, train_triples, valid_triples, test_triples, model, device): 

    edge_index_dict = {key: tdata[key]['edge_index'] for key in tdata.metadata()[1]}
    num_node_dict = {key: tdata[key].num_nodes for key in tdata.metadata()[0]}

    datas = []
    for i in range(data['num_nodes_dict']['drug']):
        head = i 
        x_dict = {node:torch.zeros((num_nodes, 1), dtype=torch.float32) for node, num_nodes in num_node_dict.items()}

        x_dict['drug'][head] = torch.ones((1,), dtype=torch.float32)

        edge_index_dict = edge_index_dict

        # Create a HeteroData object for this instance
        dat = pyg.data.HeteroData()

        # Add node features
        for node_type in x_dict:
            dat[node_type].x = x_dict[node_type]

        # Add edge indices
        for edge_type in edge_index_dict:
            src_type, relation, dst_type = edge_type
            dat[(src_type, relation, dst_type)].edge_index = edge_index_dict[edge_type]

        datas.append(dat)

    prot_idx = tdata.metadata()[0].index('protein')

    pidxs = [] 
    didxs = []
    probs = [] 
    for i,dat in enumerate(datas):
        print(f'predicting all DTIs... progress: {i}/{len(datas)}', end='\r')
        with torch.no_grad(): 
            dat = dat.to_homogeneous()
            out = model(dat.x.to(device), dat.edge_index.to(device), dat.edge_type.to(device))
            prot_mask = dat.node_type == prot_idx

            probs.append( out[prot_mask].detach() ) 
            pidxs.append( torch.arange(out[prot_mask].size(0))) 
            didxs.append( torch.ones(out[prot_mask].size(0)) * i)

    probs = torch.cat(probs).detach().cpu().numpy()
    pidxs = torch.cat(pidxs).detach().cpu().numpy()
    didxs = torch.cat(didxs).detach().cpu().numpy()

    didxs = torch.tensor(didxs, dtype=torch.long)

    dti_mask = train_triples['relation'] == 5
    train_heads = train_triples['head'][dti_mask]
    train_tails = train_triples['tail'][dti_mask]

    df = pd.DataFrame({'drug': didxs, 'protein': pidxs, 'prob': probs.ravel(), 'drug_name': np.array(data['node_name_dict']['drug'])[didxs], 'prot_name': np.array(data['node_name_dict']['protein'])[pidxs]})

    _train = pd.DataFrame({'drug': train_heads, 'protein': train_tails, 'train': True})
    df = df.merge(_train, on=['drug', 'protein'], how='left')

    _valid = pd.DataFrame({'drug': valid_triples['head'], 'protein': valid_triples['tail'], 'valid': True})
    df = df.merge(_valid, on=['drug', 'protein'], how='left')

    _test = pd.DataFrame({'drug': test_triples['head'], 'protein': test_triples['tail'], 'test': True})
    df = df.merge(_test, on=['drug', 'protein'], how='left')

    df = df.fillna(False)

    df = df.assign(negatives = ~(df['train'] | df['valid'] | df['test']))

    return df



def eval(loader, model, prot_idx, device='cpu', deterministic=True): 
    if deterministic: model.eval()
    pyhats = [] 
    pys = []
    for i,batch in enumerate(loader): 
        print(f'val batch [{i}/{len(loader)}]', end='\r')
        B = batch.y_protein.shape[0]
        with torch.no_grad(): 
            batch = batch.to_homogeneous()
            x = batch.x.to(device)
            y = batch.y.to(device)

            out = model(x, batch.edge_index.to(device), batch.edge_type.to(device))
            prot_mask = batch.node_type == prot_idx
            pout = out[prot_mask].view(B, -1)
            py = y[prot_mask].view(B, -1)
            pyhats.append(pout.detach().cpu())
            pys.append(py.detach().cpu())
            
    py = torch.cat(pys, dim=0)
    pout = torch.cat(pyhats, dim=0)
    ranks = (pout[py.nonzero(as_tuple=True)].view(-1,1) <= pout).sum(-1)
    mrr = (1/ranks).mean().item()
    topat10 = (ranks <= 10).float().mean().item()
    topat100 = (ranks <= 100).float().mean().item()
    auroc = roc_auc_score(py.detach().cpu().numpy().ravel(), pout.detach().cpu().numpy().ravel())

    return mrr, topat10, topat100, auroc



def train_gnn(config, kwargs=None): 
    '''
    '''

    device, data, train_triples, valid_triples, valid_neg_triples, test_triples, test_neg_triples  = device_and_data_loading(kwargs, return_test=True)

    tdata = process_graph(data)


    edge_index_dict = {key: tdata[key]['edge_index'] for key in tdata.metadata()[1]}
    num_node_dict = {key: tdata[key].num_nodes for key in tdata.metadata()[0]}

    train_dataset = TriplesDatasetGNN(train_triples, filter_to_relation=[5], edge_index_dict=edge_index_dict, channels=1, num_node_dict=num_node_dict)
    valid_dataset = TriplesDatasetGNN(valid_triples, filter_to_relation=[5], edge_index_dict=edge_index_dict, channels=1, num_node_dict=num_node_dict)

    train_loader = pyg.loader.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, 
                                         num_workers=config['num_workers'], persistent_workers=False)
    
    val_loader = pyg.loader.DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, 
                                       num_workers=config['num_workers'], persistent_workers=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    if config['nonlin'] == 'relu':
        nonlin = torch.nn.ReLU
    elif config['nonlin'] == 'elu':
        nonlin = torch.nn.ELU
    elif config['nonlin'] == 'gelu':
        nonlin = torch.nn.GELU
    elif config['nonlin'] == 'mish': 
        nonlin = torch.nn.Mish
    else:
        raise ValueError('nonlin not recognized')

    model = GNN(
                len(edge_index_dict), 
                channels=config['channels'], 
                layers=config['layers'], 
                dropout=config['dropout'], 
                heads=config['heads'], 
                bias=config['bias'], 
                edge_dim=config['edge_dim'], 
                norm_mode=config['norm_mode'], 
                norm_affine=False,
                nonlin=nonlin,
                checkpoint=config['checkpoint'], 
                conv=config['conv'],
                residual=config['residual']
            ).to(device)
    
    if kwargs.compile: 
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)
    
    optim = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    prot_idx = tdata.metadata()[0].index('protein')
    crit = torch.nn.BCELoss()

    stopper = EarlyStopper(patience=kwargs.patience, min_delta=0.)

    best_model = None
    best_topat10 = 0 

    metrics = {'mrr':[], 'top@10':[], 'top@100':[], 'auroc':[]}

    for epoch in range(config['n_epochs']): 
        tic = time.time() 

        losses = []
        model.train()
        for i,batch in enumerate(train_loader): 
            optim.zero_grad() 
            batch = batch.to_homogeneous()
            edge_type = batch.edge_type.to(device)
            edge_index = batch.edge_index.to(device)
            x = batch.x.to(device)
            y = batch.y.to(device)

            out = model(x, edge_index, edge_type)

            # subset on proteins only 
            prot_mask = batch.node_type == prot_idx

            pout = out[prot_mask]
            py = y[prot_mask]

            #loss = pout[py == 0].mean() - pout[py == 1].mean() # make true dtis big, and false dtis small
            loss = crit(pout.view(-1), py.view(-1))
            loss.backward()
            optim.step()
            losses.append(loss.item())
            auroc = roc_auc_score(py.detach().cpu().numpy(), pout.detach().cpu().numpy())
            print(f'train batch [{i}/{len(train_loader)}], loss: {loss.item():.2f}, auroc: {auroc:.2f}', end='\r')

        mrr, topat10, topat100, auroc = eval(val_loader, model, prot_idx, device=device, deterministic=True)
        metrics['mrr'].append(mrr)
        metrics['top@10'].append(topat10)
        metrics['top@100'].append(topat100)
        metrics['auroc'].append(auroc)

        toc = time.time() 
        print(f'-----------------> epoch: {epoch}, train loss: {np.mean(losses):.3f}, val auroc: {auroc:.3f}, val mrr: {mrr:.3f}, top@10: {topat10:.3f}, top@100: {topat100:.3f} [elapsed: {(toc-tic)/60:.1f}m]')
        
        if topat10 > best_topat10:
            best_topat10 = topat10
            best_model = model

        if epoch % kwargs.log_every == 0: 
            out_dict = {'model': model, 
                        'best_model': best_model,
                        'metrics': metrics,
                        'epoch': epoch, 
                        'train_loss': np.mean(losses), 
                        'val_mrr': mrr, 
                        'val_top@10': topat10, 
                        'val_top@100': topat100, 
                        'val_auroc': auroc}
            
            torch.save(out_dict, f'{kwargs.out}/results_{epoch}.pt')

        if stopper.step(-topat10): 
            print('early stopping at epoch: ', epoch)
            break

    df = predict_all(data, tdata, train_triples, valid_triples, test_triples, best_model, device)
    df.to_csv(f'{kwargs.out}/predictions.csv', index=False)


    