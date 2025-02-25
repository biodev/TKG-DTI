
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
import uuid 
import os 
from tkgdti.eval.evaluate import evaluate
import copy

# BUG: "RuntimeError: received 0 items of ancdata" (fix: https://stackoverflow.com/questions/71642653/how-to-resolve-the-error-runtimeerror-received-0-items-of-ancdata)
#import resource
#rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
#resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def predict_all(data, tdata, target_relation, target_relint, train_triples, valid_triples, test_triples, model, device): 

    head_target, rel_target, tail_target = target_relation

    model.eval() 

    edge_index_dict = {key: tdata[key]['edge_index'] for key in tdata.metadata()[1]}
    num_node_dict = {key: tdata[key].num_nodes for key in tdata.metadata()[0]}

    datas = []
    for i in range(data['num_nodes_dict'][head_target]):
        head = i 
        x_dict = {node:torch.zeros((num_nodes, 1), dtype=torch.float32) for node, num_nodes in num_node_dict.items()}

        x_dict[head_target][head] = torch.ones((1,), dtype=torch.float32)

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

    prot_idx = tdata.metadata()[0].index(tail_target)

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

    probs = torch.cat(probs, dim=0).detach().cpu().numpy()
    pidxs = torch.cat(pidxs, dim=0).detach().cpu().numpy().astype(int)
    didxs = torch.cat(didxs, dim=0).detach().cpu().numpy().astype(int)

    #didxs = torch.tensor(didxs, dtype=torch.long)

    dti_mask = train_triples['relation'] == target_relint
    train_heads = train_triples['head'][dti_mask]
    train_tails = train_triples['tail'][dti_mask]

    df = pd.DataFrame({'drug': didxs, 'protein': pidxs, 
                       'score': probs.ravel(), 
                       'prob': probs.ravel(),
                       'drug_name': np.array(data['node_name_dict'][head_target])[didxs], 
                       'prot_name': np.array(data['node_name_dict'][tail_target])[pidxs]})

    train_links = set(zip(train_heads, train_tails))
    df['train'] = [True if (h,t) in train_links else False for h,t in zip(didxs, pidxs)]

    valid_links = set(zip(valid_triples['head'], valid_triples['tail']))
    df['valid'] = [True if (h,t) in valid_links else False for h,t in zip(didxs, pidxs)]

    test_links = set(zip(test_triples['head'], test_triples['tail']))
    df['test'] = [True if (h,t) in test_links else False for h,t in zip(didxs, pidxs)]

    # BUG FIX: https://stackoverflow.com/questions/77900971/pandas-futurewarning-downcasting-object-dtype-arrays-on-fillna-ffill-bfill
    df = df.infer_objects(copy=False).fillna(False)

    df = df.assign(negatives = ~(df['train'] | df['valid'] | df['test']))

    return df



def eval(data, tdata, target_relation, target_relint, train_triples, valid_triples, test_triples, model, device='cpu', partition='valid'): 

    df = predict_all(data, tdata, target_relation, target_relint, train_triples, valid_triples, test_triples, model, device)
    
    _metrics = evaluate(df, partition=partition, verbose=False)

    mrr = _metrics['MRR']
    topat10 = _metrics['Top10']
    topat100 = _metrics['Top100']
    auroc = _metrics['avg_AUC']

    return mrr, topat10, topat100, auroc




def train_gnn(config, kwargs=None): 
    '''
    '''
    uid = uuid.uuid4()
    print(f'uid: {uid}')
    os.makedirs(kwargs.out + '/' + str(uid), exist_ok=True)
    root_out = kwargs.out
    kwargs.out = kwargs.out + '/' + str(uid)

    device, data, train_triples, valid_triples, valid_neg_triples, test_triples, test_neg_triples  = device_and_data_loading(kwargs, return_test=True)

    tdata = process_graph(data, heteroA=kwargs.heteroA)

    rel2int = {k:v[0] for k,v in data.edge_reltype.items()}
    target_relint = rel2int[kwargs.target_relation]
    target_relation = kwargs.target_relation
    if type(target_relint) is not int: target_relint = target_relint.item()
    head_target, _ , tail_target = kwargs.target_relation
    
    print() 
    print('---------------------------------')
    print('target relation: ', kwargs.target_relation)
    print('target relation int: ', target_relint)
    print('---------------------------------')
    print()

    edge_index_dict = {key: tdata[key]['edge_index'] for key in tdata.metadata()[1]}
    num_node_dict = {key: tdata[key].num_nodes for key in tdata.metadata()[0]}

    train_dataset = TriplesDatasetGNN(train_triples, filter_to_relation=[target_relint], edge_index_dict=edge_index_dict, channels=1, target_relation=target_relation, num_node_dict=num_node_dict)
    valid_dataset = TriplesDatasetGNN(valid_triples, filter_to_relation=[target_relint], edge_index_dict=edge_index_dict, channels=1, target_relation=target_relation, num_node_dict=num_node_dict)

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
                nonlin=nonlin,
                checkpoint=config['checkpoint'], 
                conv=config['conv'],
                residual=config['residual'],
                norm=config['norm'],
            ).to(device)
    
    if kwargs.compile: 
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)
    
    optim = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    prot_idx = tdata.metadata()[0].index(tail_target)
    crit = torch.nn.BCELoss()

    stopper = EarlyStopper(patience=kwargs.patience, min_delta=0.)

    best_model_state_dict = None
    best_metric = -np.inf 

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

            loss = crit(pout.view(-1), py.view(-1))
            loss.backward()
            optim.step()
            losses.append(loss.item())
            auroc = roc_auc_score(py.detach().cpu().numpy(), pout.detach().cpu().numpy())
            print(f'train batch [{i}/{len(train_loader)}], loss: {loss.item():.2f}, auroc: {auroc:.2f}', end='\r')

        if epoch % kwargs.log_every == 0: 
            mrr, topat10, topat100, auroc = eval(data, tdata, target_relation, target_relint, train_triples, valid_triples, test_triples, model, device=device, partition='valid')
            metrics['mrr'].append(mrr)
            metrics['top@10'].append(topat10)
            metrics['top@100'].append(topat100)
            metrics['auroc'].append(auroc)

            if best_metric <= metrics[kwargs.target_metric][-1]:
                best_metric = metrics[kwargs.target_metric][-1]
                best_model_state_dict = copy.deepcopy(model.state_dict())

            out_dict = {'best_state_dict': best_model_state_dict,
                        'args': kwargs,
                        'metrics': metrics,
                        'epoch': epoch, 
                        'train_loss': np.mean(losses), 
                        'val_mrr': mrr, 
                        'val_top@10': topat10, 
                        'val_top@100': topat100, 
                        'val_auroc': auroc}
            
            torch.save(out_dict, f'{kwargs.out}/results.pt')

            if stopper.step(-metrics[kwargs.target_metric][-1]): 
                print('early stopping at epoch: ', epoch)
                break

            toc = time.time() 
            print(f'-----------------> epoch: {epoch}, train loss: {np.mean(losses):.3f}, val auroc: {auroc:.3f}, val mrr: {mrr:.3f}, top@10: {topat10:.3f}, top@100: {topat100:.3f} [elapsed: {(toc-tic)/60:.1f}m]')
    

    # load best model 
    best_model = model 
    best_model.load_state_dict(best_model_state_dict)
    best_model.eval() 
    torch.save(best_model, f'{kwargs.out}/best_model.pt')

    # make predictions and evaluations 

    df = predict_all(data, tdata, target_relation, target_relint, train_triples, valid_triples, test_triples, best_model, device)
    df.to_csv(f'{kwargs.out}/predictions.csv', index=False)

    #val metrics 
    val_metrics = evaluate(df, partition='valid') 

    print('valid set metrics:')
    print(val_metrics)

    val_metrics['uid'] = uid
    val_metrics = {**config, **val_metrics}

    torch.save(val_metrics, f'{kwargs.out}/valid_metrics.pt')

    metrics_df = pd.DataFrame(val_metrics, index=[0])

    try: 
        if os.path.exists(f'{root_out}/valid_metrics.csv'): 
            metrics_df.to_csv(f'{root_out}/valid_metrics.csv', mode='a', header=False, index=False)
        else:
            metrics_df.to_csv(f'{root_out}/valid_metrics.csv', index=False)
    except:
        raise 
    
    # test metrics 
    test_metrics = evaluate(df, partition='test') 

    print('test set metrics:')
    print(test_metrics)

    test_metrics['uid'] = uid
    test_metrics = {**config, **test_metrics}

    torch.save(test_metrics, f'{kwargs.out}/test_metrics.pt')

    metrics_df = pd.DataFrame(test_metrics, index=[0])

    try: 
        if os.path.exists(f'{root_out}/test_metrics.csv'): 
            metrics_df.to_csv(f'{root_out}/test_metrics.csv', mode='a', header=False, index=False)
        else:
            metrics_df.to_csv(f'{root_out}/test_metrics.csv', index=False)
    except:
        raise 

    with open(f'{kwargs.out}/completed.txt', 'w') as f: 
        f.write(f'{uid}\n')


    return uid, val_metrics, test_metrics