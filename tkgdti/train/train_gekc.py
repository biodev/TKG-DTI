import torch 
import numpy as np
from tkgdti.models.ComplEx2 import ComplEx2
from tkgdti.data.TriplesDataset import TriplesDataset
from tkgdti.train.utils import device_and_data_loading, training_inits
import os
import time 
from tkgdti.train.EarlyStopper import EarlyStopper
from sklearn.metrics import roc_auc_score
import pandas as pd
import uuid
from tkgdti.eval.evaluate import evaluate
import copy

# BUG FIX: https://stackoverflow.com/questions/77900971/pandas-futurewarning-downcasting-object-dtype-arrays-on-fillna-ffill-bfill
pd.set_option('future.no_silent_downcasting', True)

def predict_all(data, train_triples, valid_triples, test_triples, model, device, 
                batch_size=10000, target_relint=None, head_target='drug', tail_target='protein',
                verbose=True): 

    ndrugs = data['num_nodes_dict'][head_target]
    nprots = data['num_nodes_dict'][tail_target]

    heads = [] 
    tails = [] 

    _is = range(ndrugs)
    _js = np.arange(nprots)

    for i in _is: 
        for j in _js: 
            heads.append(i)
            tails.append(j)
    heads = torch.tensor(heads, dtype=torch.long)
    tails = torch.tensor(tails, dtype=torch.long)
    relations = torch.tensor([target_relint]*len(heads), dtype=torch.long)

    scores = [] 
    with torch.no_grad():
        for idx in torch.split(torch.arange(len(heads)), batch_size):
            if verbose: print(f'[progress: {idx[0].item()}/{len(heads)}]', end='\r')
            out = model(head=heads[idx].to(device), 
                        relation=relations[idx].to(device), 
                        tail=tails[idx].to(device))
            scores.append(out.detach().cpu())

    scores = torch.cat(scores)

    heads = heads.detach().cpu().numpy()
    tails = tails.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    
    df = pd.DataFrame({'drug': heads, 'protein': tails, 'score': scores.ravel(), 
                       'drug_name': np.array(data['node_name_dict'][head_target])[heads], 
                       'prot_name': np.array(data['node_name_dict'][tail_target])[tails],
                       'drug_name': np.array(data['node_name_dict'][head_target])[heads], 
                       'prot_name': np.array(data['node_name_dict'][tail_target])[tails]})

    dti_mask = train_triples['relation'] == target_relint
    train_heads = train_triples['head'][dti_mask]
    train_tails = train_triples['tail'][dti_mask]
    train_links = set(zip(train_heads, train_tails))
    df['train'] = [True if (h,t) in train_links else False for h,t in zip(heads, tails)]

    valid_links = set(zip(valid_triples['head'], valid_triples['tail']))
    df['valid'] = [True if (h,t) in valid_links else False for h,t in zip(heads, tails)]

    test_links = set(zip(test_triples['head'], test_triples['tail']))
    df['test'] = [True if (h,t) in test_links else False for h,t in zip(heads, tails)]

    # BUG FIX: https://stackoverflow.com/questions/77900971/pandas-futurewarning-downcasting-object-dtype-arrays-on-fillna-ffill-bfill
    df = df.infer_objects(copy=False).fillna(False)

    df = df.assign(negatives = ~(df['train'] | df['valid'] | df['test']))

    # use min/max to scale scores to prob 
    df['prob'] = (df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min())

    return df


def eval(data, train_triples, valid_triples, test_triples, model, device, batch_size, 
         target_relint, head_target, tail_target, partition='valid'):
    
    df = predict_all(data, train_triples, valid_triples, test_triples, model, device, batch_size=batch_size,
                    target_relint=target_relint, head_target=head_target, tail_target=tail_target, verbose=True)

    _metrics = evaluate(df, partition=partition, verbose=False)

    mrr = _metrics['MRR']
    topat10 = _metrics['Top10']
    topat100 = _metrics['Top100']
    auroc = _metrics['avg_AUC']

    return mrr, topat10, topat100, auroc

def train_gekc(config, kwargs=None): 
    '''
    training Tractable KBC Circuit as described in: How to Turn Your Knowledge Graph Embeddings into Generative Models 
    https://proceedings.neurips.cc/paper_files/paper/2023/file/f4b768188be63b8d2680a46934fd295a-Paper-Conference.pdf
    '''

    uid = uuid.uuid4()
    print(f'uid: {uid}')
    os.makedirs(kwargs.out + '/' + str(uid), exist_ok=True)
    root_out = kwargs.out
    kwargs.out = kwargs.out + '/' + str(uid)

    device, data, train_triples, valid_triples, valid_neg_triples, test_triples, test_neg_triples = device_and_data_loading(kwargs, return_test=True)

    if kwargs.use_cpu: 
        device = 'cpu'

    rel2int = {k:v[0] for k,v in data.edge_reltype.items()}
    target_relint = rel2int[kwargs.target_relation]
    if type(target_relint) is not int: target_relint = target_relint.item()
    head_target, _ , tail_target = kwargs.target_relation
    
    print() 
    print('---------------------------------')
    print('target relation: ', kwargs.target_relation)
    print('target relation int: ', target_relint)
    print('---------------------------------')
    print()

    model = ComplEx2(data            = data, 
                        hidden_channels  = config['channels'], 
                        dropout          = config['dropout']).to(device)

    optim, scheduler = training_inits(kwargs, config, model)

    dataset = TriplesDataset(train_triples)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             num_workers=kwargs.num_workers, 
                                             batch_size=config['batch_size'], 
                                             shuffle=True, 
                                             persistent_workers=True)
    
    valid_dataset = TriplesDataset(valid_triples)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, 
                                               num_workers=kwargs.num_workers, 
                                               batch_size=config['batch_size'], 
                                               shuffle=False, 
                                               persistent_workers=True)
    
    stopper = EarlyStopper(kwargs.patience, min_delta=0)

    best_model_state_dict = None
    best_metric = -np.inf

    metrics = {'mrr':[], 'top@10':[], 'top@100':[], 'auroc':[]}

    tic = None
    for epoch in range(config['n_epochs']): 

        tot_loss = 0
        model.train()
        for i, (pos_head, pos_tail, pos_relation) in enumerate(dataloader):
            optim.zero_grad() 
            
            nll = -model.forward(head        = pos_head.to(device), 
                                relation    = pos_relation.to(device),
                                tail        = pos_tail.to(device)).squeeze(-1)
            
            loss = nll.mean()
            loss.backward() 
            optim.step()

            tot_loss += loss.detach().item()
            print(f'Epoch: {epoch} [iter: {i}/{len(dataloader)}] --> train loss: {loss.item():.3f}', end='\r')

        if (epoch % kwargs.log_every) == 0: 
            if tic is not None: 
                elapsed = time.time() - tic
            else: 
                elapsed = -1

            tic = time.time()

            #mrr, topat10, topat100, auroc = eval(valid_loader, model, device=device, tail_target=tail_target)
            mrr, topat10, topat100, auroc = eval(data, 
                                                 train_triples, 
                                                 valid_triples, 
                                                 test_triples, 
                                                 model, 
                                                 device, 
                                                 kwargs.batch_size, 
                                                 target_relint, 
                                                 head_target, 
                                                 tail_target)
            metrics['mrr'].append(mrr)
            metrics['top@10'].append(topat10)
            metrics['top@100'].append(topat100)
            metrics['auroc'].append(auroc)

            if best_metric <= metrics[kwargs.target_metric][-1]:
                best_metric = metrics[kwargs.target_metric][-1]
                best_model_state_dict = copy.deepcopy(model.state_dict())

            print(f'Epoch: {epoch} -{"-"*15}> mean loss (train): {tot_loss/(i+1):.3f} || (valid) MRR: {mrr:.4f} || top@(10, 100): ({topat10:.3f},{topat100:.3f}) || AUROC: {auroc:.4f} || elapsed: {elapsed:.3f} sec')

            out_dict = {'best_model': best_model_state_dict,
                        'args': kwargs,
                        'config': config,
                        'metrics': metrics,
                        'epoch': epoch, 
                        'train_loss': tot_loss/(i+1), 
                        'val_mrr': mrr, 
                        'val_top@10': topat10, 
                        'val_top@100': topat100, 
                        'val_auroc': auroc}
            
            torch.save(out_dict, f'{kwargs.out}/results.pt')

            if stopper.step(-metrics[kwargs.target_metric][-1]): 
                print('early stopping @ epoch ', epoch)
                break
    
    best_model = model 
    best_model.load_state_dict(best_model_state_dict)

    df = predict_all(data, train_triples, valid_triples, test_triples, best_model, device, batch_size=kwargs.batch_size,
                     target_relint=target_relint, head_target=head_target, tail_target=tail_target)
    
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







'''
deprecated 


def eval(loader, model, device='cpu', nneg=10000, tail_target='protein'): 
    
    ranks = [] 
    y = [] 
    yhat = []
    for i, (pos_head, pos_tail, pos_relation) in enumerate(loader):

        print(f'val batch [{i}/{len(loader)}]', end='\r')
        with torch.no_grad(): 

            pos_head = pos_head.to(device)
            pos_tail = pos_tail.to(device)
            pos_relation = pos_relation.to(device)

            pos_scores = model.forward(head=pos_head, relation=pos_relation, tail=pos_tail).squeeze(-1)

            pos_head_ = pos_head.view(-1, 1).expand(-1, nneg).contiguous().view(-1)
            pos_relation_ = pos_relation.view(-1, 1).expand(-1, nneg).contiguous().view(-1)
            neg_tail = torch.randint(0, model.data['num_nodes_dict'][tail_target], (pos_head.size(0), nneg)).to(device).view(-1)
            neg_scores = model.forward(head=pos_head_, relation=pos_relation_, tail=neg_tail).view(-1, nneg)

            ranks.append( (neg_scores >= pos_scores.view(-1, 1)).sum(-1) + 1 )
            y.append( torch.cat((torch.ones_like(pos_scores), torch.zeros_like(neg_scores.view(-1))), dim=-1) )
            yhat.append( torch.cat([pos_scores, neg_scores.view(-1)], dim=-1) )
            
    y = torch.cat(y, dim=0)
    yhat = torch.cat(yhat, dim=0)
    ranks = torch.cat(ranks, dim=0)
    mrr = (1/ranks.float()).mean().item()
    topat10 = (ranks <= 10).float().mean().item()
    topat100 = (ranks <= 100).float().mean().item()
    auroc = roc_auc_score(y.detach().cpu().numpy(), yhat.detach().cpu().numpy())

    return mrr, topat10, topat100, auroc

'''