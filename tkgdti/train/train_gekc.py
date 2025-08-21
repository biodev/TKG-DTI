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

    
    df = pd.DataFrame({'drug': heads, 'protein': tails, 'score': scores.ravel()})
                       #'drug_name': np.array(data['node_name_dict'][head_target])[heads], 
                       # 'prot_name': np.array(data['node_name_dict'][tail_target])[tails]})

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

    # use min/max to scale scores to make a "prob-like" score 
    df['prob'] = (df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min())

    return df


def predict_with_negatives(data, train_triples, valid_triples, test_triples, model, device, 
                          batch_size=10000, target_relint=None, head_target='drug', tail_target='protein',
                          verbose=True, partition='valid'):
    """
    Predict scores for positive edges and pre-sampled negative edges only.
    This is much more tractable for large datasets.
    """
    import os
    
    # Load negative samples from the data directory
    data_dir = data.get('data_dir', None)
    if data_dir is None:
        raise ValueError("Data directory not specified. Required for loading negative samples.")
    
    # Load negative samples
    neg_valid_path = os.path.join(data_dir, "neg_valid.pt")
    neg_test_path = os.path.join(data_dir, "neg_test.pt")
    
    if not os.path.exists(neg_valid_path) or not os.path.exists(neg_test_path):
        raise FileNotFoundError(f"Negative samples not found. Run negative sampling script first.")
    
    neg_valid = torch.load(neg_valid_path)
    neg_test = torch.load(neg_test_path)
    
    # Check if negative samples are properly generated (not None placeholders)
    if neg_valid['head'] is None or neg_test['head'] is None:
        raise ValueError("Negative samples contain None values. Please run the negative sampling script first.")
    
    # Select the appropriate negative set based on partition
    if partition == 'valid':
        neg_triples = neg_valid
        pos_triples = valid_triples
    elif partition == 'test':
        neg_triples = neg_test
        pos_triples = test_triples
    else:
        raise ValueError(f"Unsupported partition for negatives evaluation: {partition}")
    
    # Filter for target relation
    pos_mask = pos_triples['relation'] == target_relint
    pos_heads = pos_triples['head'][pos_mask]
    pos_tails = pos_triples['tail'][pos_mask]
    
    neg_mask = neg_triples['relation'] == target_relint
    neg_heads = neg_triples['head'][neg_mask]
    neg_tails = neg_triples['tail'][neg_mask]
    
    # Ensure all heads and tails are tensors for concatenation
    if not isinstance(pos_heads, torch.Tensor):
        pos_heads = torch.tensor(pos_heads, dtype=torch.long)
    if not isinstance(pos_tails, torch.Tensor):
        pos_tails = torch.tensor(pos_tails, dtype=torch.long)
    if not isinstance(neg_heads, torch.Tensor):
        neg_heads = torch.tensor(neg_heads, dtype=torch.long)
    if not isinstance(neg_tails, torch.Tensor):
        neg_tails = torch.tensor(neg_tails, dtype=torch.long)
    
    # Combine positive and negative edges
    all_heads = torch.cat([pos_heads, neg_heads])
    all_tails = torch.cat([pos_tails, neg_tails])
    all_relations = torch.tensor([target_relint] * len(all_heads), dtype=torch.long)
    
    # Predict scores
    scores = []
    with torch.no_grad():
        for idx in torch.split(torch.arange(len(all_heads)), batch_size):
            if verbose: 
                print(f'[predicting with negatives, progress: {idx[0].item()}/{len(all_heads)}]', end='\r')
            
            out = model(head=all_heads[idx].to(device), 
                       relation=all_relations[idx].to(device), 
                       tail=all_tails[idx].to(device))
            scores.append(out.detach().cpu())
    
    
    scores = torch.cat(scores)
    
    # Convert to numpy
    all_heads = all_heads.detach().cpu().numpy()
    all_tails = all_tails.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    
    # Create DataFrame
    df = pd.DataFrame({
        'drug': all_heads, 
        'protein': all_tails, 
        'score': scores.ravel()
    })
    
    # Add partition labels
    n_pos = len(pos_heads)
    df[partition] = [True] * n_pos + [False] * (len(df) - n_pos)
    df['negative_sample'] = [False] * n_pos + [True] * (len(df) - n_pos)
    
    # Normalize scores to probabilities
    df['prob'] = (df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min())
    
    return df


def eval(data, train_triples, valid_triples, test_triples, model, device, batch_size, 
         target_relint, head_target, tail_target, partition='valid', eval_method='all'):
    
    if eval_method == 'all':
        df = predict_all(data, train_triples, valid_triples, test_triples, model, device, batch_size=batch_size,
                        target_relint=target_relint, head_target=head_target, tail_target=tail_target, verbose=True)
    elif eval_method == 'negatives':
        df = predict_with_negatives(data, train_triples, valid_triples, test_triples, model, device, batch_size=batch_size,
                                   target_relint=target_relint, head_target=head_target, tail_target=tail_target, verbose=True, partition=partition)
    else:
        raise ValueError(f"Unknown eval_method: {eval_method}")

    _metrics = evaluate(df, partition=partition, verbose=True, method=eval_method)

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
    
    # Add data directory for loading negative samples
    data['data_dir'] = kwargs.data
    
    # Set default eval_method if not specified
    if not hasattr(kwargs, 'eval_method'):
        kwargs.eval_method = 'all'

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
                                                 tail_target,
                                                 eval_method=kwargs.eval_method)
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

    if kwargs.eval_method == 'all':
        df = predict_all(data, train_triples, valid_triples, test_triples, best_model, device, batch_size=kwargs.batch_size,
                         target_relint=target_relint, head_target=head_target, tail_target=tail_target)
    elif kwargs.eval_method == 'negatives':
        # For negatives method, we need to create separate dataframes for validation and test
        df_valid = predict_with_negatives(data, train_triples, valid_triples, test_triples, best_model, device, 
                                         batch_size=kwargs.batch_size, target_relint=target_relint, 
                                         head_target=head_target, tail_target=tail_target, partition='valid')
        df_test = predict_with_negatives(data, train_triples, valid_triples, test_triples, best_model, device, 
                                        batch_size=kwargs.batch_size, target_relint=target_relint, 
                                        head_target=head_target, tail_target=tail_target, partition='test')
        # For compatibility with the existing workflow, combine them for saving
        df = pd.concat([df_valid, df_test], ignore_index=True)
    else:
        raise ValueError(f"Unknown eval_method: {kwargs.eval_method}")
    
    df.to_csv(f'{kwargs.out}/predictions.csv', index=False)

    #val metrics 
    if kwargs.eval_method == 'all':
        val_metrics = evaluate(df, partition='valid', method=kwargs.eval_method)
    else:
        val_metrics = evaluate(df_valid, partition='valid', method=kwargs.eval_method)

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
    if kwargs.eval_method == 'all':
        test_metrics = evaluate(df, partition='test', method=kwargs.eval_method)
    else:
        test_metrics = evaluate(df_test, partition='test', method=kwargs.eval_method)

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