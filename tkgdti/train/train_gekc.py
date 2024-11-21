import torch 
import numpy as np
from tkgdti.models.ComplEx2 import ComplEx2
from tkgdti.data.TriplesDataset import TriplesDataset
from tkgdti.eval.utils import predict_all_dtis, assign_proba
from tkgdti.eval.Evaluator import Evaluator
from tkgdti.eval.checkpoint import checkpoint, load_checkpoint
from tkgdti.train.utils import device_and_data_loading, training_inits, NewTriplesHolder
from tkgdti.models.init import init_model
import os
from tkgdti.eval.Logger import Logger
import time 
from tkgdti.train.EarlyStopper import EarlyStopper

def train_gekc(config, kwargs=None): 
    '''
    training Tractable KBC Circuit as described in: How to Turn Your Knowledge Graph Embeddings into Generative Models 
    https://proceedings.neurips.cc/paper_files/paper/2023/file/f4b768188be63b8d2680a46934fd295a-Paper-Conference.pdf
    '''
    device, data, train_triples, valid_triples, valid_neg_triples, test_triples, test_neg_triples = device_and_data_loading(kwargs, return_test=True)

    model = model = ComplEx2(data            = data, 
                        hidden_channels  = config['channels'], 
                        dropout          = config['dropout']).to(device)

    optim, scheduler = training_inits(kwargs, config, model)

    dataset = TriplesDataset(train_triples)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             num_workers=kwargs.num_workers, 
                                             batch_size=config['batch_size'], 
                                             shuffle=True, 
                                             persistent_workers=True)

    logger = Logger(kwargs, 
                    config,
                    root=kwargs.out, 
                    target_relation=kwargs.target_relations, 
                    triples=(valid_triples, valid_neg_triples))
    
    stopper = EarlyStopper(kwargs.patience)

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
            if kwargs.verbose: print(f'Epoch: {epoch} [iter: {i}/{len(dataloader)}] --> train loss: {loss.item():.3f}', end='\r')

        if (epoch % kwargs.log_every) == 0: 
            if tic is not None: 
                elapsed = time.time() - tic
            else: 
                elapsed = -1

            tic = time.time()

            logger.log(model)
            if scheduler is not None: scheduler.step(logger.get_last(kwargs.metric))
            
            if kwargs.verbose: print(f'Epoch: {epoch} -{"-"*15}> mean loss (train): {tot_loss/(i+1):.3f} || (valid) MRR: {logger.get_last('mrr'):.4f} || top@(1,3,10): ({logger.get_last('top@1'):.3f},{logger.get_last('top@3'):.3f},{logger.get_last('top@10'):.3f}) || AUROC: {logger.get_last('auroc'):.4f} || AUPR: {logger.get_last('aupr'):.4f} || elapsed: {elapsed:.3f} sec')

            if stopper.step(-logger.get_last('mrr')): 
                print('early stopping @ epoch ', epoch)
                break

    # evaluating on test set
    eval = Evaluator(test_triples, test_neg_triples, target_relations=kwargs.target_relations)
    model = logger.load_best_model(model).to(device)
    mrr, _, top_at_1, top_at_3, top_at_10 = eval.mrr(model, device)
    aurocs, auprs = eval.auroc(model, device) 

    test_dict = {'mrr':mrr,
                 'top@1':top_at_1,
                 'top@3':top_at_3,
                 'top@10':top_at_10,
                 'auroc':aurocs,
                 'aupr':auprs}
    
    os.makedirs(kwargs.out + '/test/', exist_ok=True)
    torch.save(test_dict, kwargs.out + '/test/' + str(logger.uid) + '.pt')

    # predict all edges 
    triples = (train_triples, valid_triples, test_triples) 
    res = predict_all_dtis(model, triples, batch_size=25000)
    res = assign_proba(res)
    os.makedirs(kwargs.out + '/DTIs/', exist_ok=True)
    torch.save(res, kwargs.out + '/DTIs/' + str(logger.uid) + '.pt')

    return model
                             
                