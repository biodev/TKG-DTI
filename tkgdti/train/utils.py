import torch
from tkgdti.data.TriplesDataset import TriplesDataset
import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*weights_only.*")

class NewTriplesHolder(): 

    def __init__(self, num_nodes_dict, triples, rel2type): 
        all_heads = [] 
        all_tails = [] 
        all_relations = []
        edge_dict = {rt[1]:set() for rt in rel2type.values()}
        for h,r,t in zip(triples['head'], triples['relation'], triples['tail']): 
            edge_dict[rel2type[r][1]].add((h,t))

        print('generating new edges...')
        for rel in rel2type.keys(): 
            print(rel, end='\r')
            ht, rt, tt = rel2type[rel]
            for i in range(num_nodes_dict[ht]): 
                for j in range(num_nodes_dict[tt]): 

                    if (i,j) not in edge_dict[rt]:
                        all_heads.append(i)
                        all_tails.append(j)
                        all_relations.append(rel)

        self.new_triples = {'head':np.array(all_heads), 'tail':np.array(all_tails), 'relation':np.array(all_relations)}
         


    def predict_new(self, model, device, topk=1000, verbose=True, batch_size=100000, num_workers=10): 

        model.eval()
        new_dataset = TriplesDataset(self.new_triples)
        new_dataloader = torch.utils.data.DataLoader(new_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)

        scores = []
        for i, (head, tail, relation, _) in enumerate(new_dataloader):
            print(f'progress: {i}/{len(new_dataloader)}', end='\r')

        with torch.no_grad(): 
            ll = -model(head=head.to(device), relation=relation.to(device), tail=tail.to(device)) 
            #score = model.score(head_idx=head.to(device), relation_idx=relation.to(device), tail_idx=tail.to(device), headtype='drug', tailtype='protein')
            scores.append( ll.detach().cpu().numpy())

        scores = np.concatenate(scores)

        idx = np.argsort(scores)[::-1][:topk]

        hit_triples = {'head':self.new_triples['head'][idx], 'tail':self.new_triples['tail'][idx], 'relation':self.new_triples['relation'][idx]}
        self.new_triples['head'] = np.delete(self.new_triples['head'], idx)
        self.new_triples['tail'] = np.delete(self.new_triples['tail'], idx)
        self.new_triples['relation'] = np.delete(self.new_triples['relation'], idx)

        return hit_triples

def device_and_data_loading(kwargs, return_test=False): 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if kwargs.verbose: print('device:', device)

    # load data 
    data = torch.load(f'{kwargs.data}/Data.pt')
    train_triples = torch.load(f'{kwargs.data}/pos_train.pt' )

    valid_triples = torch.load(f'{kwargs.data}/pos_valid.pt')
    valid_neg_triples = torch.load(f'{kwargs.data}/neg_valid.pt')

    if return_test: 
        test_triples = torch.load(f'{kwargs.data}/pos_test.pt')
        test_neg_triples = torch.load(f'{kwargs.data}/neg_test.pt')

    # filter utilities 
    if kwargs.remove_relation_idx is not None:

        # filter triples 
        mask = train_triples['relation'] != kwargs.remove_relation_idx
        train_triples = {k:v[mask] for k,v in train_triples.items()}

        # filter data edge index 
        rel2int = {k:v[0] for k,v in data.edge_reltype.items()} 
        int2rel = {v:k for k,v in rel2int.items()}

        print() 
        print('-----------------------------------------------------------------------------------')
        print('WARNING: removing relation:', int2rel[kwargs.remove_relation_idx])
        print('-----------------------------------------------------------------------------------')
        print() 

        rel = int2rel[kwargs.remove_relation_idx]
        data.edge_index_dict[rel] = torch.tensor([], dtype=torch.long)

    if return_test: 
        return device, data, train_triples, valid_triples, valid_neg_triples, test_triples, test_neg_triples
    
    else: 
        return device, data, train_triples, valid_triples, valid_neg_triples
    
def get_optim(optim): 

    if optim == 'adam': 
        return torch.optim.Adam 
    elif optim == 'adan': 
        try: 
            from adan import Adan
        except: 
            raise ImportError('adan not installed. Please install adan (see: https://github.com/sail-sg/Adan)')
        return Adan
    elif optim == 'sgd': 
        return torch.optim.SGD 
    elif optim == 'rmsprop': 
        return torch.optim.RMSprop
    else:
        raise ValueError(f'unrecognized optim argument: {optim}')
    

def training_inits(kwargs, config, model): 

    if kwargs.verbose and (kwargs.target_relations is not None):
        print('target relations:')
        for r in kwargs.target_relations: 
            if hasattr(model, 'rel2type'): 
                print('\t', model.rel2type[r])
            else: 
                print('\t', model.model.rel2type[r])
        print()
    
    if config['optim'] == 'adam': 
        optim = torch.optim.Adam
    elif config['optim'] == 'adan': 
        try: 
            from adan import Adan
        except: 
            raise ImportError('adan not installed. Please install adan (see: https://github.com/sail-sg/Adan)')
        optim = Adan
    elif config['optim'] == 'adagrad': 
        optim = torch.optim.Adagrad
    elif config['optim'] == 'sgd': 
        optim = torch.optim.SGD
    else: 
        raise ValueError(f'Unrecognized `optim`: {config["optim"]}')
    
    optim = optim(model.parameters(), lr=config['lr'], weight_decay=config['wd'])

    if config['lr_scheduler']: 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=0.1, patience=10) 
    else: 
        scheduler = None

    if kwargs.verbose: 
        n_params = sum([p.numel() for p in model.parameters()])
        print('# model params:', n_params)

    return optim, scheduler
