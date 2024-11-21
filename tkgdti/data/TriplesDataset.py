

import torch 
from torch.utils.data import Dataset
import sys 
import numpy as np
sys.path.append('../')

class TriplesDataset(Dataset):
    """"""

    def __init__(self, triples, filter_to_relation=None ):
        """

        """
        self.pos_heads = torch.tensor(triples['head'], dtype=torch.long)
        self.pos_tails = torch.tensor(triples['tail'], dtype=torch.long)
        self.pos_relations = torch.tensor(triples['relation'], dtype=torch.long)

        if filter_to_relation is not None: 
            idxs = torch.isin(self.pos_relations, torch.tensor(filter_to_relation, dtype=torch.long)).nonzero(as_tuple=True)[0]
            self.pos_heads = self.pos_heads[idxs]
            self.pos_tails = self.pos_tails[idxs]
            self.pos_relations = self.pos_relations[idxs]

    def __len__(self):
        return len(self.pos_heads)

    def __getitem__(self, idx):
        
        pos_head = self.pos_heads[idx].detach()
        pos_tail = self.pos_tails[idx].detach()
        pos_relation = self.pos_relations[idx].detach()

        return pos_head, pos_tail, pos_relation