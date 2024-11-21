'''
Heterogenous version of (my original implementation; distinct from paper implementation)
How to Turn Your Knowledge Graph Embeddings into Generative Models (https://arxiv.org/pdf/2305.15944) 
'''
import numpy as np
import torch 
import torch_geometric as pyg
import time
import sys 


def make_relgroups(rel2type): 
    # we want to get the rel types (ints) groups that all connect the same head-tail types
    # thought: this allows us to learn prob distribution across multiple rel types which may contain useful information
    relgroups = {}
    for relint, reltup in rel2type.items(): 
        h,r,t = reltup 
        key = h + '->' + t 
        if key in relgroups: 
            relgroups[key].append(relint)
        else: 
            relgroups[key] = [relint]

    relgroups = {k:torch.tensor(v, dtype=torch.long) for k,v in relgroups.items()}

    return relgroups

class ComplEx2(torch.nn.Module): 

    def __init__(self, data, hidden_channels=512, scale_grad_by_freq=False, dtype=torch.float32, dropout=0.): 

        super().__init__()

        self.data = data 
        self.rel2type = {v.item(0):k for k,v in data['edge_reltype'].items()}
        self.relations = self.rel2type.keys()
        self.hidden_channels = hidden_channels
        self.dtype = dtype

        embed_kwargs = {'max_norm':None, # this causes an error when not None by in-place modifications 
                        'scale_grad_by_freq':scale_grad_by_freq,
                        'dtype':self.dtype}

        # these now represent categorical logit embeddings
        self.head_embedding_real_dict = torch.nn.ModuleDict({nodetype:torch.nn.Embedding(num_nodes, embedding_dim=hidden_channels, **embed_kwargs) for nodetype, num_nodes in self.data['num_nodes_dict'].items()})
        self.head_embedding_imag_dict = torch.nn.ModuleDict({nodetype:torch.nn.Embedding(num_nodes, embedding_dim=hidden_channels, **embed_kwargs) for nodetype, num_nodes in self.data['num_nodes_dict'].items()})
        self.tail_embedding_real_dict = torch.nn.ModuleDict({nodetype:torch.nn.Embedding(num_nodes, embedding_dim=hidden_channels, **embed_kwargs) for nodetype, num_nodes in self.data['num_nodes_dict'].items()})
        self.tail_embedding_imag_dict = torch.nn.ModuleDict({nodetype:torch.nn.Embedding(num_nodes, embedding_dim=hidden_channels, **embed_kwargs) for nodetype, num_nodes in self.data['num_nodes_dict'].items()})
        self.relation_real_embedding = torch.nn.Embedding(len(self.data['edge_index_dict']), embedding_dim=hidden_channels, **embed_kwargs)
        self.relation_imag_embedding = torch.nn.Embedding(len(self.data['edge_index_dict']), embedding_dim=hidden_channels, **embed_kwargs)
        
        self.nodetype2int = {nodetype:torch.tensor([i],dtype=torch.long) for i,nodetype in enumerate(data['num_nodes_dict'].keys())}

        # weight initialization
        for key in self.head_embedding_real_dict: 
            self.init_params(self.head_embedding_real_dict[key].weight.data)
            self.init_params(self.head_embedding_imag_dict[key].weight.data)
            self.init_params(self.tail_embedding_real_dict[key].weight.data)
            self.init_params(self.tail_embedding_imag_dict[key].weight.data)
        self.init_params(self.relation_real_embedding.weight.data)
        self.init_params(self.relation_imag_embedding.weight.data)

        # Trying to emulate https://github.com/april-tools/gekcs/blob/main/src/kbc/gekc_models.py ; line 579 
        # number of consistent triples, in this case edge type specific  
        # I think this justifies the use of log1p
        self.eps_dict = {(h,r,t):1/(self.data['num_nodes_dict'][h] * self.data['num_nodes_dict'][t]) for (h,r,t) in self.data['edge_index_dict'].keys()}

        self.dropout = dropout 

    def init_params(self, tensor, init_loc=0, init_scale=10e-3): 
        # https://github.com/april-tools/gekcs/blob/main/src/kbc/distributions.py ## line 60 
        # This initial outputs of ComplEx^2 will be approx. normally distributed and centered (in log-space)
        init_loc = np.log(tensor.shape[-1]) / 3.0 + 0.5 * (init_scale ** 2)
        t = torch.exp(torch.randn(*tensor.shape, dtype=self.dtype) * init_scale - init_loc)
        tensor.copy_(t.float())

    def partition_function(self, headtype:str, relint:int, tailtype:str) -> torch.tensor: 
        #Squared ComplEx partition function as described in "How to Turn our KGE in generative models" 
        #Slightly different version than used in paper, since we are conditioning on relation. Faster and less memory. 
        #Subject (real) ~ Sr
        Sr = self.head_embedding_real_dict[headtype].weight
        Si = self.head_embedding_imag_dict[headtype].weight
        Pr = self.relation_real_embedding(relint).view(1,-1)
        Pi = self.relation_imag_embedding(relint).view(1,-1)
        Or = self.tail_embedding_real_dict[tailtype].weight
        Oi = self.tail_embedding_imag_dict[tailtype].weight

        if self.do_node_real is not None: 
            Sr = Sr*self.do_node_real[headtype]
            Si = Si*self.do_node_imag[headtype]
            Or = Or*self.do_node_real[tailtype]
            Oi = Oi*self.do_node_imag[tailtype]

        SrSr = Sr.T@Sr  ;  OrOr = Or.T@Or
        SiSi = Si.T@Si  ;  OiOi = Oi.T@Oi 
        SrSi = Sr.T@Si  ;  OrOi = Or.T@Oi 
        SiSr = Si.T@Sr  ;  OiOr = Oi.T@Or
        # SrSi =/= SiSr

        A2 = (Pr @ (SrSr * OrOr) @ Pr.T).sum()
        B2 = (Pr @ (SiSi * OiOi) @ Pr.T).sum()
        C2 = (Pi @ (SrSr * OiOi) @ Pi.T).sum()
        D2 = (Pi @ (SiSi * OrOr) @ Pi.T).sum()
        AB = (Pr @ (SrSi * OrOi) @ Pr.T).sum() # AB == BA
        AC = (Pr @ (SrSr * OrOi) @ Pi.T).sum()
        AD = (Pr @ (SrSi * OrOr) @ Pi.T).sum()
        BC = (Pr @ (SiSr * OiOi) @ Pi.T).sum()
        BD = (Pr @ (SiSi * OiOr) @ Pi.T).sum()
        CD = (Pi @ (SrSi * OiOr) @ Pi.T).sum()

        return A2 + B2 + C2 + D2 + 2*AB + 2*AC + 2*BC - 2*AD - 2*BD - 2*CD

    
    def score(self, head_idx, relation_idx, tail_idx, headtype, tailtype):

        u_re = self.head_embedding_real_dict[headtype](head_idx)
        u_im = self.head_embedding_imag_dict[headtype](head_idx)
        v_re = self.tail_embedding_real_dict[tailtype](tail_idx)
        v_im = self.tail_embedding_imag_dict[tailtype](tail_idx)
        r_re = self.relation_real_embedding(relation_idx) 
        r_im = self.relation_imag_embedding(relation_idx)

        if self.do_node_real is not None: 
            u_re *= self.do_node_real[headtype][head_idx]
            u_im *= self.do_node_imag[headtype][head_idx]
            v_re *= self.do_node_real[tailtype][tail_idx]
            v_im *= self.do_node_imag[tailtype][tail_idx]

        scores = (triple_dot(u_re, r_re, v_re) +
                  triple_dot(u_im, r_re, v_im) +
                  triple_dot(u_re, r_im, v_im) -
                  triple_dot(u_im, r_im, v_re))**2
        
        return scores
    
    def set_dropout_masks(self, device): 

        if (self.training) and (self.dropout > 0): 

            self.do_node_real = {}
            self.do_node_imag = {} 
            for nodetype in self.head_embedding_imag_dict.keys(): 
                self.do_node_real[nodetype] = 1.*(torch.rand(size=self.head_embedding_imag_dict[nodetype].weight.size(), device=device) > self.dropout)
                self.do_node_imag[nodetype] = 1.*(torch.rand(size=self.head_embedding_imag_dict[nodetype].weight.size(), device=device) > self.dropout) 
        else: 
            self.do_node_real = None
            self.do_node_imag = None

    def forward(self, head, relation, tail):
        ''''''
        log_prob = torch.zeros((head.size(0)), dtype=self.dtype, device=head.device)

        self.set_dropout_masks(device=head.device)

        for rel in torch.unique(relation): 
            h,r,t   = self.rel2type[rel.item()]
            rel_idx = torch.nonzero(relation == rel, as_tuple=True)[0]
            Z       = self.partition_function(headtype=h, relint=rel, tailtype=t)
            phi     = self.score(head_idx          = head[rel_idx],
                                 relation_idx      = relation[rel_idx], 
                                 tail_idx          = tail[rel_idx], 
                                 headtype          = h, 
                                 tailtype          = t)
            
            log_prob[rel_idx] = torch.log(phi + self.eps_dict[(h,r,t)]) - torch.log1p(Z)
            
        return log_prob


def triple_dot(x,y,z):
    return (x * y * z).sum(dim=-1)


'''

    def forward(self, head, relation, tail):
        ''''''
        log_prob = torch.zeros((head.size(0)), dtype=self.dtype, device=head.device)

        self.set_dropout_masks(device=head.device)

        for rel in self.relations: 
            h,r,t   = self.rel2type[rel]
            rel_idx = torch.nonzero(relation == rel, as_tuple=True)[0]
            Z       = self.partition_function(headtype=h, relint=rel, tailtype=t)
            phi     = self.score(head_idx          = head[rel_idx],
                                 relation_idx      = relation[rel_idx], 
                                 tail_idx          = tail[rel_idx], 
                                 headtype          = h, 
                                 tailtype          = t)
            
            log_prob[rel_idx] = torch.log(phi + self.eps_dict[(h,r,t)]) - torch.log1p(Z)
            
        return log_prob
'''

'''
old partition function 


        SrSr = Sr.T@Sr  ;  PrPr = Pr.T@Pr  ;  OrOr = Or.T@Or
        SiSi = Si.T@Si  ;  PiPi = Pi.T@Pi  ;  OiOi = Oi.T@Oi 
        SrSi = Sr.T@Si  ;  PrPi = Pr.T@Pi  ;  OrOi = Or.T@Oi 
        SiSr = Si.T@Sr  ;                     OiOr = Oi.T@Or
        # SrSi =/= SiSr
        # PiPr = Pi.T@Pr not used

        A2 = (SrSr * PrPr * OrOr).sum()
        B2 = (SiSi * PrPr * OiOi).sum()
        C2 = (SrSr * PiPi * OiOi).sum()
        D2 = (SiSi * PiPi * OrOr).sum()
        
        AB = (SrSi * PrPr * OrOi).sum() # AB = BA
        AC = (SrSr * PrPi * OrOi).sum()
        AD = (SrSi * PrPi * OrOr).sum()
        BC = (SiSr * PrPi * OiOi).sum()
        BD = (SiSi * PrPi * OiOr).sum()
        CD = (SrSi * PiPi * OiOr).sum()

        return A2 + B2 + C2 + D2 + 2*AB + 2*AC + 2*BC - 2*AD - 2*BD - 2*CD


def forward(self, head, relation, tail):
        ''''''
        log_prob = torch.zeros((head.size(0)), dtype=self.dtype, device=head.device)

        for rel in torch.unique(relation): 
            h,r,t = self.rel2type[rel.item()]
            rel_idx = torch.nonzero(relation == rel, as_tuple=True)[0]
            Z = self.partition_function(headtype=h, relint=rel, tailtype=t)
            phi = self.score(head_idx           = head[rel_idx], 
                                      relation_idx      = relation[rel_idx], 
                                      tail_idx          = tail[rel_idx], 
                                      headtype          = h, 
                                      relint            = rel, 
                                      tailtype          = t)
            
            log_prob[rel_idx] = torch.log(phi + 1e-24) - torch.log1p(Z)
            
        return log_prob




def partition_function(self, headtype:str, relint:int, tailtype:str) -> torch.tensor: 
        #Squared ComplEx partition function as described in "How to Turn our KGE in generative models" 
        #Slightly different version than used in paper, since we are conditioning on relation. Faster and less memory. 
        #Subject (real) ~ Sr
        Sr = self.node_embedding_real_dict[headtype].weight
        Si = self.node_embedding_imag_dict[headtype].weight
        Pr = self.relation_real_embedding(relint).view(1,-1)
        Pi = self.relation_imag_embedding(relint).view(1,-1)
        Or = self.node_embedding_real_dict[tailtype].weight
        Oi = self.node_embedding_imag_dict[tailtype].weight

        SrSr = Sr.T@Sr  ;  OrOr = Or.T@Or
        SiSi = Si.T@Si  ;  OiOi = Oi.T@Oi 
        SrSi = Sr.T@Si  ;  OrOi = Or.T@Oi 
        SiSr = Si.T@Sr  ;  OiOr = Oi.T@Or
        # SrSi =/= SiSr

        A2 = (Pr @ (SrSr * OrOr) @ Pr.T).sum()
        B2 = (Pr @ (SiSi * OiOi) @ Pr.T).sum()
        C2 = (Pi @ (SrSr * OiOi) @ Pi.T).sum()
        D2 = (Pi @ (SiSi * OrOr) @ Pi.T).sum()
        AB = (Pr @ (SrSi * OrOi) @ Pr.T).sum() # AB == BA
        AC = (Pr @ (SrSr * OrOi) @ Pi.T).sum()
        AD = (Pr @ (SrSi * OrOr) @ Pi.T).sum()
        BC = (Pr @ (SiSr * OiOi) @ Pi.T).sum()
        BD = (Pr @ (SiSi * OiOr) @ Pi.T).sum()
        CD = (Pi @ (SrSi * OiOr) @ Pi.T).sum()

        return A2 + B2 + C2 + D2 + 2*AB + 2*AC + 2*BC - 2*AD - 2*BD - 2*CD

'''