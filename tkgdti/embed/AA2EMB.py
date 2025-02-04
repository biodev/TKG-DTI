import numpy as np
from transformers import BertModel, BertTokenizer, pipeline
import torch 
import re
from torch.utils.data import Dataset, DataLoader

class AA2EMB:
    def __init__(self, model_name="Rostlab/prot_bert"): 
        '''
        "Rostlab/prot_bert" (https://huggingface.co/Rostlab/prot_bert)
        AMINO ACID FORMAT: 1-letter code
        '''

        self.model_name = model_name 
        self.model = BertModel.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=False)

    def embed(self, aas, batch_size=124, device='cuda', verbose=True):
        # Preprocessing amino acid sequences
        # 1. Convert each sequence into spaced single-letter tokens: "A E T C ..."
        # 2. Replace uncommon amino acids [U,Z,O,B] with [X]
        aas = [' '.join(list(a)) for a in aas]
        aas = [re.sub(r"[UZOB]", "X", a) for a in aas]

        fe = pipeline('feature-extraction', model=self.model, tokenizer=self.tokenizer, device=device)

        outputs = []
        for i in range(0, len(aas), batch_size):
            if verbose:
                print(f'Progress: {i}/{len(aas)}', end='\r')

            batch_seqs = aas[i:i+batch_size]
            # fe(...) returns a list of embeddings: one entry per sequence

            with torch.no_grad(): 
                batch_embeddings = fe(batch_seqs)  # shape: [batch_size, seq_length, hidden_dim]

            for seq_emb in batch_embeddings:
                # seq_emb is a list (or numpy array) of shape: [seq_length, hidden_dim]
                seq_tensor = torch.tensor(seq_emb)
                # Compute the mean over the token dimension (dim=0)
                seq_mean = torch.mean(seq_tensor, dim=1)
                outputs.append(seq_mean.cpu())

            torch.cuda.empty_cache()

        # Stack all outputs into a single tensor: shape [num_sequences, hidden_dim]
        return torch.cat(outputs, 0)