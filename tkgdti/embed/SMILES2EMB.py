import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch 
import re


class SMILES2EMB: 

    def __init__(self, model_name="ibm/MoLFormer-XL-both-10pct"):
        '''
        
        ibm/MoLFormer-XL-both-10pct (https://huggingface.co/ibm/MoLFormer-XL-both-10pct) 
            SMILES FORMAT: Canonical, No isomeric information. 
            MAX LENGTH: 202 tokens

        "seyonec/ChemBERTa-zinc-base-v1"
        "DeepChem/ChemBERTa-10M-MTR"
        '''

        self.model_name = model_name
        self.model = AutoModel.from_pretrained(self.model_name, deterministic_eval=True, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.max_len = self.tokenizer.model_max_length

    def embed(self, smiles, batch_size=512, device='cuda', verbose=True): 
        
        model = self.model.to(device)

        outputs = []
        with torch.no_grad():
            for i in range(0, len(smiles), batch_size):
                if verbose: print(f'embedding smiles...: {i}/{len(smiles)}', end='\r')
                
                smiles_batch = smiles[i:i+batch_size] 
                
                inputs = self.tokenizer(
                    smiles_batch,
                    return_tensors="pt",
                    padding=True,
                )

                inputs = {k: v.cuda() for k, v in inputs.items()}
                batch_outputs = model(**inputs)
                
                outputs.append(batch_outputs.pooler_output.cpu())
                #outputs.append(batch_outputs.last_hidden_state[:, 0, :].cpu())

                torch.cuda.empty_cache()

        return torch.cat(outputs)    