import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import torch
import re


class SMILES2EMB:

    def __init__(self, model_name: str = "yzimmermann/ChemBERTa-77M-MLM-safetensors", 
                       max_len: int = 2048, 
                       batch_size: int = 512,
                       repr: str = 'cls'):
        """
        SMILES embedding using ChemBERTa models.
        
        Supports both BERT-based and RoBERTa-based ChemBERTa models:
          - ibm/MoLFormer-XL-both-10pct (requires canonical SMILES, max ~202 tokens)
          - DeepChem/ChemBERTa-10M-MTR
          - seyonec/ChemBERTa-zinc-base-v1
          - yzimmermann/ChemBERTa-77M-MLM-safetensors
          - yzimmermann/ChemBERTa-zinc-base-v1-safetensors (RoBERTa-based)

        Args:
            model_name: HuggingFace model identifier
            max_len: Maximum sequence length for tokenization
            batch_size: Batch size for embedding generation
            repr: Representation type ('cls' or 'mean')
        """
        self.model_name = model_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                              use_safetensors = True, 
                                              trust_remote_code=True )

        # Use AutoModel for embeddings (no need for masked LM head)
        self.model = AutoModel.from_pretrained(model_name, 
                                               use_safetensors = True, 
                                               trust_remote_code=True )

        self.max_len = max_len
        self.repr = repr
        self.batch_size = batch_size

    def _calc_max_len(self, smiles_list):
        # Follow notebook logic: char length + 1 as an upper bound for tokenizer max_length
        char_max = max(len(s) for s in smiles_list) if len(smiles_list) > 0 else 1
        # Respect tokenizer model_max_length when defined
        model_max = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(model_max, int) and model_max > 0:
            return min(char_max + 1, model_max)
        max_len = min(char_max + 1, self.max_len) 
        return max_len

    def embed(self, smiles, device: str = "cuda", verbose: bool = True) -> torch.Tensor:
        # Validate input
        if len(smiles) == 0:
            raise ValueError("Input smiles list is empty")
        
        # Check for empty SMILES
        empty_smiles = [i for i, smi in enumerate(smiles) if not smi or len(smi.strip()) == 0]
        if empty_smiles:
            print(f"WARNING: Found {len(empty_smiles)} empty SMILES at indices: {empty_smiles[:10]}{'...' if len(empty_smiles) > 10 else ''}")
        
        model = self.model.to(device)
        max_len = self._calc_max_len(smiles)

        if self.repr == 'cls': 
            # Handle both BERT-style [CLS] and RoBERTa-style <s> tokens
            try: 
                # Try BERT-style [CLS] first
                CLS_IDX = self.tokenizer.get_vocab()['[CLS]']
            except: 
                try:
                    # Try RoBERTa-style <s> (BOS token)
                    CLS_IDX = self.tokenizer.bos_token_id or 0
                    print(f'INFO: Using BOS token (<s>) at index {CLS_IDX} for CLS representation')
                except:
                    print('WARNING: Neither [CLS] nor BOS token found in tokenizer. Using first token instead.')
                    CLS_IDX = 0

        outputs = []
        with torch.no_grad():
            for i in range(0, len(smiles), self.batch_size):
                if verbose:
                    print(f"embedding smiles...: {i}/{len(smiles)}", end="\r")

                smiles_batch = smiles[i : i + self.batch_size]
                inputs = self.tokenizer(
                    smiles_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                )

                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Check for all-zero attention masks
                attention_mask = inputs['attention_mask']
                attention_sums = attention_mask.sum(dim=1)
                zero_attention_mask = (attention_sums == 0)
                if zero_attention_mask.any():
                    zero_indices = zero_attention_mask.nonzero(as_tuple=True)[0]
                    print(f"WARNING: Found {len(zero_indices)} SMILES with all-zero attention masks in batch starting at {i}")
                    print(f"Zero attention indices in batch: {zero_indices.tolist()}")
                    # Set minimum attention for sequences with all zeros (at least attend to first token)
                    attention_mask[zero_attention_mask, 0] = 1
                    inputs['attention_mask'] = attention_mask

                with torch.no_grad():
                    outputs_ = self.model(**inputs, return_dict=True)
                
                if self.repr == 'cls': 
                    # Use last_hidden_state instead of hidden_states[-1] for AutoModel
                    z_drug = outputs_.last_hidden_state[:, CLS_IDX, :].detach().cpu() # extract the CLS/BOS token embedding
                elif self.repr == 'mean':
                    # mask-aware mean pooling over tokens
                    hidden = outputs_.last_hidden_state  # [B, T, H]
                    attention_mask = inputs['attention_mask']
                    mask = attention_mask.unsqueeze(-1).type_as(hidden)
                    summed = (hidden * mask).sum(dim=1)
                    denom = mask.sum(dim=1).clamp_min(1e-6)
                    z_drug = (summed / denom).detach().cpu()
                else: 
                    raise ValueError(f'Invalid representation type: {self.repr}. Supported types: \'cls\', \'mean\'')
                
                # Validate output embeddings
                if torch.isnan(z_drug).any():
                    print(f"WARNING: NaN values detected in SMILES embeddings for batch starting at {i}")
                if torch.isinf(z_drug).any():
                    print(f"WARNING: Inf values detected in SMILES embeddings for batch starting at {i}")

                outputs.append(z_drug)

                if device == "cuda":
                    torch.cuda.empty_cache()

        if len(outputs) == 0:
            print("WARNING: No valid embeddings generated")
            return torch.empty((0, 0))
        
        out = torch.cat(outputs)
        return out  # Return tensor to match AA2EMB pattern