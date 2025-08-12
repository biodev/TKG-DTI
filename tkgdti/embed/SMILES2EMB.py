import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import re


class SMILES2EMB:

    def __init__(self, model_name: str = "seyonec/ChemBERTa-zinc-base-v1"):
        """
        Default: ChemBERTa (seyonec/ChemBERTa-zinc-base-v1)

        Prior models:
          - ibm/MoLFormer-XL-both-10pct (requires canonical SMILES, max ~202 tokens)
          - DeepChem/ChemBERTa-10M-MTR
        """
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _calc_max_len(self, smiles_list):
        # Follow notebook logic: char length + 1 as an upper bound for tokenizer max_length
        char_max = max(len(s) for s in smiles_list) if len(smiles_list) > 0 else 1
        # Respect tokenizer model_max_length when defined
        model_max = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(model_max, int) and model_max > 0:
            return min(char_max + 1, model_max)
        return char_max + 1

    def embed(self, smiles, batch_size: int = 512, device: str = "cuda", verbose: bool = True) -> torch.Tensor:
        model = self.model.to(device)
        max_len = self._calc_max_len(smiles)

        outputs = []
        with torch.no_grad():
            for i in range(0, len(smiles), batch_size):
                if verbose:
                    print(f"embedding smiles...: {i}/{len(smiles)}", end="\r")

                smiles_batch = smiles[i : i + batch_size]
                inputs = self.tokenizer(
                    smiles_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                )

                inputs = {k: v.to(device) for k, v in inputs.items()}
                batch_outputs = model(**inputs)
                # Use [CLS] token embedding as in 02_drug_drug_similarity
                cls_emb = batch_outputs.last_hidden_state[:, 0, :].detach().cpu()
                outputs.append(cls_emb)

                if device == "cuda":
                    torch.cuda.empty_cache()

        return torch.cat(outputs) if len(outputs) > 0 else torch.empty(0)