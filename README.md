# Targetome Knowledge Graph for high-quality drug target interaction predictions 

> corresponding author: `evansna@ohsu.edu`

## Setting up environment with conda/mamba: 
(expectation of a cuda enabled GPU)

```bash
$ mamba env create -f environment.yml
$ mamba activate tkgdti 

# install tkgdti package
(kgdti) $ pip install -e . 
```

## Downloading and processing datasets 

We currently have methods to download the OGB-biokg [dataset](https://ogb.stanford.edu/docs/leader_linkprop/#ogbl-biokg) and the Hetero-A [dataset](https://www.nature.com/articles/s41467-017-00680-8). 

```bash 
(kgdti) $ python ./scripts/create_heteroa.py 
(kgdti) $ python ./scripts/create_biokg.py
```

## Training a Complex^2 model 

```bash 
(tkgdti) $ python train_complex2.py --data /path/to/data/ --out /path/to/output 
```

## Training a GNN model 

```bash 
(tkgdti) $ python train_complex2.py --data /path/to/data/ --out /path/to/output 
```

## Results 

FOLD-0 HeteroA 
GNN best performance: 
MRR      Top1      Top3     Top10   avg_AUC
0.340086  0.205928  0.410296  0.611544  0.930691

Complex2 best performance 
MRR      Top1      Top3     Top10   avg_AUC
0.329474  0.218409  0.391576  0.542902  0.862168



---
# Future work / feature requests 
---

1. Consider using page rank to identify smaller knowledge subgraph relevant to drugs/targets. This can improve scalability of the method. 

2. Consider adding a relation type from beat aml patients -> pathway using gene expression enrichment of pathways, see `/notebooks/tkg/10_beataml_patient_response_and_omics.ipynb` for more info. 