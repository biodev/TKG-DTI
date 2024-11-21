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


--- 

See `/docs/` for additional details.  