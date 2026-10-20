# Reproduce `workflow/full-tkg`

This runs the published ComplEx² pipeline: knowledge-graph construction, 10-fold training, aggregation, and removal of pairs already annotated in Targetome. The published settings are already in `workflow/full-tkg/config.yaml` (`experiment.id: full-tkg-02`, assay cutoff 100 nM, 10 folds). Change only the paths below.

Licenses for each source are in [data_availability.md](data_availability.md).

## Requirements

- Linux x86_64 with an NVIDIA GPU and a driver that supports CUDA 12.4
- [Miniforge](https://github.com/conda-forge/miniforge) (`mamba`) and `wget`
- Network on the first run
- Disk for a 35,518,405,386-byte LINCS matrix, the other inputs below, HuggingFace caches, and workflow outputs. Plan on well over 40 GB before outputs.



## 1. Install

`environment.lock.yml` is the environment used for the published run (Python 3.9.19, PyTorch 2.4.1, CUDA 12.4, Snakemake 7.32.4, `datasets` 3.6.0). Four pip lines are editable checkouts and make `mamba env create` fail. Strip those lines, create the env, then install this repo plus the two external packages the workflow imports.

```bash
git clone https://github.com/biodev/TKG-DTI.git
cd TKG-DTI

mamba env create -f environment.yml
mamba activate tkgdti

pip install -e .
pip install 'git+https://github.com/nathanieljevans/GSNN'
pip install 'git+https://github.com/nathanieljevans/hypernet'
```

GSNN provides the UniProt ID mapping used in step 01. HyperNet is the auxiliary binding model in steps 09b–09c. ComplEx² training uses `tkgdti/models/ComplEx2.py` in this repository. `datasets` loads the jGlaser affinity set in step 09a and is already in the lock file.

## 2. Download

Choose one directory and put these files in it. Step 01 reads the January 2025 Targetome CSVs from the share below. The public deposit [10.6084/m9.figshare.34367808](https://doi.org/10.6084/m9.figshare.34367808.v1) is the March 2026 release plus the prediction subset; this pipeline does not read those March tables.

`scripts/get_tkg_raw_files.sh` downloads the three CTD files, the UniProt FASTA, and `beataml_drug_families.xlsx`. This workflow uses the CTD files and the FASTA. The workbook is an input to `workflow/aml-tkg`.

```bash
DATA=/path/to/data
mkdir -p "$DATA"
cd "$DATA"

wget -O targetome_extended-01-23-25.csv \
  "https://figshare.com/ndownloader/files/70408455?private_link=c207e1c7fabbcdacbf6d"
wget -O targetome_extended_drugs-01-23-25.csv \
  "https://figshare.com/ndownloader/files/70408458?private_link=c207e1c7fabbcdacbf6d"

wget -O CTD_chemicals_diseases.csv.gz https://ctdbase.org/reports/CTD_chemicals_diseases.csv.gz
wget -O CTD_curated_genes_diseases.csv.gz https://ctdbase.org/reports/CTD_curated_genes_diseases.csv.gz
wget -O CTD_genes_pathways.csv.gz https://ctdbase.org/reports/CTD_genes_pathways.csv.gz
gunzip -f CTD_chemicals_diseases.csv.gz CTD_curated_genes_diseases.csv.gz CTD_genes_pathways.csv.gz

wget -O UP000005640_9606.fasta.gz \
  https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.fasta.gz
gunzip -f UP000005640_9606.fasta.gz

BASE=https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020
wget -c -O level5_beta_trt_cp_n720216x12328.gctx "$BASE/level5/level5_beta_trt_cp_n720216x12328.gctx"
wget -c -O siginfo_beta.txt "$BASE/siginfo_beta.txt"
wget -c -O compoundinfo_beta.txt "$BASE/compoundinfo_beta.txt"
wget -c -O geneinfo_beta.txt "$BASE/geneinfo_beta.txt"
```

`$DATA` then contains:

```text
targetome_extended-01-23-25.csv
targetome_extended_drugs-01-23-25.csv
CTD_chemicals_diseases.csv
CTD_curated_genes_diseases.csv
CTD_genes_pathways.csv
UP000005640_9606.fasta
level5_beta_trt_cp_n720216x12328.gctx
siginfo_beta.txt
compoundinfo_beta.txt
geneinfo_beta.txt
```

The first Snakemake run also downloads OmniPath, PubChem name-to-structure hits, UniProt mappings, `facebook/esm2_t30_150M_UR50D`, `yzimmermann/ChemBERTa-zinc-base-v1-safetensors`, and `jglaser/binding_affinity`. Those HuggingFace repos were ungated at the time of the published run. Step 03 walks many CTD names through PubChem; keep the rate at or below 5 requests per second and keep the caches the scripts write.

## 3. Point the config at those paths

In `workflow/full-tkg/config.yaml`, set absolute paths:

```yaml
dirs:
  data_root: "/path/to/data/"
  out_root: "/path/to/output/"
  script_dir: "/path/to/TKG-DTI/workflow/scripts"

step_14:
  targetome_expanded_path: "/path/to/data/targetome_extended-01-23-25.csv"
```

Leave the remaining keys as they are in the file.

## 4. Run

From the workflow directory, with `tkgdti` active:

```bash
cd /path/to/TKG-DTI/workflow/full-tkg
snakemake -j 1
```

`-j 1` runs one rule at a time. Each ComplEx² fold still uses 12 dataloader workers and the GPU (`step_11.use_cpu: false`). The default target runs steps 01–11, then aggregation (13) and Targetome filtering (14). Path-GNN (step 12) is outside that target.

The filtered prediction table is:

```text
/path/to/output/full-tkg-02/output/complex2/aggregated_and_filtered_predictions.csv
```

Fold graphs are under `full-tkg-02/output/targetomeKG/processed/FOLD_*`. Per-fold ComplEx² scores are under `full-tkg-02/output/complex2/FOLD_*`.

