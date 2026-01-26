# Data Availability

> **Note:** This documentation was reviewed and edited with assistance from an LLM. Please verify critical details against the source code.

This document describes all external data sources required to reproduce the TKG-DTI knowledge graph construction and model training workflows.

To run any of the workflows (`/workflows/`), the following data will need to be downloaded into a folder `/data/` and the paths updated in respective workflow `config.yaml` files.

**Helper Script:** A download script is provided at `scripts/get_tkg_raw_files.sh` that can automatically download several of the required datasets (CTD files, BeatAML data, UniProt proteome). Update the `ROOT` path in the script before running.

**Note:** Some data (OmniPath, PubChem, jGlaser, HuggingFace models) is automatically retrieved via API during workflow execution.

---

## 1. Expanded Cancer Targetome

**Description:** Curated drug-target interactions (DTIs) with binding affinity data, assay types, and confidence tiers. This is the primary DTI source for the knowledge graph.

**Files required:**
- `targetome_extended-01-23-25.csv` — DTI data with assay values
- `targetome_extended_drugs-01-23-25.csv` — Drug metadata (InChIKey, names, clinical phase)

**Access:** The Expanded Cancer Targetome will be released shortly. Please contact the corresponding author (`evansna@ohsu.edu`) for access.

**License:** TBD

**Used in:** `kg_construction_01__drug_interacts_protein.py`

---

## 2. BeatAML Drug Families (Optional)

**Description:** Drug-gene target mappings from the BeatAML study, used in alternative targetome construction.

**Files required:**
- `beataml_drug_families.xlsx`

**Access:** https://github.com/biodev/beataml2.0_data (auto-downloaded by `scripts/get_tkg_raw_files.sh`)

**License:** See BeatAML data use terms

**Used in:** `kg_construction_01__drug_interacts_protein.py`

---

## 3. Comparative Toxicogenomics Database (CTD)

**Description:** Curated chemical-disease, gene-disease, and gene-pathway associations.

**Files required:**
- `CTD_chemicals_diseases.csv` — Chemical-disease associations
- `CTD_curated_genes_diseases.csv` — Gene-disease associations (curated)
- `CTD_genes_pathways.csv` — Gene-pathway memberships

**Access:** http://ctdbase.org/downloads/ (auto-downloaded by `scripts/get_tkg_raw_files.sh`)

**License:** CTD data is freely available for academic research. See http://ctdbase.org/about/legal.jsp for terms.

**Used in:**
- `kg_construction_03__drug_associates_disease.py`
- `kg_construction_04__protein_associates_disease.py`
- `kg_construction_06__protein_isin_pathway.py`

---

## 4. LINCS L1000 (Phase II)

**Description:** Gene expression signatures from drug perturbation experiments across multiple cell lines.

**Files required:**
- `siginfo_beta.txt` — Signature metadata
- `compoundinfo_beta.txt` — Compound metadata with InChIKeys
- `geneinfo_beta.txt` — Gene metadata
- `level5_beta_trt_cp_n720216x12328.gctx` — Level 5 expression matrix (HDF5 format)

**Access:** https://clue.io/data/CMap2020#LINCS2020

**License:** LINCS data is publicly available. See https://clue.io/data/terms for terms.

**Used in:** `kg_construction_07__lincs_drug_perturbed_expression.py`

---

## 5. UniProt Human Reference Proteome

**Description:** Canonical amino acid sequences for human proteins, used for protein embedding.

**Files required:**
- `UP000005640_9606.fasta` — Human proteome (Swiss-Prot + TrEMBL)

**Access:** https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000005640/ (auto-downloaded by `scripts/get_tkg_raw_files.sh`)

**License:** Creative Commons Attribution 4.0 (CC BY 4.0). See https://www.uniprot.org/help/license

**Used in:** `kg_construction_08a__embed_amino_acids.py`

---

## 6. OmniPath

**Description:** Comprehensive protein-protein interaction database aggregating multiple literature-curated sources.

**Files required:** None (accessed via Python API)

**Access:** Installed via `pip install omnipath`. Data retrieved at runtime from https://omnipathdb.org/

**License:** OmniPath data is licensed under CC BY 4.0. See https://omnipathdb.org/info

**Used in:** `kg_construction_05__protein_interacts_protein.py`

---

## 7. PubChem

**Description:** Used to resolve drug names to canonical SMILES and InChIKey identifiers.

**Files required:** None (accessed via PubChemPy API)

**Access:** Installed via `pip install pubchempy`. See https://pubchem.ncbi.nlm.nih.gov/

**License:** PubChem data is in the public domain (US Government work).

**Used in:** `kg_construction_01__drug_interacts_protein.py`, `kg_construction_03__drug_associates_disease.py`

---

## 8. jGlaser Binding Affinity Dataset

**Description:** Large-scale drug-protein binding affinity dataset from HuggingFace, used to train an auxiliary binding affinity predictor.

**Files required:** None (loaded via HuggingFace `datasets` library)

**Access:** `load_dataset("jglaser/binding_affinity")` — see https://huggingface.co/datasets/jglaser/binding_affinity

**License:** Aggregate dataset, see individual licenses of source datasets - specified in https://huggingface.co/datasets/jglaser/binding_affinity

**Used in:** `kg_construction_09a__binding_affinity_make.py`

---

## Pre-trained Models (HuggingFace)

The following pre-trained models are downloaded automatically from HuggingFace during workflow execution:

### ChemBERTa (Drug Embeddings)

**Description:** Transformer model for SMILES-based molecular representation.

ChemBerta github: https://github.com/seyonechithrananda/bert-loves-chemistry

**Model options:**
- `yzimmermann/ChemBERTa-zinc-base-v1-safetensors` (default)
- `yzimmermann/ChemBERTa-77M-MLM-safetensors`

**Access:** https://huggingface.co/yzimmermann

**License:** MIT License (original work)

**Used in:** `kg_construction_02__drug_similar_drug.py`, `kg_construction_09a__binding_affinity_make.py`

---

### ProtBert / ESM2 (Protein Embeddings)

**Description:** Transformer models for protein sequence representation.

**Model options:**
- `facebook/esm2_t30_150M_UR50D` (default)

**Access:**
- ESM2: https://huggingface.co/facebook/esm2_t30_150M_UR50D

**License:**
- ESM2: MIT License

**Used in:** `kg_construction_08a__embed_amino_acids.py`, `kg_construction_09a__binding_affinity_make.py`

---

## Data Directory Structure

After downloading all required data, organize files as follows:

```
data/
├── targetome_extended-01-23-25.csv
├── targetome_extended_drugs-01-23-25.csv
├── beataml_drug_families.xlsx          # optional
├── CTD_chemicals_diseases.csv
├── CTD_curated_genes_diseases.csv
├── CTD_genes_pathways.csv
├── siginfo_beta.txt
├── compoundinfo_beta.txt
├── geneinfo_beta.txt
├── level5_beta_trt_cp_n720216x12328.gctx
└── UP000005640_9606.fasta
```

Configure the `dirs.data_root` path in `workflow/*/config.yaml` to point to this directory.

---

## Notes

- **API Dependencies:** OmniPath and PubChem data are retrieved via API at runtime. Ensure network access during workflow execution.
- **Storage Requirements:** The LINCS GCTX file is quite large (>20GB). Ensure sufficient disk space.
- **GPU Recommended:** Protein and drug embedding steps (08a, 09a) benefit significantly from GPU acceleration.
