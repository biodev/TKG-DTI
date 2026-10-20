> **Note:** This documentation was drafted with assistance from a large language model. File names, URLs, sizes, and license summaries were checked against primary sources and this repository’s code in August 2026. The Expanded Cancer Targetome Figshare record was rechecked on 20 October 2026. Confirm legal terms on the source sites before commercial use or redistribution.

# Data Availability

This document lists the **external** data and models needed to run the TKG-DTI knowledge-graph workflows, how to obtain them, and how this project uses each source. It is not a license grant.

---

## What this project releases

This repository releases **software** (GNU GPL v3.0; see `LICENSE.md` and [https://doi.org/10.5281/zenodo.18809293](https://doi.org/10.5281/zenodo.18809293)).

The companion data product is the Figshare dataset **[Targetome-Expanded Release](https://figshare.com/articles/dataset/Targetome-Expanded_Release/34367808)** ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); [https://doi.org/10.6084/m9.figshare.34367808.v1](https://doi.org/10.6084/m9.figshare.34367808.v1)). That record contains:

1. Curated Expanded Cancer Targetome tables (DTIs, drug metadata, synonyms).
2. **A filtered subset of ComplEx² drug–target predictions** (`TargetomeKG_Complex2_DTI_preds_03-19-2026.csv.gz`) — aggregated, thresholded candidate pairs, not the full pairwise score matrix.

**This project does not release:**

- Trained model weights (ComplEx², Path-GNN, or the auxiliary binding-affinity model)
- The constructed knowledge graph, relation CSVs, embeddings, or fold-level prediction dumps
- Copies of CTD, Reactome/KEGG pathway membership (via the CTD file), LINCS, UniProt, OmniPath, jGlaser, BeatAML extra tables, OGB BioKG, or DTINet/HeteroA

Sources other than the companion Targetome tables are used **only to train or evaluate models**. Their records are not redistributed here. Model outputs (the prediction subset above) are scores for drug–protein pairs; they are not a dump of those training databases.

---



## Workflows and configuration

For `workflow/full-tkg/` and `workflow/aml-tkg/`, put the files in the directory-structure section into one folder and set `dirs.data_root` in that workflow’s `config.yaml`. Also set `dirs.out_root` and `dirs.script_dir`. `workflow/full-tkg/` additionally uses `step_14.targetome_expanded_path` (path to a Targetome DTI table used only to drop already-annotated pairs from predictions).

`workflow/biokg/` and `workflow/hetero-a/` are separate benchmark pipelines. They do not need the Targetome/CTD/LINCS/UniProt folder.

**Helper script:** `scripts/get_tkg_raw_files.sh` downloads CTD, the UniProt proteome FASTA, `beataml_drug_families.xlsx`, and extra BeatAML `*_dbgap.txt` tables that **these workflows do not read**. Edit `ROOT` before running. It does **not** download Targetome or LINCS. Download the Targetome tables with `rebuild_tge.py` (section 1).

**Fetched at runtime (network required on first use):** OmniPath; PubChem name resolution; UniProt ID mapping via GSNN helpers; HuggingFace checkpoints and the jGlaser dataset; OGB BioKG and the DTINet tarball in their workflows.

Both `full-tkg` and `aml-tkg` restrict the drug universe to inhibitors in [beataml_drug_families.xlsx](https://github.com/biodev/beataml2.0_data/raw/main/beataml_drug_families.xlsx).

---



## Summary

Sizes below were recorded on 28 August 2026. CTD and UniProt “current” files change over time.


| Source                                           | Obtain how                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Size (approx.)                       | License (summary)                                                                                                                    | This project redistributes?                                           |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------- |
| Expanded Cancer Targetome                        | [Figshare](https://figshare.com/articles/dataset/Targetome-Expanded_Release/34367808) ([doi:10.6084/m9.figshare.34367808.v1](https://doi.org/10.6084/m9.figshare.34367808.v1))                                                                                                                                                                                                                                                                                      | 202,567,127 bytes (20 Oct 2026)      | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (Figshare item license)                                                    | **Yes** — companion tables + a **subset of predictions**. No weights. |
| BeatAML drug families                            | [Manual](https://github.com/biodev/beataml2.0_data/raw/main/beataml_drug_families.xlsx) or helper script                                                                                                                                                                                                                                                                                                                                                            | 58 KB                                | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (processed BeatAML repo)                                                   | No (already public)                                                   |
| CTD (3 files)                                    | [CTD_chemicals_diseases.csv.gz](https://ctdbase.org/reports/CTD_chemicals_diseases.csv.gz), [CTD_curated_genes_diseases.csv.gz](https://ctdbase.org/reports/CTD_curated_genes_diseases.csv.gz), [CTD_genes_pathways.csv.gz](https://ctdbase.org/reports/CTD_genes_pathways.csv.gz), or helper script                                                                                                                                                                | ~157 MB gzipped                      | Free for non-commercial use; commercial license required                                                                             | No — training/eval only                                               |
| Reactome (pathway membership)                    | Via [CTD_genes_pathways.csv.gz](https://ctdbase.org/reports/CTD_genes_pathways.csv.gz) (not OmniPath)                                                                                                                                                                                                                                                                                                                                                               | In the CTD pathways file above       | Reactome **data:** [CC0](https://creativecommons.org/publicdomain/zero/1.0/) ([reactome.org/license](https://reactome.org/license)). | No — training/eval only                                               |
| LINCS L1000 2020 `trt_cp`                        | [level5_beta_trt_cp_n720216x12328.gctx](https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level5/level5_beta_trt_cp_n720216x12328.gctx), [siginfo_beta.txt](https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/siginfo_beta.txt), [compoundinfo_beta.txt](https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/compoundinfo_beta.txt), [geneinfo_beta.txt](https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/geneinfo_beta.txt) | 35,518,405,386 bytes GCTX + metadata | NIH LINCS: reuse with citation                                                                                                       | No — training/eval only                                               |
| UniProt UP000005640 FASTA                        | [UP000005640_9606.fasta.gz](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.fasta.gz) or helper script                                                                                                                                                                                                                                                                       | 7,752,225 bytes gzipped              | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)                                                                            | No — training/eval only                                               |
| OmniPath (`datasets=omnipath`)                   | [interactions TSV](https://omnipathdb.org/interactions?datasets=omnipath&genesymbols=1)or via [pypath](https://pypath.omnipathdb.org/)                                                                                                                                                                                                                                                                                                                             | 86,189 edges (28 Aug 2026)           | No umbrella license; 72 primary sources (see §6). 19 academic/NC + PhosphoNetworks (non-profit) per pypath                           | No — training/eval only                                               |
| PubChem                                          | [PUG REST](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest) or via [PubChemPy](https://github.com/mcs07/PubChemPy)                                                                                                                                                                                                                                                                                                                                                   | Runtime                              | NCBI-created content is public domain; depositor data may differ                                                                     | No dumps; InChIKeys appear in released Targetome tables               |
| jGlaser binding affinity                         | [all.parquet](https://huggingface.co/datasets/jglaser/binding_affinity/resolve/main/data/all.parquet)                                                                                                                                                                                                                                                                                                                                                               | ~1.25 GB cache                       | Script metadata: BSD-2-Clause; **sources have their own terms**                                                                      | No — used only to train an auxiliary model (weights not released)     |
| `yzimmermann/ChemBERTa-zinc-base-v1-safetensors` | [HuggingFace](https://huggingface.co/yzimmermann/ChemBERTa-zinc-base-v1-safetensors/tree/main) (workflow default)                                                                                                                                                                                                                                                                                                                                                   | HF cache                             | Conversion card: no license. Original `seyonec/ChemBERTa-zinc-base-v1` project: MIT (paper does not name a license)                  | No — published checkpoint                                             |
| `yzimmermann/ChemBERTa-77M-MLM-safetensors`      | [HuggingFace](https://huggingface.co/yzimmermann/ChemBERTa-77M-MLM-safetensors/tree/main) (optional)                                                                                                                                                                                                                                                                                                                                                                | HF cache                             | Conversion card: no license. `DeepChem/ChemBERTa-77M-MLM`: no Hub license. ChemBERTa-2 paper does not name a license                 | No — published checkpoint                                             |
| `facebook/esm2_t30_150M_UR50D`                   | [HuggingFace](https://huggingface.co/facebook/esm2_t30_150M_UR50D/tree/main) (workflow default)                                                                                                                                                                                                                                                                                                                                                                     | HF cache                             | MIT (named in the *Science* paper; Hub tag `license:mit`)                                                                            | No — published checkpoint                                             |
| `Rostlab/prot_bert`                              | [HuggingFace](https://huggingface.co/Rostlab/prot_bert/tree/main) (optional)                                                                                                                                                                                                                                                                                                                                                                                        | HF cache                             | AFL-3.0 in the ProtTrans README; Hub card has no license tag; paper does not name a license                                          | No — published checkpoint                                             |
| OGB `ogbl-biokg`                                 | [biokg.zip](https://snap.stanford.edu/ogb/data/linkproppred/biokg.zip)                                                                                                                                                                                                                                                                                                                                                                                              | OGB cache                            | OGB compilation: CC0; sources vary                                                                                                   | No — `workflow/biokg/` only                                           |
| DTINet HeteroA                                   | [data.tar.gz](https://github.com/luoyunan/PyDTINet/raw/main/data.tar.gz)                                                                                                                                                                                                                                                                                                                                                                                            | ~8.7 MB                              | DTINet: GPL-3.0; sources vary (DrugBank is NC)                                                                                       | No — `workflow/hetero-a/` only                                        |




Budget **~40 GB** for these inputs plus HuggingFace caches, and additional space for workflow outputs. The LINCS GCTX dominates.

---



## 1. Expanded Cancer Targetome

**Description:** Curated drug–target interactions with assay type, assay value, and drug metadata. Primary DTI labels for the knowledge graph.

**This project redistributes:** Yes, as a separate Figshare dataset (the tables are not in this software repository). The curated tables and a **filtered subset of ComplEx² predictions** are in [Targetome-Expanded Release](https://figshare.com/articles/dataset/Targetome-Expanded_Release/34367808). Model weights and full score matrices are **not** released.

**Figshare record** (public version 1, published 15 October 2026):

- Landing page: [https://figshare.com/articles/dataset/Targetome-Expanded_Release/34367808](https://figshare.com/articles/dataset/Targetome-Expanded_Release/34367808)
- DOI: [https://doi.org/10.6084/m9.figshare.34367808.v1](https://doi.org/10.6084/m9.figshare.34367808.v1)
- Item license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

**Files on that record** (sizes from the Figshare API, 20 October 2026):

- `targetome_expanded_03-26-26.tsv.gz` — DTI table (42,400,021 bytes)
- `targetome_expanded_drugs_03-26-26.tsv.gz` — drug metadata (68,165,134 bytes)
- `targetome_expanded_syns_03-26-26.tsv.gz` — synonyms (not read by this pipeline; 91,787,100 bytes)
- `TargetomeKG_Complex2_DTI_preds_03-19-2026.csv.gz` — released prediction **subset** (not a KG-construction input; 181,167 bytes)
- `data_dictionary.md` (13,566 bytes)
- `LICENSE.txt` (20,139 bytes)

**Files this pipeline reads** (names hard-coded in `kg_construction_01__drug_interacts_protein.py`):

- `targetome_extended-01-23-25.csv`
- `targetome_extended_drugs-01-23-25.csv`

Those names refer to a January 2025 CSV snapshot, which was used for ComplEx² training. The Figshare tables add columns, include high-confidence predicted interactions, and update the aggregated drug–target records. `rebuild_tge.py` downloads the two Figshare tables this pipeline needs and writes the filenames above into `--out` (cache: `<out>/_figshare_cache`).

Default (no `--bit-identical`) is for retraining on the current release. Predicted DTI rows are dropped (`assay_type == Predicted` or `evidence_level == 0`). Every other released assay row is kept, including interactions added after January 2025, and the DTI file is written with the eight columns step 01 reads. Every released drug row is kept, with all fifteen released drug columns. Drug names are not rewritten to the January 2025 strings.

```bash
python rebuild_tge.py --out ./path/to/data/dir/
```

`--bit-identical 01-23-25` rebuilds the January 2025 training files byte for byte, including the January drug names and the four dropped drug columns (`drugbank_status`, `drugbank_type`, `iuphar_type`, `iuphar_status`). Filtering the Figshare tables alone cannot do that: some January rows were removed and some cells changed. The patch is not in this repository. Download `patch_01-23-25.zip` from the [Targetome-Expanded Release](https://figshare.com/articles/dataset/Targetome-Expanded_Release/34367808) Figshare record, unzip it, and pass the resulting directory as `--patch_path`:

```bash
unzip patch_01-23-25.zip
python rebuild_tge.py --out ./path/to/data/dir/ --bit-identical 01-23-25 --patch_path ./patch_01-23-25
```

`patch_01-23-25/` contains the row index into the March tables and the January lines those tables cannot regenerate.

Copy the two CSV files into `dirs.data_root`.

Step 01 assigns **every** Expanded Targetome DTI row `targetome_adj_tier='TIER_1'` (the public tables use `evidence_level` 0–III; that field is not read). It then keeps only drugs whose names appear in BeatAML `drug_gene`. Assay filters come from `config.yaml` (`assay_types`, `max_assay_value`, `TIERS`).

**License:** The Figshare item is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (attribution required; no ShareAlike term on the item). The `LICENSE.txt` deposited with version 1 is still the CC BY-SA 4.0 legal text; cite the item license above. The tables aggregate records from primary resources (see `data_dictionary.md` on the Figshare record, `database` column); those sources keep their own terms.

**Citation:** Bottomly, D., Evans, N. & McWeeney, S. K. Expanding the set of high evidence drug-target interactions in the Cancer Targetome. Submitted. Dataset: Bottomly, D., Evans, N. & McWeeney, S. (2026). Targetome-Expanded Release. figshare. [https://doi.org/10.6084/m9.figshare.34367808.v1](https://doi.org/10.6084/m9.figshare.34367808.v1).

**Used in:** `kg_construction_01__drug_interacts_protein.py`. `filter_targetome_from_predictions.py` (`full-tkg` step 14) reads a DTI table path from `step_14.targetome_expanded_path`.

---



## 2. BeatAML drug families (required by the construction scripts)

**Description:** Drug–gene mappings used as the inhibitor name set and as additional Targetome-tier annotations. Step 01 always loads this workbook and intersects Expanded Targetome drugs with its `inhibitor` names.

**This project redistributes:** No.

**File:** `beataml_drug_families.xlsx`, sheet `drug_gene`. Columns read: `inhibitor`, `Symbol`, `GeneID`, `targetome_adj_tier`, `targetome_min_assay`.

**Access:** [https://github.com/biodev/beataml2.0_data](https://github.com/biodev/beataml2.0_data)  

```text
https://github.com/biodev/beataml2.0_data/raw/main/beataml_drug_families.xlsx
```

(58,330 bytes on 28 August 2026.)

**License:** The BeatAML 2.0 **processed** repository is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ([https://biodev.github.io/BeatAML2/](https://biodev.github.io/BeatAML2/), licensing FAQ). Raw sequencing is controlled-access (dbGaP / NCI GDC) and is **not** used here.

**Citation:** Bottomly et al., *Cancer Cell* 40:850–864.e9 (2022). doi:[10.1016/j.ccell.2022.07.002](https://doi.org/10.1016/j.ccell.2022.07.002). Original Cancer Targetome: Blucher et al., *Trends Pharmacol. Sci.* 38:1085–1099 (2017). doi:[10.1016/j.tips.2017.08.006](https://doi.org/10.1016/j.tips.2017.08.006).

**Used in:** `kg_construction_01__drug_interacts_protein.py`

### Extra BeatAML files (not used)

The helper script also fetches inhibitor-response, mutation, and expression `*_dbgap.txt` files. `kg_construction_10_make_tkg.py` can drop `dbgap_subject` relations; `full-tkg` and `aml-tkg` set `exclude_patient_relations: true`. Those extra files are **not** inputs to the current workflows.

---



## 3. Comparative Toxicogenomics Database (CTD)

**Description:** Chemical–disease, gene–disease, and gene–pathway associations used as knowledge-graph edges during training.

**This project redistributes:** No. CTD files are not shipped in this repo or in the companion Targetome release.

**Files required** (uncompressed CSV, these names):


| File                             | Role in this pipeline                                                                                                                | Size gzipped (28 Aug 2026) |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | -------------------------- |
| `CTD_chemicals_diseases.csv`     | Step 03: chemical–disease edges. **Not** filtered to `DirectEvidence`; inferred associations are retained after drug matching        | 162,841,513 bytes          |
| `CTD_curated_genes_diseases.csv` | Step 04: curated gene–disease edges, then restricted to diseases from step 03                                                        | 568,689 bytes              |
| `CTD_genes_pathways.csv`         | Step 06: gene–pathway membership. CTD’s compiled KEGG + **Reactome** annotations (not an OmniPath subset). See the Reactome section. | 1,095,188 bytes            |


**Access:** [https://ctdbase.org/downloads/](https://ctdbase.org/downloads/)  
The HTML site may present a captcha. The helper script uses:

```text
https://ctdbase.org/reports/CTD_chemicals_diseases.csv.gz
https://ctdbase.org/reports/CTD_curated_genes_diseases.csv.gz
https://ctdbase.org/reports/CTD_genes_pathways.csv.gz
```

Gunzip after download. Files start with `#` comments; the scripts use `pandas.read_csv(..., comment='#')`. CTD updates monthly; archive the files you used if you need a frozen graph.

**License:** CTD states that content is freely available to **non-commercial** users and that **commercial** users must purchase a license to download data files (NCSU copyright; licensing via [Catalyst Research Alliance](https://www.catalystresearch.io/products/ctd)). See [http://ctdbase.org/about/legal.jsp](http://ctdbase.org/about/legal.jsp) and Davis et al., *Nucleic Acids Res.* (2025 CTD update). Cite as requested at [https://ctdbase.org/about/publications/#citing](https://ctdbase.org/about/publications/#citing).

**Runtime:** Step 03 maps CTD `ChemicalName` values to InChIKey via PubChem (first result) and writes `extdata/ctd_targetome_drug_overlap.csv`. Reuse that cache; delete it if the Targetome drug set changes. Only chemicals that match Targetome InChIKeys are kept.

**Used in:** `kg_construction_03__drug_associates_disease.py`, `kg_construction_04__protein_associates_disease.py`. Step 06 reads the same CTD pathways file for Reactome/KEGG membership (next section).

---



## Reactome (pathway membership)

**Description:** Gene–pathway membership edges for the knowledge graph (`gene → isin_fwd → pathway` and the reverse). This is a **separate** use of Reactome from the incidental `Reactome_`* tags inside the OmniPath PPI pull.

**This project redistributes:** No.

**How it is obtained:** Not from reactome.org and not from pypath. Step 06 reads the CTD-distributed file `CTD_genes_pathways.csv` (helper script: `https://ctdbase.org/reports/CTD_genes_pathways.csv.gz`). CTD compiles gene–pathway associations from **Reactome** and **KEGG** ([CTD pathway vocabulary](https://ctdbase.org/voc.go?type=pathway); PathwayIDs such as `REACT:R-HSA-…` and `KEGG:hsa…`). `kg_construction_06__protein_isin_pathway.py` keeps every row; it does not filter to Reactome-only IDs.

**Files required:** `CTD_genes_pathways.csv` (same file as in the CTD section; 1,095,188 bytes gzipped on 28 August 2026).

**License:**

- **Reactome data** (annotation / identifier-mapping style content): [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Reactome’s license page: “All data in the Reactome database and files derived from that data are licensed under the Creative Commons Public Domain Dedication (CC0).” Attribution is encouraged but not required. Software, pathway illustrations, and database dumps remain **CC BY 4.0**; this pipeline does not use those. [https://reactome.org/license](https://reactome.org/license)
- **The CTD file** that actually ships the membership table is still a CTD download. Commercial use of that file requires a CTD license even though the underlying Reactome records are CC0.

**Citation:** Milacic, M. et al. The Reactome Pathway Knowledgebase 2024. *Nucleic Acids Res.* **52**, D672–D678 (2024). [https://doi.org/10.1093/nar/gkad1025](https://doi.org/10.1093/nar/gkad1025). Also cite CTD for the download.

**Used in:** `kg_construction_06__protein_isin_pathway.py`

---



## 4. LINCS L1000 / Connectivity Map 2020 (compound treatments)

**Description:** Level 5 gene-expression signatures for small-molecule treatments. Used only to build per-cell-line perturbation edges for training.

**This project redistributes:** No.

The scripts expect the **LINCS 2020 beta** compound-treatment matrix (`trt_cp`), not GEO Phase I/II GCTX files (different names, dimensions, and metadata). Those GEO files are **not** drop-in replacements.

**Files required:**


| File                                    | Role                                         | Bytes (28 Aug 2026) |
| --------------------------------------- | -------------------------------------------- | ------------------- |
| `level5_beta_trt_cp_n720216x12328.gctx` | Level 5 matrix (HDF5/GCTX), read with `h5py` | 35,518,405,386      |
| `siginfo_beta.txt`                      | Signature metadata                           | 465,242,319         |
| `compoundinfo_beta.txt`                 | Compound metadata, including InChIKey        | 4,631,014           |
| `geneinfo_beta.txt`                     | Gene metadata                                | 1,141,389           |


**Access:** clue.io retired on 31 January 2026. As of 28 August 2026 the 2020 build still answered on the historical CMap S3 bucket (HTTP 200):

```bash
BASE=https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020
wget -c "$BASE/level5/level5_beta_trt_cp_n720216x12328.gctx"
wget -c "$BASE/siginfo_beta.txt"
wget -c "$BASE/geneinfo_beta.txt"
wget -c "$BASE/compoundinfo_beta.txt"
```

That bucket is not an archival repository and may disappear. NIH-funded L1000 data are also in GEO ([GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742), [GSE70138](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138)), but using them requires changing this code.

**How step 07 subsets the data:** Targetome-matched InChIKeys; `pert_dose == 10.0`; `pert_time == 24.0`; `is_exemplar_sig == 1`; cell lines `MCF7`, `SKB`, `PC3`, `VCAP`, `PHH`, `HT29`, `HA1E`, `A375`. Score sign and threshold come from `config.yaml` (`sig_sign`, `score_threshold`).

**License:** NIH LINCS policy is release without restriction other than correct citation ([data release policy](https://lincsproject.org/LINCS/data/release-policy)). Connectivity Map redistribution notes ask that redistributors cite the GEO/source accession and describe processing differences ([Connectopedia](https://clue.io/connectopedia/data_redistribution)). Academic and for-profit users may use NIH-funded L1000 data. Retirement of clue.io does not, by itself, change that data policy.

**Used in:** `kg_construction_07__lincs_drug_perturbed_expression.py`

---



## 5. UniProt human reference proteome

**Description:** Amino-acid sequences used to embed proteins that appear in the graph.

**This project redistributes:** No.

**File:** `UP000005640_9606.fasta` — reference proteome UP000005640 (taxonomy 9606). This is the representative (canonical) sequence set, Swiss-Prot and TrEMBL. The `*_additional.fasta` isoform file is not used. Gunzip the `.fasta.gz`.

**Access:**

```text
https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.fasta.gz
```

7,752,225 bytes gzipped on 28 August 2026. The `current_release` path is updated with each UniProt release. Archive the FASTA you used for a frozen embedding set.

**Runtime:** Step 01 calls GSNN `uniprot2symbol` / `symbol2uniprot` (network). Step 08a reads only the local FASTA.

**License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) on copyrightable UniProt content. [https://www.uniprot.org/help/license](https://www.uniprot.org/help/license). Attribute the UniProt Consortium.

**Used in:** `kg_construction_08a__embed_amino_acids.py`; ID mapping in `kg_construction_01__drug_interacts_protein.py`

---



## 6. OmniPath

**Description:** Literature-aggregated directed protein–protein signaling. Step 05 calls `omnipath.interactions.OmniPath().get(genesymbols=True)` and splits edges using `consensus_stimulation` / `consensus_inhibition` into `stimulates_`*,* `inhibits_`, and `interacts_`*.

That client method requests the web-service `**omnipath` dataset** (literature-supported interactions, in the spirit of Türei et al. 2016), not OmniPath’s other datasets (`dorothea`, `collectri`, `kinaseextra`, `ligrecextra`, etc.). The script does **not** pass `license=`, `resources=`, or a version pin.

**This project redistributes:** No.

**Files required:** None. `omnipath` (and `pypath-omnipath`) are in `environment.yml`. Network is required on first run. Archive the downloaded table (`omnipath__protein_interacts_protein.csv`) if you need a frozen graph.

**Access:** [https://omnipathdb.org/](https://omnipathdb.org/) — equivalent HTTP query: `https://omnipathdb.org/interactions?datasets=omnipath&genesymbols=1`

**Software (not the data):** the `omnipath` Python client is **MIT** ([saezlab/omnipath](https://github.com/saezlab/omnipath)). The database builder `pypath` is **GPLv3** ([saezlab/pypath](https://github.com/saezlab/pypath)).

### How OmniPath treats licenses

OmniPath does **not** license the assembled graph. From the project site: “The data redistributed by the OmniPath web service doesn't have a license, each original resource carries over its license.” The 2021 *Molecular Systems Biology* paper says the same and points at Dataset EV12; for-profit users are told to limit queries (Türei et al. 2021). Resource metadata and the licenses OmniPath recorded are at [https://omnipathdb.org/info](https://omnipathdb.org/info) and in [pypath `resources.json](https://github.com/saezlab/pypath/blob/master/pypath/resources/data/resources.json)`. How those strings were collected is described on the info page (webpage/FAQ search; if nothing was found they contacted authors or labeled **Unspecified NC-SA**).

The clients expose a `license=` filter (`academic`, `commercial` / `for_profit`, `nonprofit`). **Default is** `academic`, which OmniPath documents as returning **all** records the service will give you. `commercial` is supposed to keep only records supported by sources that pypath marks as allowing for-profit use (`purpose: commercial` in `pypath/data/licenses/*.json`).

This pipeline never sets `license=`. On **28 August 2026**, `license=academic`, `license=commercial`, `license=for_profit`, and `license=nonprofit` on `datasets=omnipath` each returned the **same 86,189 edges and the same 101 source tags**. Do not assume the live filter is dropping non-commercial sources; inspect the `sources` column.

**Composite tags** (underscore names) mean evidence routed through a second pipeline, e.g. `PhosphoSite_MIMP`, `SIGNOR_ProtMapper`, `HPRD_LRdb`. The table below assigns those tags to the **first** resource’s OmniPath license (HPRD, PhosphoSite, SIGNOR, …). An edge can list several tags; unique-edge counts therefore overlap and sum to more than 86,189.

### Snapshot of what this query returns (28 August 2026)


|                                                                 |        |
| --------------------------------------------------------------- | ------ |
| Edges                                                           | 86,189 |
| Source tags                                                     | 101    |
| Primary resources (after collapsing composites)                 | 72     |
| Edges that list at least one academic/NC or PhosphoNetworks tag | 24,179 |
| Edges that list **only** those restricted tags                  | 1,770  |
| Edges that also list at least one pypath-“commercial” source    | 84,419 |


Largest contributors (unique edges listing that resource): SIGNOR 65,554; HPRD 9,696; ProtMapper 8,274; Wang 7,916; SPIKE 7,163; PhosphoSitePlus 6,979.

### Resources pypath marks academic / non-profit (in this snapshot)

These are the licenses that, on pypath’s mapping, are **not** `purpose: commercial`. They are present in the default pull this repo uses.


| Primary resource               | Unique edges | OmniPath-stated license | pypath `purpose` | What pypath’s license file says                                                |
| ------------------------------ | ------------ | ----------------------- | ---------------- | ------------------------------------------------------------------------------ |
| HPRD                           | 9,696        | HPRD                    | academic         | Custom “HPRD License”; FAQ URL (site discontinued)                             |
| Wang (Human Signaling Network) | 7,916        | CC BY-NC 3.0            | academic         | Non-commercial                                                                 |
| PhosphoSitePlus                | 6,979        | CC BY-NC-SA 3.0         | academic         | NC + share-alike                                                               |
| KEA                            | 6,931        | Unspecified NC-SA       | academic         | Placeholder: no public license, or OmniPath has redistribution permission only |
| iPTMnet                        | 5,952        | CC BY-NC-SA 4.0         | academic         | NC + share-alike                                                               |
| HINT                           | 4,993        | Unspecified NC-SA       | academic         | Same placeholder as KEA                                                        |
| Cellinker                      | 3,763        | CC BY-NC 4.0            | academic         | Non-commercial                                                                 |
| phospho.ELM                    | 3,478        | ELM                     | academic         | [ELM academic license PDF](http://elm.eu.org/media/Elm_academic_license.pdf)   |
| iTALK                          | 2,532        | CC BY-NC-SA 3.0         | academic         | NC + share-alike                                                               |
| Baccin2019                     | 1,286        | CC BY-NC 4.0            | academic         | Non-commercial                                                                 |
| HPMR                           | 1,087        | CC BY-NC 4.0            | academic         | Non-commercial                                                                 |
| DEPOD                          | 828          | CC BY-NC 4.0            | academic         | Non-commercial                                                                 |
| PhosphoNetworks                | 609          | PhosphoNetworks         | non_profit       | Academic/non-profit free; commercial users must contact the authors            |
| DLRP                           | 534          | Unspecified NC-SA       | academic         | Placeholder                                                                    |
| ELM                            | 412          | ELM                     | academic         | Same ELM academic PDF as phospho.ELM                                           |
| LMPID                          | 242          | CC BY-NC 2.0            | academic         | Non-commercial                                                                 |
| Li2012                         | 177          | CC BY-NC 3.0            | academic         | Non-commercial                                                                 |
| TRIP                           | 136          | CC BY-NC-SA 3.0         | academic         | NC + share-alike                                                               |
| Kirouac2010                    | 128          | CC BY-NC-ND 3.0         | academic         | NC + no derivatives                                                            |
| PDZBase                        | 101          | Unspecified NC-SA       | academic         | Placeholder                                                                    |




### All primary resources in this snapshot

Licenses and “pypath purpose” below are **OmniPath/pypath’s recorded classification**, not a fresh legal review of every source site. Confirm on the source if you redistribute or use commercially. Unique-edge counts overlap.


| Primary resource            | Unique edges | OmniPath-stated license | pypath purpose              |
| --------------------------- | ------------ | ----------------------- | --------------------------- |
| SIGNOR                      | 65,554       | CC BY 4.0               | commercial                  |
| HPRD                        | 9,696        | HPRD                    | academic                    |
| ProtMapper                  | 8,274        | CC BY 4.0               | commercial                  |
| Wang                        | 7,916        | CC BY-NC 3.0            | academic                    |
| SPIKE                       | 7,163        | CC BY 4.0               | commercial                  |
| PhosphoSitePlus             | 6,979        | CC BY-NC-SA 3.0         | academic                    |
| KEA                         | 6,931        | Unspecified NC-SA       | academic                    |
| iPTMnet                     | 5,952        | CC BY-NC-SA 4.0         | academic                    |
| HINT                        | 4,993        | Unspecified NC-SA       | academic                    |
| MIMP                        | 4,612        | LGPLv3                  | commercial                  |
| NetworKIN                   | 3,838        | CC BY 4.0               | commercial                  |
| Cellinker                   | 3,763        | CC BY-NC 4.0            | academic                    |
| ConnectomeDB2025            | 3,519        | MIT                     | commercial                  |
| phospho.ELM                 | 3,478        | ELM                     | academic                    |
| CellTalkDB                  | 3,451        | GPLv3                   | commercial (copyleft)       |
| PhosphoPoint                | 3,225        | CC BY-SA 4.0            | commercial (share-alike)    |
| LRdb                        | 3,171        | GPLv3                   | commercial (copyleft)       |
| REACH                       | 2,581        | CC BY 4.0               | commercial                  |
| CellChatDB                  | 2,562        | GPLv3                   | commercial (copyleft)       |
| CellPhoneDB                 | 2,533        | MIT                     | commercial                  |
| iTALK                       | 2,532        | CC BY-NC-SA 3.0         | academic                    |
| FANTOM (tag `Fantom5_LRdb`) | 2,426        | CC BY 4.0               | commercial                  |
| talklr                      | 2,410        | GPLv3                   | commercial (copyleft)       |
| HuRI Lit-BM-17              | 2,382        | CC BY 4.0               | commercial                  |
| Sparser                     | 2,317        | CC BY 4.0               | commercial                  |
| connectomeDB2020            | 2,313        | MIT                     | commercial                  |
| Ramilowski2015              | 2,068        | CC BY 4.0               | commercial                  |
| IntAct                      | 2,004        | CC BY 4.0               | commercial                  |
| Cui2007                     | 1,893        | CC BY 4.0               | commercial                  |
| InnateDB                    | 1,837        | DSL                     | commercial (share-alike)    |
| SignaLink3                  | 1,799        | CC BY-SA 3.0            | commercial (share-alike)    |
| BioGRID                     | 1,773        | MIT                     | commercial                  |
| ICELLNET                    | 1,626        | GPLv3                   | commercial (copyleft)       |
| STRING                      | 1,353        | CC BY 4.0               | commercial                  |
| RLIMS-P                     | 1,306        | CC BY 4.0               | commercial                  |
| Baccin2019                  | 1,286        | CC BY-NC 4.0            | academic                    |
| EMBRACE                     | 1,203        | CC BY 4.0               | commercial                  |
| NetPath                     | 1,200        | CC BY 2.5               | commercial                  |
| ACSN                        | 1,129        | CC BY 4.0               | commercial                  |
| HPMR                        | 1,087        | CC BY-NC 4.0            | academic                    |
| Reactome                    | 1,049        | CC0 1.0                 | commercial                  |
| Adhesome                    | 1,008        | CC BY 4.0               | commercial                  |
| CA1                         | 951          | CC BY 4.0               | commercial                  |
| dbPTM                       | 873          | NAR Open Access         | commercial                  |
| Kinexus                     | 833          | CC BY 4.0               | commercial                  |
| DEPOD                       | 828          | CC BY-NC 4.0            | academic                    |
| CellCall                    | 815          | Artistic-2.0            | commercial                  |
| KEGG-MEDICUS                | 804          | CC BY-SA 4.0            | commercial (share-alike)    |
| DIP                         | 769          | CC BY-ND 3.0            | commercial (no derivatives) |
| Guide to Pharmacology       | 654          | CC BY-SA 3.0            | commercial (share-alike)    |
| PhosphoNetworks             | 609          | PhosphoNetworks         | non-profit                  |
| NCI-PID                     | 535          | BSD                     | commercial                  |
| DLRP                        | 534          | Unspecified NC-SA       | academic                    |
| Macrophage                  | 499          | CC BY 4.0               | commercial                  |
| Wojtowicz2020               | 456          | CC BY 4.0               | commercial                  |
| scConnect                   | 455          | MIT                     | commercial                  |
| ELM                         | 412          | ELM                     | academic                    |
| CancerCellMap               | 396          | CC BY 2.5               | commercial                  |
| DOMINO                      | 335          | CC BY 2.5               | commercial                  |
| BEL-Large-Corpus            | 303          | NAR Open Access         | commercial                  |
| UniProt                     | 262          | CC BY 4.0               | commercial                  |
| LMPID                       | 242          | CC BY-NC 2.0            | academic                    |
| HuRI                        | 241          | CC BY 4.0               | commercial                  |
| Li2012                      | 177          | CC BY-NC 3.0            | academic                    |
| TRIP                        | 136          | CC BY-NC-SA 3.0         | academic                    |
| Kirouac2010                 | 128          | CC BY-NC-ND 3.0         | academic                    |
| PDZBase                     | 101          | Unspecified NC-SA       | academic                    |
| NRF2ome                     | 99           | CC BY-SA 3.0            | commercial (share-alike)    |
| ARN                         | 91           | CC BY-SA 3.0            | commercial (share-alike)    |
| MPPI                        | 86           | MPPI                    | commercial                  |
| AlzPathway                  | 31           | CC BY 3.0               | commercial                  |
| DeathDomain                 | 6            | NAR Open Access         | commercial                  |


Notes on a few labels:

- **NAR Open Access:** pypath treats Nucleic Acids Research database-issue resources as usable commercially (“freely usable”). That is OmniPath’s rule of thumb, not a CC license on every such database.
- **KEGG-MEDICUS:** OmniPath records **CC BY-SA 4.0** for this MEDICUS subset. That is not the same as the general KEGG academic/FTP license. This query does not use the classic `KEGG` resource (pypath files that under `pathway_noref`, without literature references).
- **Unspecified NC-SA:** not a real SPDX license. OmniPath uses it when terms were unclear or they only have permission to redistribute.
- **GPLv3 / LGPLv3 / Artistic / DSL** on interaction databases are the licenses OmniPath attached (often the project’s software license). Copyleft/share-alike may still matter if you redistribute a derived graph.



**Used in:** `kg_construction_05__protein_interacts_protein.py`

---



## 7. PubChem

**Description:** Name-to-structure lookup (canonical SMILES, InChIKey).

**This project redistributes:** No PubChem dump. InChIKeys and names appear in the companion Targetome tables (those identifiers are documented there as PubChem-derived).

**Files required:** None. `pubchempy` is in `environment.yml`.

- Step 01: `pubchempy.get_properties` (first hit per name), cached at `extdata/tg_inchikeys.csv`
- Step 03: PUG REST via `tkgdti.data.utils.get_smiles_inchikey` (first hit), cached in the CTD overlap file

**License:** Information created by or for the US government on NCBI sites is public domain ([NCBI policies](https://www.ncbi.nlm.nih.gov/home/about/policies/)). PubChem also contains third-party deposits with their own terms ([PubChem downloads](https://pubchem.ncbi.nlm.nih.gov/docs/downloads)).

**Usage:** PubChem asks that clients not exceed **5 requests per second**. Step 03 iterates many CTD names and is slow; keep the caches.

**Used in:** `kg_construction_01__drug_interacts_protein.py`, `kg_construction_03__drug_associates_disease.py`

---



## 8. jGlaser binding-affinity dataset

**Description:** Protein sequence / ligand SMILES pairs with experimental affinities. Used **only** to train and apply an auxiliary affinity model (steps 09a–09d). Predicted strong-binder edges are written into the **training** graph. That auxiliary model’s **weights are not released**. The jGlaser tables are not redistributed.

**This project redistributes:** No.

**Access:** [https://huggingface.co/datasets/jglaser/binding_affinity](https://huggingface.co/datasets/jglaser/binding_affinity)  

```python
from datasets import load_dataset
train = load_dataset("jglaser/binding_affinity", split="train[:90%]")
test  = load_dataset("jglaser/binding_affinity", split="train[90%:]")
```

The dataset card describes ~1.9 million unique pairs; the HuggingFace `train` split reports on the order of 3.7 million rows and ~1.25 GB. The first run caches under the HuggingFace datasets cache. These repos are ungated (no login as of this check). The `datasets` library is **not** listed in `environment.yml`. The captured `tkgdti` environment already has conda-forge `datasets` 3.6.0 (`pyhd8ed1ab_0`); see Reproducibility below. A fresh env built only from `environment.yml` needs `datasets` installed before step 09a.

Step 09a drops pairs that match known Targetome SMILES–sequence DTIs.

**License:** The dataset builder sets `_LICENSE = "BSD two-clause"`. That is script metadata, not a standalone LICENSE file. The card lists BindingDB, PDBbind-cn, BioLiP, and BindingMOAD. Those sources keep their own terms, including:


| Source             | Notes (verify on the source site)                                                                                                                                           |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BindingDB          | BindingDB-curated records: CC BY 4.0 (older site text: CC BY 3.0). ChEMBL-derived BindingDB records: CC BY-SA 3.0. [https://www.bindingdb.org/](https://www.bindingdb.org/) |
| PDBbind / PDBbind+ | Restrictive terms; redistribution generally requires permission. [https://www.pdbbind-plus.org.cn/termofuse](https://www.pdbbind-plus.org.cn/termofuse)                     |
| BioLiP             | Code/database described as BSD by the authors. [https://zhanggroup.org/BioLiP/](https://zhanggroup.org/BioLiP/)                                                             |
| BindingMOAD        | Project sunset; later commercial licensing of backend data has been reported. Confirm rights before any redistribution.                                                     |


**Used in:** `kg_construction_09a__binding_affinity_make.py` through `kg_construction_09d__predicted_binding_affinity.py`

---



## Pre-trained models (HuggingFace)

Downloaded on first use into the HuggingFace cache (often `~/.cache/huggingface/`). The checkpoints below were ungated as of this check.

**This project redistributes:** No. Users download the published checkpoints. This project does not release fine-tuned copies.

Workflow `config.yaml` files in this repo set:

- `shared.prot_model_name`: `facebook/esm2_t30_150M_UR50D`
- `shared.smiles_model_name`: `yzimmermann/ChemBERTa-zinc-base-v1-safetensors`

If you run the Python scripts **without** those flags, argparse defaults differ (`Rostlab/prot_bert`; `yzimmermann/ChemBERTa-77M-MLM-safetensors` in several scripts). The Snakefiles pass the `shared.`* values from config.

### ChemBERTa (SMILES embeddings)

Used for drug–drug similarity (step 02) and affinity-model features (step 09a). Two generations of ChemBERTa exist; this repo can load either.

**ChemBERTa (2020) — workflow default (**`…zinc-base-v1…`**)**

- **Paper:** Chithrananda, S., Grand, G. & Ramsundar, B. ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction. arXiv:2010.09885 (2020). [https://doi.org/10.48550/arXiv.2010.09885](https://doi.org/10.48550/arXiv.2010.09885)
- **What the paper states (no license named):** RoBERTa-style MLM on SMILES. Quote: “we have released 15 pre-trained ChemBERTa models on the Huggingface’s model hub.” The authors also “release a curated dataset of 77M SMILES from PubChem.” The PDF does **not** name MIT, Apache, or any other software license.
- **Code + weights project:** [bert-loves-chemistry](https://github.com/seyonechithrananda/bert-loves-chemistry) is **MIT** (Copyright 2020 Seyone Chithrananda; `LICENSE` and README “License: MIT License”). The README states that weights pretrained on ZINC (100k, 250k) and PubChem (100k–10M) are on [https://huggingface.co/seyonec](https://huggingface.co/seyonec), and gives `seyonec/ChemBERTa-zinc-base-v1` as the load example. Training notes there: RoBERTa MLM on **ZINC 250k**, ~10 epochs, until loss ~0.26.
- **Hub ID used here:** `yzimmermann/ChemBERTa-zinc-base-v1-safetensors` — safetensors conversion of `seyonec/ChemBERTa-zinc-base-v1`. The conversion card does **not** declare a license. The original project’s MIT file covers the software and is the only license the authors attached to that release.

**ChemBERTa-2 (2022) — optional (**`…77M-MLM…`**)**

- **Paper:** Ahmad, W., Simon, E., Chithrananda, S., Grand, G. & Ramsundar, B. ChemBERTa-2: Towards Chemical Foundation Models. arXiv:2209.01712 (2022). [https://doi.org/10.48550/arXiv.2209.01712](https://doi.org/10.48550/arXiv.2209.01712)
- **What the paper states (no license named):** MLM and multi-task regression on the same **77M unique PubChem SMILES** set as ChemBERTa-1. Quote: “We open source the trained models from this project.” The PDF does **not** name a license.
- **Hub ID:** `DeepChem/ChemBERTa-77M-MLM` (this repo may load `yzimmermann/ChemBERTa-77M-MLM-safetensors`). The DeepChem Hub page has **no model card and no license tag**. DeepChem library code is **MIT**. The bert-loves-chemistry README asks users of those models/notebooks to cite the ChemBERTa-2 paper.

**Used in:** `kg_construction_02__drug_similar_drug.py`, `kg_construction_09a__binding_affinity_make.py`

### ESM-2 (protein embeddings; workflow default)

- **Paper:** Lin, Z. et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* **379**, 1123–1130 (2023). [https://doi.org/10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)  
Preprint: bioRxiv 10.1101/2022.07.20.500902. Weights archive: [https://doi.org/10.5281/zenodo.7566741](https://doi.org/10.5281/zenodo.7566741)
- **Checkpoint used here:** `facebook/esm2_t30_150M_UR50D` — 30 layers, 150M parameters, trained on UniRef50/D (2021_04), as listed in [facebookresearch/esm](https://github.com/facebookresearch/esm).
- **What the *Science* paper states (Data and materials availability):** “ESM-2 and ESMFold model source code and parameters are available at [https://github.com/facebookresearch/esm](https://github.com/facebookresearch/esm) and archived on Zenodo (57). … Models and data have been released under permissive licenses (**MIT license for model source code and parameters** and **CC-BY for predicted structures**).”
- **Repo / Hub:** The GitHub README licenses “This source code … under the MIT license found in the `LICENSE` file” (Copyright Meta Platforms, Inc. and affiliates) and separately licenses the ESM Metagenomic Atlas under **CC BY 4.0**. HuggingFace tags `facebook/esm2_t30_150M_UR50D` as `license:mit`.
- **Scope:** This pipeline uses the ESM-2 **language-model checkpoint**, not Atlas structures.

**Used in:** `kg_construction_08a__embed_amino_acids.py`, `kg_construction_09a__binding_affinity_make.py`

### ProtBert (protein embeddings; optional, not the workflow default)

- **Paper:** Elnaggar, A. et al. ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning. *IEEE Trans. Pattern Anal. Mach. Intell.* **44**, 7112–7127 (2022). [https://doi.org/10.1109/TPAMI.2021.3095381](https://doi.org/10.1109/TPAMI.2021.3095381) (PMID 34232869). Preprint: arXiv:2007.06225.
- **What the paper states (no model license named):** ProtBert is the BERT protein LM in the ProtTrans suite, pretrained on UniRef100 (217 million sequences; see the `Rostlab/prot_bert` card). Closing sentence: “All our models are available through [https://github.com/agemagician/ProtTrans.”](https://github.com/agemagician/ProtTrans.”) The IEEE PDF does **not** name AFL, MIT, or any other license for the weights.
- **Model license as stated by the authors:** The ProtTrans README License section: “The ProtTrans pretrained models are released under the under terms of the [Academic Free License v3.0](https://choosealicense.com/licenses/afl-3.0/).” The same README asks users of the code or pretrained models to cite the IEEE paper. Weights: `Rostlab/prot_bert` (no license tag on the Hub card) and [Zenodo 4633691](https://zenodo.org/record/4633691).
- **Caveat:** The `LICENSE` file in [agemagician/ProtTrans](https://github.com/agemagician/ProtTrans) currently contains MIT text (Copyright 2025 Ahmed Elnaggar), and GitHub’s license badge follows that file. That disagrees with the README’s AFL-3.0 statement for the **pretrained models**. For compliance, treat the README’s AFL-3.0 line as the authors’ statement about the weights unless they clarify otherwise.

**Used in:** same scripts as ESM-2, only if `shared.prot_model_name` / `--model_name` is set to `Rostlab/prot_bert`.

---



## Benchmark workflows (not used by `full-tkg` / `aml-tkg`)



### 9. OGB BioKG (`workflow/biokg/`)

**This project redistributes:** No.

`ogbl-biokg` is downloaded by `ogb.linkproppred.LinkPropPredDataset` into `dirs.data_root` (`make_biokg.py`). Package: `ogb` in `environment.yml`.

OGB documents the **benchmark compilation** as CC0 ([https://ogb.stanford.edu/docs/linkprop/](https://ogb.stanford.edu/docs/linkprop/)). Underlying biomedical sources have their own licenses.

### 10. DTINet HeteroA (`workflow/hetero-a/`)

**This project redistributes:** No.

Default URL in `make_heteroa.py` / `step_01_heteroa.url`:

```text
https://github.com/luoyunan/PyDTINet/raw/main/data.tar.gz
```

That URL returned HTTP 200 with Content-Length 9,131,801 bytes on 28 August 2026. Matrices also exist under [https://github.com/luoyunan/DTINet/tree/master/data](https://github.com/luoyunan/DTINet/tree/master/data).

DTINet is **GPL-3.0**; PyDTINet is **MIT**. Source databases include DrugBank (**CC BY-NC 4.0** without a commercial DrugBank license) and others. Treat commercial use as restricted unless you have rights to every source.

---



## Directory layout (`full-tkg` / `aml-tkg`)

```
data/
├── targetome_extended-01-23-25.csv
├── targetome_extended_drugs-01-23-25.csv
├── beataml_drug_families.xlsx
├── CTD_chemicals_diseases.csv
├── CTD_curated_genes_diseases.csv
├── CTD_genes_pathways.csv
├── siginfo_beta.txt
├── compoundinfo_beta.txt
├── geneinfo_beta.txt
├── level5_beta_trt_cp_n720216x12328.gctx
└── UP000005640_9606.fasta
```

Set `dirs.data_root` to this directory. HuggingFace and OGB caches stay in their defaults unless you set `HF_HOME`, `HF_DATASETS_CACHE`, or the OGB root.

---



## Runtime, storage, and compute

- **Network on first run:** OmniPath, PubChem, UniProt ID mapping, HuggingFace, and (if used) OGB / DTINet. Offline reruns need those caches.
- **Disk:** LINCS GCTX is 35,518,405,386 bytes. Uncompressed `CTD_chemicals_diseases.csv` is much larger than its gzip. Add HuggingFace caches and `dirs.out_root`.
- **GPU:** Steps 08a and 09a–09c benefit from a GPU. ComplEx² / GNN training uses GPU unless `use_cpu` is set.
- **Reproducibility:** CTD (monthly), UniProt `current_release`, and the OmniPath API are live resources. Keep copies and dates of what you downloaded.

**Conda environment (`tkgdti`).** Reported runs used `/home/exacloud/gscratch/mcweeney_lab/evans/external/miniforge3/envs/tkgdti` (linux-64). It was created on 12 August 2025 with `mamba create -f environment.yml` (miniforge3; conda 24.11.0). `environment.yml` does not pin versions. The resolved environment, exported 20 October 2026 with `conda env export -n tkgdti` (machine `prefix` removed), is [`environment.lock.yml`](environment.lock.yml). Channel order in that export is `pyg`, `pytorch`, `nvidia`, `defaults`, `conda-forge`, `nodefaults`.

Packages that define the runtime:

| Package | Version | Build | Channel |
| --- | --- | --- | --- |
| python | 3.9.19 | `h0755675_0_cpython` | conda-forge |
| pytorch | 2.4.1 | `py3.9_cuda12.4_cudnn9.1.0_0` | pytorch |
| torchvision | 0.19.1 | `py39_cu124` | pytorch |
| pytorch-cuda | 12.4 | `hc786d27_7` | pytorch |
| pyg | 2.6.1 | `py39_torch_2.4.0_cu124` | pyg |
| cuda-runtime | 12.4.1 | `0` | nvidia |
| numpy | 1.26.4 | `py39h474f0d3_0` | conda-forge |
| pandas | 2.2.3 | `py39h3b40f6f_1` | conda-forge |
| scipy | 1.13.1 | `py39haf93ffa_0` | conda-forge |
| scikit-learn | 1.6.1 | `py39h4b7350c_0` | conda-forge |
| matplotlib-base | 3.3.2 | `py39h98787fa_1` | conda-forge |
| seaborn-base | 0.13.0 | `pyhd8ed1ab_0` | conda-forge |
| transformers | 4.55.0 | `pyhd8ed1ab_0` | conda-forge |
| tokenizers | 0.21.4 | `py39h4a49e08_0` | conda-forge |
| huggingface_hub | 0.34.4 | `pyhd8ed1ab_0` | conda-forge |
| datasets | 3.6.0 | `pyhd8ed1ab_0` | conda-forge |
| h5py | 3.13.0 | `nompi_py39h30a5a8d_100` | conda-forge |
| statsmodels | 0.14.5 | `py39h7c0b2d6_0` | conda-forge |
| networkx | 3.2 | `pyhd8ed1ab_0` | conda-forge |
| pyarrow | 16.1.0 | `py39h8003fee_1` | conda-forge |
| pubchempy | 1.0.4 | `pyhd8ed1ab_1` | conda-forge |
| rdkit | 2025.3.5 | — | pypi |
| biopython | 1.85 | — | pypi |
| omnipath | 1.0.12 | — | pypi |
| pypath-omnipath | 0.16.20 | — | pypi |
| ogb | 1.3.6 | — | pypi |
| snakemake | 7.32.4 | — | pypi |
| umap-learn | 0.5.9.post2 | — | pypi |

Every other installed package, including build strings, is in `environment.lock.yml`.

Four pip entries in that file are editable checkouts, not PyPI releases. `mamba env create -f environment.lock.yml` fails on them. Remove `tkgdti==0.1`, `gsnn==0.2`, `complex2==0.1`, and `hnet==0.1` from the pip section, create the env, then install the checkouts that were on `sys.path` when this snapshot was taken:

- `tkgdti` 0.1 — this repository, `pip install -e .` (editable; the running code is the working tree, not a frozen tag)
- `GSNN` 0.2 — `/home/exacloud/gscratch/mcweeney_lab/evans/GSNN`, commit `aacb74bc6572` (branch `dev2`; uncommitted changes were present)
- `complex2` 0.1 — `/home/exacloud/gscratch/mcweeney_lab/evans/complex2`, commit `8257454a4652` (uncommitted changes were present)
- `hnet` 0.1 — `/home/exacloud/gscratch/mcweeney_lab/evans/hypernet`, commit `bfd2830be6ff`

---



## License notes (reuse and commercial use)

GPL-3.0 on this software does not replace dataset licenses. This project does not redistribute the training databases. If **you** publish a graph, embeddings, or weights built from them, you must satisfy each source. Items that commonly need a second look:

1. **CTD** — commercial download/use requires a paid license.
2. **Companion Targetome tables** — Figshare item license CC BY 4.0 (attribution). [https://doi.org/10.6084/m9.figshare.34367808.v1](https://doi.org/10.6084/m9.figshare.34367808.v1). The `LICENSE.txt` in that deposit is still the CC BY-SA 4.0 legal text.
3. **OmniPath** — no umbrella license. The default `datasets=omnipath` pull includes HPRD, PhosphoSitePlus, Wang, iPTMnet, HINT, ELM/phospho.ELM, and other NC or unspecified-NC sources (see §6). This pipeline does not set `license=`. A `license=commercial` query on 28 Aug 2026 did not change the result.
4. **jGlaser / PDBbind** — PDBbind terms restrict redistribution; ChEMBL share-alike may apply to subsets.
5. **HeteroA / DrugBank** — DrugBank is non-commercial without a separate license.
6. **ProtBert** (only if selected) — authors state AFL-3.0 for pretrained models; see the ProtBert subsection (README vs `LICENSE` file).
7. **ChemBERTa Hub conversions** (`yzimmermann/…`) — no license on the conversion cards; cite and follow the original ChemBERTa / ChemBERTa-2 projects.

---



## Citations

Cite every resource you used. For CTD, UniProt, and PubChem, use the citation requested on those sites (they update the preferred paper).

- **This software:** [https://doi.org/10.5281/zenodo.18809293](https://doi.org/10.5281/zenodo.18809293)
- **Expanded Targetome (companion data + prediction subset):** Bottomly, D., Evans, N. & McWeeney, S. K. Expanding the set of high evidence drug-target interactions in the Cancer Targetome. Submitted. Dataset: Bottomly, D., Evans, N. & McWeeney, S. (2026). Targetome-Expanded Release. figshare. [https://doi.org/10.6084/m9.figshare.34367808.v1](https://doi.org/10.6084/m9.figshare.34367808.v1)
- **Cancer Targetome:** Blucher et al., *Trends Pharmacol. Sci.* 38:1085–1099 (2017)
- **BeatAML 2.0:** Bottomly et al., *Cancer Cell* 40:850–864.e9 (2022)
- **CTD:** [https://ctdbase.org/about/publications/#citing](https://ctdbase.org/about/publications/#citing)
- **LINCS L1000:** Subramanian et al., *Cell* 171:1437–1452.e17 (2017), and the LINCS 2020 / GEO accession you actually downloaded
- **UniProt:** [https://www.uniprot.org/help/publications](https://www.uniprot.org/help/publications)
- **OmniPath (2016 dataset definition):** Türei, D., Korcsmáros, T. & Saez-Rodriguez, J. OmniPath: guidelines and gateway for literature-curated signaling pathway resources. *Nat. Methods* **13**, 966–967 (2016). [https://doi.org/10.1038/nmeth.4077](https://doi.org/10.1038/nmeth.4077)
- **OmniPath (current resource/license model):** Türei, D. et al. Integrated intra- and intercellular signaling knowledge for multicellular omics analysis. *Mol. Syst. Biol.* **17**, e9923 (2021). [https://doi.org/10.15252/msb.20209923](https://doi.org/10.15252/msb.20209923) (PMID 33749993). Resource licenses: [https://omnipathdb.org/info](https://omnipathdb.org/info)
- **PubChem:** [https://pubchem.ncbi.nlm.nih.gov/docs/citations-and-acknowledgements](https://pubchem.ncbi.nlm.nih.gov/docs/citations-and-acknowledgements)
- **ChemBERTa (zinc-base-v1):** Chithrananda, S., Grand, G. & Ramsundar, B. ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction. arXiv:2010.09885 (2020). [https://doi.org/10.48550/arXiv.2010.09885](https://doi.org/10.48550/arXiv.2010.09885)
- **ChemBERTa-2 (77M-MLM, if used):** Ahmad, W., Simon, E., Chithrananda, S., Grand, G. & Ramsundar, B. ChemBERTa-2: Towards Chemical Foundation Models. arXiv:2209.01712 (2022). [https://doi.org/10.48550/arXiv.2209.01712](https://doi.org/10.48550/arXiv.2209.01712). The ChemBERTa GitHub README also asks users of that project’s models/notebooks to cite this paper.
- **ESM-2:** Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., Smetanin, N., Verkuil, R., Kabeli, O., Shmueli, Y., dos Santos Costa, A., Fazel-Zarandi, M., Sercu, T., Candido, S. & Rives, A. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* **379**, 1123–1130 (2023). [https://doi.org/10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574). Weights: [https://doi.org/10.5281/zenodo.7566741](https://doi.org/10.5281/zenodo.7566741)
- **ProtBert / ProtTrans (if used):** Elnaggar, A., Heinzinger, M., Dallago, C., Rehawi, G., Yu, W., Jones, L., Gibbs, T., Feher, T., Angerer, C., Steinegger, M., Bhowmik, D. & Rost, B. ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning. *IEEE Trans. Pattern Anal. Mach. Intell.* **44**, 7112–7127 (2022). [https://doi.org/10.1109/TPAMI.2021.3095381](https://doi.org/10.1109/TPAMI.2021.3095381)
- **jGlaser compilation:** [https://huggingface.co/datasets/jglaser/binding_affinity](https://huggingface.co/datasets/jglaser/binding_affinity) (cite BindingDB / PDBbind / BioLiP / BindingMOAD if you rely on those sources)

