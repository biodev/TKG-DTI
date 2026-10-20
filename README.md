# TKG-DTI: Targetome Knowledge Graph for Drug-Target Interaction Prediction

![DOI](https://img.shields.io/badge/doi-10.6084%2Fm9.figshare.34367808-blue)

TKG-DTI is a knowledge graph-based framework for predicting high-confidence drug-target interactions (DTIs) in cancer therapeutics. The system integrates multiple biomedical data sources into a heterogeneous knowledge graph and uses graph-based machine learning models (ComplEx² and Path-GNN) to predict novel DTIs.

## Features

- **Knowledge Graph Construction**: Automated pipeline to build heterogeneous biomedical knowledge graphs from multiple curated sources (Targetome, CTD, OmniPath, LINCS L1000)
- **ComplEx² Model**: Heterogeneous adaptation of the ComplEx² knowledge graph embedding model with normalized log-probabilities. Used for all downstream predictions. 
- **Path-GNN Model**: Diffusion-based graph neural network for *interpretable* DTI predictions. Not used for downstream predictions. 
- **Cross-Validation**: K-fold cross-validation framework with aggregated prediction scoring
- **Reproducible Workflows**: Snakemake pipelines for end-to-end reproducibility

## Quick Start 

See `quickstart.md` for instructions on installing and running the `full-tkg` workflow. 

## Data

File names, licenses, and where to get each source are in [data_availability.md](data_availability.md).

`scripts/get_tkg_raw_files.sh` downloads the local inputs used by `workflow/full-tkg/` and `workflow/aml-tkg/`, except Targetome and LINCS. Set `ROOT` in that script to your data directory, then run it. It fetches:

- `beataml_drug_families.xlsx`, read by `aml-tkg` step 01
- `CTD_chemicals_diseases.csv`, `CTD_curated_genes_diseases.csv`, and `CTD_genes_pathways.csv`
- the UniProt reference proteome `UP000005640_9606.fasta`

It does not download the Expanded Cancer Targetome or LINCS L1000. LINCS locations are in [data_availability.md](data_availability.md).

To reproduce this workflow, download the pre-release Targetome tables from Figshare ([https://figshare.com/s/c207e1c7fabbcdacbf6d](https://figshare.com/s/c207e1c7fabbcdacbf6d)) and put them in the data directory set as `dirs.data_root`:

- `[targetome_extended_drugs-01-23-25.csv](https://figshare.com/ndownloader/files/70408458?private_link=c207e1c7fabbcdacbf6d)`
- `[targetome_extended-01-23-25.csv](https://figshare.com/ndownloader/files/70408455?private_link=c207e1c7fabbcdacbf6d)`



## Documentation

- [TKG-DTI Methods](docs/tkgdti_methods.md) --- Detailed workflow and model documentation
- [GNN Architecture](docs/gnn.md) --- Path-based GNN approach for interpretable predictions
- [Aggregation Guide](docs/full-tkg-agg.md) --- Cross-fold prediction aggregation and filtering
- [KG Design](docs/tkg.md) --- Original knowledge graph design proposal
- [Data Availability](data_availability.md) --- Required datasets and access instructions

## Citation

If you use TKG-DTI in your research, please cite:

> Bottomly, Daniel; Evans, Nathaniel; McWeeney, Shannon (2026). Targetome-Expanded Release. figshare. Dataset. [https://doi.org/10.6084/m9.figshare.34367808.v1](https://doi.org/10.6084/m9.figshare.34367808.v1)



## License

Copyright (C) 2026. Oregon Health & Science University Knight Cancer Institute

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

This work was supported by the Edward P. Evans Foundation (EvansMDS Artificial Intelligence and Machine Learning in MDS award)

## Contact

For questions or issues, please open a GitHub issue or contact the maintainer at `evansna@ohsu.edu`.