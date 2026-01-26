# TKG-DTI: Targetome Knowledge Graph for Drug-Target Interaction Prediction

> **Corresponding author:** Nathaniel Evans (`evansna@ohsu.edu`)

TKG-DTI is a knowledge graph-based framework for predicting high-confidence drug-target interactions (DTIs) in cancer therapeutics. The system integrates multiple biomedical data sources into a heterogeneous knowledge graph and uses graph-based machine learning models (ComplEx² and Path-GNN) to predict novel DTIs.

## Features

- **Knowledge Graph Construction**: Automated pipeline to build heterogeneous biomedical knowledge graphs from multiple curated sources (Targetome, CTD, OmniPath, LINCS L1000)
- **ComplEx² Model**: Heterogeneous adaptation of the ComplEx² knowledge graph embedding model with normalized log-probabilities
- **Path-GNN Model**: Diffusion-based graph neural network for interpretable DTI predictions
- **Cross-Validation**: K-fold cross-validation framework with aggregated prediction scoring
- **Reproducible Workflows**: Snakemake pipelines for end-to-end reproducibility

## Installation

```bash
# Clone the repository
git clone https://github.com/biodev/TKG-DTI.git
cd TKG-DTI

# Create conda environment
mamba env create -f environment.yml
mamba activate tkgdti

# Install the package
pip install -e .
```

There are a few helper functions that we borrow from the GSNN [repo](https://github.com/nathanieljevans/GSNN), in the future we will remove this dependency. To install GSNN: 

```bash 
$ pip install git+https://github.com/nathanieljevans/GSNN 
```

## Quick Start

### 1. Download Required Data

See [data_availability.md](data_availability.md) for complete data requirements.

```bash
# Download publicly available datasets (CTD, UniProt, BeatAML)
# Edit ROOT path in script first
bash scripts/get_tkg_raw_files.sh
```

**Note:** The Expanded Cancer Targetome and LINCS L1000 data require separate download. See [data_availability.md](data_availability.md) for access instructions.

### 2. Configure Workflow

Edit the configuration file for your workflow:

```bash
# Edit paths and parameters
vim workflow/full-tkg/config.yaml
```

### 3. Run the Pipeline

```bash
cd workflow/full-tkg
snakemake -j 1  # Use -j N for N parallel jobs
```

## Project Structure

```
TKG-DTI/
├── tkgdti/                 # Python package
│   ├── data/               # Data loading and graph construction
│   ├── models/             # ComplEx² and GNN model implementations
│   ├── train/              # Training utilities
│   ├── eval/               # Evaluation metrics
│   └── embed/              # Drug/protein embedding utilities
├── workflow/               # Snakemake workflows
│   ├── full-tkg/           # Full TKG workflow
│   ├── aml-tkg/            # AML-focused workflow
│   ├── hetero-a/           # HeteroA baseline
│   └── scripts/            # KG construction scripts (steps 01-10)
├── scripts/                # Utility scripts
├── docs/                   # Documentation
├── environment.yml         # Conda environment specification
└── data_availability.md    # Data sources and access instructions
```

## Documentation

- [TKG-DTI Methods](docs/tkgdti_methods.md) — Detailed workflow and model documentation
- [GNN Architecture](docs/gnn.md) — Path-based GNN approach for interpretable predictions
- [Aggregation Guide](docs/full-tkg-agg.md) — Cross-fold prediction aggregation and filtering
- [KG Design](docs/tkg.md) — Original knowledge graph design proposal
- [Data Availability](data_availability.md) — Required datasets and access instructions

## Citation

If you use TKG-DTI in your research, please cite:

> citation coming soon. 

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

This work was supported by the OHSU Knight Cancer Institute and the [BioDev](https://github.com/biodev) Lab (PI: Dr. Shannon McWeeney). 

## Contact

For questions or issues, please open a GitHub issue or contact the corresponding author at `evansna@ohsu.edu`.
