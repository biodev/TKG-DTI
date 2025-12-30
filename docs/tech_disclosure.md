# Technology Disclosure: Full-TKG Drug-Target Interaction Prediction System

## A. TECHNOLOGY

**Title:** Targetome Knowledge Graph System for Drug-Target Interaction Prediction and Discovery

---

### 1. Key Features and Benefits

| Feature | Description | Novel? |
|---------|-------------|--------|
| Heterogeneous Knowledge Graph Construction | Integrates multiple biological data sources (protein-protein interactions, drug-disease associations, pathway memberships, expression perturbations, molecular similarities) into a unified heterogeneous graph representation | |
| ComplEx² Adaptation | Heterogeneous adaptation of the ComplEx² generative knowledge graph embedding model that enables probabilistic link prediction with normalized log-probabilities per relation type | |
| Diffusion-based GNN | Path-based graph neural network architecture that models drug-to-target signal propagation as a diffusion process, enabling explainable predictions | |
| Cross-validation Aggregation | Multi-fold cross-validation with statistical aggregation to estimate prediction confidence using false negative rate (FNR) and false positive rate (FPR) calibration | |
| Predicted Binding Affinity Integration | Incorporates machine learning-predicted binding affinities as auxiliary relation types to enhance DTI prediction | |
| End-to-end Workflow | Fully automated Snakemake pipeline from raw data processing through model training and prediction aggregation | |

**Key Benefits:**
- Enables discovery of novel drug-target interactions by leveraging multiple lines of biological evidence
- Provides confidence estimates for predictions through statistical calibration across cross-validation folds
- Generates explainable predictions through path-based reasoning in the knowledge graph
- Reduces experimental screening costs by prioritizing high-confidence candidates

---

### 2. Overview of Technology

**Name:** TKG-DTI (Targetome Knowledge Graph for Drug-Target Interaction Prediction)

**Description:** TKG-DTI is a computational pipeline for predicting drug-target interactions using heterogeneous knowledge graphs. The system constructs a multi-relational graph from curated biological databases and uses machine learning models to predict missing links between drugs and protein targets.

The workflow consists of three major phases:
1. **Knowledge Graph Construction:** Automated scripts integrate relations from multiple sources including the Cancer Targetome, OmniPath, CTD (Comparative Toxicogenomics Database), and LINCS perturbation signatures.
2. **Model Training:** Two complementary models are trained—a knowledge graph embedding model (ComplEx²) and a graph neural network (GNN)—using k-fold cross-validation.
3. **Prediction Aggregation:** Predictions are aggregated across folds with statistical calibration to generate ranked candidate DTIs with confidence metrics.

---

### 3. Technical Description

**Knowledge Graph Structure:**
The knowledge graph is a heterogeneous multigraph with node types including drugs, genes/proteins, diseases, and pathways. Relation types include:
- Drug-targets-Gene (primary prediction target)
- Drug-associates-Disease / Gene-associates-Disease
- Gene-interacts/stimulates/inhibits-Gene (from OmniPath)
- Gene-is_in-Pathway
- Drug-similar_to-Drug (ChemBERTa embedding similarity)
- Gene-similar_to-Gene (ProtBert/ESM2 embedding similarity)
- Drug-perturbs_expression-Gene (LINCS L1000)
- Drug-predicted_binding-Gene (ML-predicted binding affinity)

**Model 1: ComplEx² (Knowledge Graph Embedding)**
A heterogeneous adaptation of ComplEx² ([Loconte et al., 2023](https://arxiv.org/abs/2305.15944)) that learns complex-valued embeddings for nodes and relations. The model computes a tractable partition function conditioned on relation type, enabling normalized log-probability scores for link prediction. Key adaptations include separate embedding tables per node type and relation-conditioned scoring.

**Model 2: Path-based GNN**
A graph neural network that formulates DTI prediction as node classification rather than link prediction. For each drug, a forward pass activates only that drug node and propagates signals through the knowledge graph via graph attention convolutions. The resulting protein node embeddings encode path-based evidence from drug to protein. This design enables:
- Explainable predictions through path analysis
- Tractable prediction of all DTIs for a drug in a single forward pass
- Compatibility with GNN explanation methods

**Cross-validation and Aggregation:**
The target DTI relation is split into k folds (typically 10), with all auxiliary relations present in every fold. For each fold, models predict scores for held-out DTIs and candidate negatives. Per-observation FNR/FPR estimates are computed using Gaussian Process interpolation, enabling statistically calibrated filtering of high-confidence predictions.

---

### 4. Advantages Over Other Technologies

| Competing Approach | Limitation | TKG-DTI Advantage |
|-------------------|------------|-------------------|
| Single-relation link prediction (e.g., standard ComplEx, DistMult) | Cannot leverage heterogeneous biological context | Integrates multiple relation types and data modalities |
| Molecular fingerprint similarity | Limited to chemical structure; ignores biological context | Incorporates pathway, disease, expression, and interaction data |
| Sequence-based deep learning (e.g., DeepDTA) | Requires protein sequence/structure; no graph context | Leverages relational knowledge graph structure |
| Standard GNN link prediction | Requires embedding comparison for each drug-protein pair | Diffusion formulation enables single-pass prediction per drug |
| Black-box models | No interpretability | Path-based GNN enables explainable predictions |

**Additional Advantages:**
- **Probabilistic interpretation:** ComplEx² provides normalized probabilities, unlike most KGE models that produce unbounded scores
- **Scalability:** Prediction scales with number of drugs rather than drug×protein combinations
- **Modularity:** Pipeline architecture allows easy addition/removal of relation types for ablation studies
- **Reproducibility:** Snakemake workflow ensures reproducible end-to-end execution

---

### 5. Commercialization Potential

*Potential applications (speculative):*

- **Pharmaceutical R&D:** Early-stage drug target identification and validation prioritization
- **Drug repurposing:** Identification of novel targets for existing approved drugs
- **Academic licensing:** Software-as-a-service or licensed platform for academic research institutions
- **Data integration service:** Custom knowledge graph construction for proprietary compound libraries

---

## References and Licensing

**Core Dependencies:**
- PyTorch Geometric: MIT License ([GitHub](https://github.com/pyg-team/pytorch_geometric))
- ChemBERTa: MIT License ([GitHub](https://github.com/seyonechithrananda/bert-loves-chemistry))
- ProtBert: BSD-3 License
- ESM2: MIT License (Meta Platforms, Inc.)

**Key Papers:**
- ComplEx²: Loconte, L. et al. "How to Turn Your Knowledge Graph Embeddings into Generative Models." arXiv:2305.15944 (2023)
- OmniPath: Türei, D. et al. "OmniPath: guidelines and gateway for literature-curated signaling pathway resources." Nature Methods (2016)
- PyTorch Geometric: Fey, M. & Lenssen, J.E. "Fast Graph Representation Learning with PyTorch Geometric." arXiv:1903.02428 (2019)

**Data Sources:**
- Comparative Toxicogenomics Database (CTD)
- LINCS L1000
- OmniPath (contact omnipathdb@gmail.com for database licensing)
