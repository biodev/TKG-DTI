> **Note:** This documentation was reviewed and edited with assistance from an LLM. Please verify critical details against the source code.

# Targetome Knowledge Graph

> **Important:** This document represents the original design proposal. Not all relation types listed below were implemented in the final system. See [tkgdti_methods.md](tkgdti_methods.md) for documentation of the implemented relations.

## Summary (Original Proposal)

Construct a knowledge graph using the cancer targetome and a number of additional relation types described in the table below. Some relations described below may be excluded depending on practical utility, ability to match identifiers, and the rationale inclusion balanced with the difficulty of inclusion. We will then perform hyperoptimization to select relations that maximize performance on a hold-out validation set (predicting targetome level 3 DTIs). The outcome of this project will be a set of DTIs that have similar attributes to the cancer targetome level 3 DTIs. We will initially focus on drugs used in the BeatAML project, and potentially expand to a wider selection depending on performance and future goals. 

## Proposed Relation Types

The table below shows originally proposed relation types. **Implemented relations are marked with ✓**.

---

| Source   | Relation             | Target                | Undirected? | Data Source(s)                            | Implemented |
|----------|----------------------|-----------------------|-------------|-------------------------------------------|-------------|
| Drug     | targets              | Protein               | No          | Targetome \[1\]                           | ✓           |
| Drug     | similar-to           | Drug                  | Yes         | ChemBERTa embeddings                      | ✓           |
| Protein  | similar-to           | Protein               | Yes         | ProtBert/ESM2 embeddings                  | ✓           |
| Protein  | interacts-with       | Protein               | Yes         | OmniPath \[7\]                            | ✓           |
| Protein  | regulates            | Protein               | No          | OmniPath \[7\]                            | ✓           |
| Drug     | associated-with      | Disease               | Yes         | CTD                                       | ✓           |
| Protein  | associated-with      | Disease               | Yes         | CTD                                       | ✓           |
| Protein  | part-of              | Pathway               | No          | CTD                                       | ✓           |
| Drug     | perturbs-expression  | Protein               | No          | LINCS L1000                               | ✓           |
| Drug     | predicted-binding    | Protein               | No          | jGlaser model                             | ✓           |
| Protein  | located-in           | GO Cellular Component | No          | Gene Ontology \[13\]                      |             |
| Protein  | has-function         | GO Molecular Function | No          | Gene Ontology \[13\]                      |             |
| Protein  | involved-in          | GO Biological Process | No          | Gene Ontology \[13\]                      |             |
| Drug     | causes-side-effect   | Symptom               | Yes         | SIDER \[14\]; OFFSIDES \[15\]             |             |
| Disease  | has-symptom          | Symptom               | Yes         | Human Phenotype Ontology \[16\]           |             |
| Drug     | metabolized-by       | Protein               | No          | PharmGKB \[17\]; DrugBank \[3\]           |             |
| Drug     | inhibits             | Protein               | No          | ChEMBL \[20\]; BindingDB \[21\]           |             |
| Protein  | binds-to             | Protein               | Yes         | BioGRID \[22\]; IntAct \[23\]             |             |
| Drug     | transported-by       | Protein               | No          | TCDB \[24\]                               |             |
| cell-line| is-sensitive         | Drug                  | Yes         | CCLE, CTRP, NCI60                         |             |
| cell-line| has-high-expression  | Protein               | Yes         | CCLE                                      |             |
| cell-line| has-low-expression   | Protein               | Yes         | CCLE                                      |             |

---

**Protein/drug similarity (Implemented)**

Previous studies that have looked at simple structural similarity metrics (Tanimoto coefficient) showed limited predictive value for DTI prediction. The final implementation uses deep learning embeddings: ChemBERTa for drug SMILES and ProtBert/ESM2 for protein amino acid sequences. Similarity edges are created by thresholding cosine similarity between embeddings. 

---

**Drug sensitivity and genomic features (Not Implemented)**

This was a proposed idea to create paths from drug to protein via known genomic links and drug sensitivities. The rationale was that genomic markers/patterns causal of sensitivity are likely to have local impact from drug-targets. This approach was not included in the final implementation.

e.g., proposed path: "drug", "sensitivity-in", "cell-line", "overexpresses" "protein"

---

**Hyperoptimization of KG-relations (Partially Implemented)**

Understanding which relationships are useful for DTI prediction is challenging. The final implementation includes a relation ablation capability (`remove_relation_idx` parameter) to systematically test which relations contribute to predictive performance. Full Bayesian optimization across relation subsets was not implemented. 

---

**Use of multiple DTI resources (Partially Implemented)**

The primary purpose of this KG is to predict high-confidence DTI interactions using the cancer targetome. The final implementation includes predicted binding affinity edges from a trained model (jGlaser dataset), which provides auxiliary DTI-like information to augment the knowledge graph.

---

**Protein/drug localization (Not Implemented)**

Knowledge of where a protein is in the cell may help narrow the selection of functional drug targets. This feature was not included in the final implementation due to lack of comprehensive drug localization data. 

---

**References:**

\[1\] **Targetome**: Evans NJ, et al. The Cancer Targetome. [https://github.com/Targetome](https://github.com/Targetome)

\[2\] **STITCH**: Szklarczyk D, et al. STITCH 5: augmenting protein–chemical interaction networks with tissue and affinity data. *Nucleic Acids Research*, 2016;44(D1):D380–D384. [doi:10.1093/nar/gkv1277](https://doi.org/10.1093/nar/gkv1277)

\[3\] **DrugBank**: Wishart DS, et al. DrugBank 5.0: a major update to the DrugBank database for 2018. *Nucleic Acids Research*, 2018;46(D1):D1074–D1082. [doi:10.1093/nar/gkx1037](https://doi.org/10.1093/nar/gkx1037)

\[4\] **PubChem**: Kim S, et al. PubChem Substance and Compound databases. *Nucleic Acids Research*, 2016;44(D1):D1202–D1213. [doi:10.1093/nar/gkv951](https://doi.org/10.1093/nar/gkv951)

\[5\] **UniProt**: UniProt Consortium. UniProt: a worldwide hub of protein knowledge. *Nucleic Acids Research*, 2019;47(D1):D506–D515. [doi:10.1093/nar/gky1049](https://doi.org/10.1093/nar/gky1049)

\[6\] **BLAST**: Altschul SF, et al. Basic local alignment search tool. *Journal of Molecular Biology*, 1990;215(3):403–410. [doi:10.1016/S0022-2836(05)80360-2](https://doi.org/10.1016/S0022-2836(05)80360-2)

\[7\] **OmniPath**: Türei D, Korcsmáros T, Saez-Rodriguez J. OmniPath: guidelines and gateway for literature-curated signaling pathway resources. *Nature Methods*, 2016;13(12):966–967. [doi:10.1038/nmeth.4077](https://doi.org/10.1038/nmeth.4077)

\[8\] **STRING**: Szklarczyk D, et al. STRING v11: protein–protein association networks with increased coverage. *Nucleic Acids Research*, 2019;47(D1):D607–D613. [doi:10.1093/nar/gky1131](https://doi.org/10.1093/nar/gky1131)

\[9\] **RegNetwork**: Liu ZP, et al. Inference of gene regulatory network based on generalized conditional mutual information. *BMC Bioinformatics*, 2016;17(5):331. [doi:10.1186/s12859-016-1183-5](https://doi.org/10.1186/s12859-016-1183-5)

\[10\] **ClinicalTrials.gov**: U.S. National Library of Medicine. [https://clinicaltrials.gov/](https://clinicaltrials.gov/)

\[11\] **DisGeNET**: Piñero J, et al. DisGeNET: a comprehensive platform integrating information on human disease-associated genes and variants. *Nucleic Acids Research*, 2017;45(D1):D833–D839. [doi:10.1093/nar/gkw943](https://doi.org/10.1093/nar/gkw943)

\[12\] **GWAS Catalog**: Buniello A, et al. The NHGRI-EBI GWAS Catalog of published genome-wide association studies. *Nucleic Acids Research*, 2019;47(D1):D1005–D1012. [doi:10.1093/nar/gky1120](https://doi.org/10.1093/nar/gky1120)

\[13\] **Gene Ontology**: The Gene Ontology Consortium. The Gene Ontology Resource: 20 years and still GOing strong. *Nucleic Acids Research*, 2019;47(D1):D330–D338. [doi:10.1093/nar/gky1055](https://doi.org/10.1093/nar/gky1055)

\[14\] **SIDER**: Kuhn M, et al. The SIDER database of drugs and side effects. *Nucleic Acids Research*, 2016;44(D1):D1075–D1079. [doi:10.1093/nar/gkv1075](https://doi.org/10.1093/nar/gkv1075)

\[15\] **OFFSIDES**: Tatonetti NP, et al. Data-driven prediction of drug effects and interactions. *Science Translational Medicine*, 2012;4(125):125ra31. [doi:10.1126/scitranslmed.3003377](https://doi.org/10.1126/scitranslmed.3003377)

\[16\] **Human Phenotype Ontology**: Köhler S, et al. The Human Phenotype Ontology in 2017. *Nucleic Acids Research*, 2017;45(D1):D865–D876. [doi:10.1093/nar/gkw1039](https://doi.org/10.1093/nar/gkw1039)

\[17\] **PharmGKB**: Whirl-Carrillo M, et al. Pharmacogenomics knowledge for personalized medicine. *Clinical Pharmacology & Therapeutics*, 2012;92(4):414–417. [doi:10.1038/clpt.2012.96](https://doi.org/10.1038/clpt.2012.96)

\[18\] **Reactome**: Fabregat A, et al. The Reactome Pathway Knowledgebase. *Nucleic Acids Research*, 2018;46(D1):D649–D655. [doi:10.1093/nar/gkx1132](https://doi.org/10.1093/nar/gkx1132)

\[19\] **KEGG**: Kanehisa M, et al. KEGG: new perspectives on genomes, pathways, diseases and drugs. *Nucleic Acids Research*, 2017;45(D1):D353–D361. [doi:10.1093/nar/gkw1092](https://doi.org/10.1093/nar/gkw1092)

\[20\] **ChEMBL**: Gaulton A, et al. The ChEMBL database in 2017. *Nucleic Acids Research*, 2017;45(D1):D945–D954. [doi:10.1093/nar/gkw1074](https://doi.org/10.1093/nar/gkw1074)

\[21\] **BindingDB**: Gilson MK, et al. BindingDB in 2015: A public database for medicinal chemistry. *Nucleic Acids Research*, 2016;44(D1):D1045–D1053. [doi:10.1093/nar/gkv1072](https://doi.org/10.1093/nar/gkv1072)

\[22\] **BioGRID**: Oughtred R, et al. The BioGRID interaction database: 2019 update. *Nucleic Acids Research*, 2019;47(D1):D529–D541. [doi:10.1093/nar/gky1079](https://doi.org/10.1093/nar/gky1079)

\[23\] **IntAct**: Orchard S, et al. The MIntAct project—IntAct as a common curation platform for 11 molecular interaction databases. *Nucleic Acids Research*, 2014;42(D1):D358–D363. [doi:10.1093/nar/gkt1115](https://doi.org/10.1093/nar/gkt1115)

\[24\] **TCDB** (Transporter Classification Database): Saier MH Jr, et al. The Transporter Classification Database (TCDB): recent advances. *Nucleic Acids Research*, 2016;44(D1):D372–D379. [doi:10.1093/nar/gkv1103](https://doi.org/10.1093/nar/gkv1103)
