#!/bin/bash

ROOT=/home/exacloud/gscratch/mcweeney_lab/evans/data/

#Blucher AS, Choonoo G, Kulesz-Martin M, Wu G, McWeeney SK. Evidence-Based Precision Oncology with the Cancer Targetome. Trends Pharmacol Sci. 2017 Dec;38(12):1085-1099. doi: 10.1016/j.tips.2017.08.006. Epub 2017 Sep 27. PMID: 28964549; PMCID: PMC5759325.
#Bottomly D, Long N, Schultz AR, Kurtz SE, Tognon CE, Johnson K, Abel M, Agarwal A, Avaylon S, Benton E, Blucher A, Borate U, Braun TP, Brown J, Bryant J, Burke R, Carlos A, Chang BH, Cho HJ, Christy S, Coblentz C, Cohen AM, d'Almeida A, Cook R, Danilov A, Dao KT, Degnin M, Dibb J, Eide CA, English I, Hagler S, Harrelson H, Henson R, Ho H, Joshi SK, Junio B, Kaempf A, Kosaka Y, Laderas T, Lawhead M, Lee H, Leonard JT, Lin C, Lind EF, Liu SQ, Lo P, Loriaux MM, Luty S, Maxson JE, Macey T, Martinez J, Minnier J, Monteblanco A, Mori M, Morrow Q, Nelson D, Ramsdill J, Rofelty A, Rogers A, Romine KA, Ryabinin P, Saultz JN, Sampson DA, Savage SL, Schuff R, Searles R, Smith RL, Spurgeon SE, Sweeney T, Swords RT, Thapa A, Thiel-Klare K, Traer E, Wagner J, Wilmot B, Wolf J, Wu G, Yates A, Zhang H, Cogle CR, Collins RH, Deininger MW, Hourigan CS, Jordan CT, Lin TL, Martinez ME, Pallapati RR, Pollyea DA, Pomicter AD, Watts JM, Weir SJ, Druker BJ, McWeeney SK, Tyner JW. Integrative analysis of drug response and clinical outcome in acute myeloid leukemia. Cancer Cell. 2022 Aug 8;40(8):850-864.e9. doi: 10.1016/j.ccell.2022.07.002. Epub 2022 Jul 21. PMID: 35868306; PMCID: PMC9378589.
TARGETOME=https://github.com/biodev/beataml2.0_data/raw/main/beataml_drug_families.xlsx
CTD_CHEM_DIS=https://ctdbase.org/reports/CTD_chemicals_diseases.csv.gz
CTD_PROT_DIS=https://ctdbase.org/reports/CTD_curated_genes_diseases.csv.gz
CTD_PROT_PATH=https://ctdbase.org/reports/CTD_genes_pathways.csv.gz
BEATAML_INHIB_RESP=https://github.com/biodev/beataml2.0_data/raw/main/beataml_probit_curve_fits_v4_dbgap.txt
BEATAML_MUT=https://github.com/biodev/beataml2.0_data/raw/main/beataml_wes_wv1to4_mutations_dbgap.txt
BEATAML_EXPR=https://github.com/biodev/beataml2.0_data/raw/main/beataml_waves1to4_norm_exp_dbgap.txt
BEATAML_EXPR_COUNTS=https://github.com/biodev/beataml2.0_data/raw/main/beataml_waves1to4_counts_dbgap.txt
AA_SEQ=https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.fasta.gz

# Create the ROOT directory if it doesn't exist
mkdir -p "$ROOT"

# Function to download and unzip files if necessary
download_and_unzip() {
    local url=$1
    local output_path=$2
    local unzip_flag=$3

    # Determine the final file path after unzipping if needed
    local final_output_path="$output_path"
    if [ "$unzip_flag" = true ]; then
        final_output_path="${output_path%.gz}"
    fi

    # Check for the final file (unzipped if applicable)
    if [ -f "$final_output_path" ]; then
        echo "File $final_output_path already exists. Skipping download."
        return
    fi

    # If the final file doesn't exist, proceed to download
    wget -O "$output_path" "$url"
    if [ "$unzip_flag" = true ]; then
        gunzip -f "$output_path"
    fi
}

# Download the files
download_and_unzip $TARGETOME ${ROOT}beataml_drug_families.xlsx false
download_and_unzip $CTD_CHEM_DIS ${ROOT}CTD_chemicals_diseases.csv.gz true
download_and_unzip $CTD_PROT_DIS ${ROOT}CTD_curated_genes_diseases.csv.gz true
download_and_unzip $CTD_PROT_PATH ${ROOT}CTD_genes_pathways.csv.gz true
download_and_unzip $BEATAML_INHIB_RESP ${ROOT}beataml_probit_curve_fits_v4_dbgap.txt false
download_and_unzip $BEATAML_MUT ${ROOT}beataml_wes_wv1to4_mutations_dbgap.txt false
download_and_unzip $BEATAML_EXPR ${ROOT}beataml_waves1to4_norm_exp_dbgap.txt false
download_and_unzip $BEATAML_EXPR_COUNTS ${ROOT}beataml_waves1to4_counts_dbgap.txt false
download_and_unzip $AA_SEQ ${ROOT}UP000005640_9606.fasta.gz true
