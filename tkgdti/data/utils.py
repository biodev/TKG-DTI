import requests 
from Bio import Entrez, SeqIO
from io import StringIO
from urllib.parse import urlparse, parse_qs, urlencode
from requests.adapters import HTTPAdapter, Retry
import re
import time
import requests
import csv
import pandas as pd 

def get_protein_sequence_uniprot(gene_symbol, email=''):
    """Retrieve the canonical protein sequence for a given gene symbol from UniProt."""

    # Set your email (NCBI requires it for identification)
    Entrez.email = email
    try:
        # Construct the query with quotes around the gene symbol, and include only reviewed entries
        query = f'gene_exact:"{gene_symbol}" AND organism_id:9606 AND reviewed:true'
        params = {
            'query': query,
            'format': 'fasta',
            'include': 'no'  # Do not include isoforms
        }
        url = 'https://rest.uniprot.org/uniprotkb/search'
        response = requests.get(url, params=params)

        if response.status_code == 200:
            sequences = response.text.strip()
            if sequences:
                # Parse the FASTA sequences
                fasta_io = StringIO(sequences)
                seq_records = list(SeqIO.parse(fasta_io, 'fasta'))

                # Extract the sequence from the first (and should be only) record
                if seq_records:
                    # Get the sequence and the UniProt ID
                    first_record = seq_records[0]
                    sequence = str(first_record.seq)
                    uniprot_id = first_record.id.split('|')[1]  # Extract UniProt ID
                    return sequence, uniprot_id
                else:
                    print(f"No sequences found for gene symbol: {gene_symbol}")
                    return None, None
            else:
                print(f"No sequences found for gene symbol: {gene_symbol}")
                return None, None
        else:
            print(f"Failed to retrieve data for gene symbol {gene_symbol}. HTTP Status Code: {response.status_code}")
            return None, None
    except Exception as e:
        print(f"An error occurred for gene symbol {gene_symbol}: {e}")
        return None, None
    

def get_smiles_inchikey(drug_name):
    """Retrieve the Canonical and Isomeric SMILES and InChIKey for a given drug name from PubChem."""
    properties = 'CanonicalSMILES,IsomericSMILES,InChIKey'
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/{properties}/JSON"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        try:
            props = data['PropertyTable']['Properties'][0]
            canonical_smiles = props.get('CanonicalSMILES', None)
            isomeric_smiles = props.get('IsomericSMILES', None)
            inchikey = props.get('InChIKey', None)
            return canonical_smiles, isomeric_smiles, inchikey
        except (KeyError, IndexError):
            return None, None, None
    else:
        return None, None, None
    


# Configure retries for the session
retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))

API_URL = "https://rest.uniprot.org"
POLLING_INTERVAL = 3  # in seconds

def check_response(response):
    """Check the HTTP response for errors."""
    try:
        response.raise_for_status()
    except requests.HTTPError:
        print(f"Error: {response.text}")
        raise

def submit_id_mapping(from_db, to_db, ids):
    """Submit an ID mapping job."""
    response = session.post(
        f"{API_URL}/idmapping/run",
        data={"from": from_db, "to": to_db, "ids": ",".join(ids)},
    )
    check_response(response)
    return response.json()["jobId"]

def check_id_mapping_results_ready(job_id):
    """Check if the ID mapping job is ready."""
    while True:
        response = session.get(f"{API_URL}/idmapping/status/{job_id}")
        check_response(response)
        status = response.json()
        if "jobStatus" in status:
            if status["jobStatus"] in ("RUNNING", "NEW"):
                print(f"Job status: {status['jobStatus']}. Retrying in {POLLING_INTERVAL}s...")
                time.sleep(POLLING_INTERVAL)
            else:
                raise Exception(f"Job failed with status: {status['jobStatus']}")
        else:
            return True

def get_id_mapping_results_link(job_id):
    """Retrieve the redirect URL for the results."""
    response = session.get(f"{API_URL}/idmapping/details/{job_id}")
    check_response(response)
    return response.json()["redirectURL"]

def get_next_link(headers):
    """Extract the 'next' link from the headers for pagination."""
    if "Link" in headers:
        match = re.search(r'<(.+)>; rel="next"', headers["Link"])
        if match:
            return match.group(1)
    return None

def get_batch(response, file_format):
    """Retrieve batches of results if paginated."""
    next_link = get_next_link(response.headers)
    while next_link:
        response = session.get(next_link)
        check_response(response)
        yield decode_results(response, file_format)
        next_link = get_next_link(response.headers)

def decode_results(response, file_format):
    """Decode the response content based on the file format."""
    if file_format == "tsv":
        return [line for line in response.text.strip().split("\n") if line]
    elif file_format == "json":
        return response.json()
    else:
        return response.text

def combine_batches(all_results, batch_results, file_format):
    """Combine multiple batches of results."""
    if file_format == "tsv":
        return all_results + batch_results[1:]  # Skip header in subsequent batches
    elif file_format == "json":
        all_results["results"] += batch_results.get("results", [])
        all_results["failedIds"] += batch_results.get("failedIds", [])
        return all_results
    else:
        return all_results + batch_results

def get_id_mapping_results_search(url):
    """Retrieve the ID mapping results."""
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    file_format = query_params.get("format", ["json"])[0]
    size = int(query_params.get("size", [500])[0])
    query_params["size"] = size
    parsed_url = parsed_url._replace(query=urlencode(query_params, doseq=True))
    url = parsed_url.geturl()

    response = session.get(url)
    check_response(response)
    results = decode_results(response, file_format)
    total = int(response.headers.get("x-total-results", 0))
    print(f"Fetched: {min(size, total)} / {total}")

    for batch_results in get_batch(response, file_format):
        results = combine_batches(results, batch_results, file_format)
        fetched = len(results) if isinstance(results, list) else len(results.get("results", []))
        print(f"Fetched: {min(fetched, total)} / {total}")
    return results

def get_data_frame_from_tsv_results(tsv_results):
    """Convert TSV results to a Pandas DataFrame."""
    reader = csv.DictReader(tsv_results, delimiter="\t")
    return pd.DataFrame(reader)

def uniprot_ids_to_gene_symbols(uniprot_ids):
    """
    Convert a list of UniProt IDs to gene symbols using the UniProt ID mapping API.
    
    Parameters:
        uniprot_ids (list): List of UniProt IDs.
    
    Returns:
        pandas.DataFrame: DataFrame containing UniProt IDs and corresponding gene symbols.
    """
    # Submit the ID mapping job
    job_id = submit_id_mapping(
        from_db="UniProtKB_AC-ID", to_db="UniProtKB", ids=uniprot_ids
    )
    # Wait until the job is ready
    if check_id_mapping_results_ready(job_id):
        # Get the results URL with desired fields
        results_url = get_id_mapping_results_link(job_id)
        # Add desired fields and format to the URL
        parsed_url = urlparse(results_url)
        query_params = parse_qs(parsed_url.query)
        query_params["fields"] = ["accession,gene_names"]
        query_params["format"] = ["tsv"]
        parsed_url = parsed_url._replace(query=urlencode(query_params, doseq=True))
        results_url = parsed_url.geturl()
        # Fetch the results
        tsv_results = get_id_mapping_results_search(results_url)
        # Convert TSV results to DataFrame
        df = get_data_frame_from_tsv_results(tsv_results)
        return df
    else:
        print("Job did not finish successfully.")
        return None
