import os
import re
import argparse
import numpy as np
import pandas as pd
import torch

from tkgdti.data.GraphBuilder import GraphBuilder
from tkgdti.embed.AA2EMB import AA2EMB


def get_args():
    parser = argparse.ArgumentParser(description='embed amino acid sequences for human genes using ProtBert')
    parser.add_argument('--data', type=str, default='../../../data/', help='Path to the input data dir')
    parser.add_argument('--extdata', type=str, default='../../extdata/', help='Path to the extra data dir')
    parser.add_argument('--out', type=str, default='../../output/', help='Path to the output data dir')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--model_name', type=str, default='Rostlab/prot_bert', help='Model name for ProtBert')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for embedding')
    parser.add_argument('--repr', type=str, default='mean', help='Representation type for embedding')
    parser.add_argument('--max_len', type=int, default=2048, help='Maximum length of amino acid sequence')
    return parser.parse_args()


def parse_fasta_to_dataframe(path):
    records = []
    current_record = None

    organism_pattern = re.compile(r'OS=(.+?)\s+OX=')
    gene_pattern = re.compile(r'GN=(\S+)')

    with open(path, 'r') as fasta_file:
        for line in fasta_file:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_record:
                    records.append(current_record)
                header_line = line[1:].strip()
                current_record = {'id': header_line.split()[0], 'organism': None, 'gene_name': None, 'sequence': []}
                organism_match = organism_pattern.search(header_line)
                if organism_match:
                    current_record['organism'] = organism_match.group(1)
                gene_match = gene_pattern.search(header_line)
                if gene_match:
                    current_record['gene_name'] = gene_match.group(1)
            else:
                if current_record is not None:
                    current_record['sequence'].append(line)
        if current_record:
            records.append(current_record)

    for record in records:
        record['sequence'] = ''.join(record['sequence'])

    df = pd.DataFrame(records, columns=['id', 'organism', 'gene_name', 'sequence'])
    return df


if __name__ == '__main__':
    print('------------------------------------------------------------------')
    print('kg_construction_08a__embed_amino_acids.py')
    print('------------------------------------------------------------------')
    print()

    args = get_args()
    print('-------------------------------------------------------------------')
    print(args)
    print('-------------------------------------------------------------------')
    print()

    os.makedirs(f'{args.out}/meta', exist_ok=True)

    # seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build genespace from existing relations
    root_rel = f'{args.out}/relations'
    relnames = os.listdir(root_rel) if os.path.exists(root_rel) else []
    GB = GraphBuilder(root=root_rel, relnames=relnames, val_idxs=None, test_idxs=None)
    print('building...')
    GB.build()

    genespace = np.unique(
        GB.relations[lambda x: x.src_type == 'gene'].src.values.tolist() +
        GB.relations[lambda x: x.dst_type == 'gene'].dst.values.tolist()
    )

    fasta_path = os.path.join(args.data, 'UP000005640_9606.fasta')
    gene2aa = parse_fasta_to_dataframe(fasta_path)
    gene2aa = gene2aa[lambda x: x.gene_name.isin(genespace)]
    gene2aa = gene2aa.groupby('gene_name').first().reset_index()
    gene2aa.to_csv(f'{args.out}/meta/gene2aa.csv', index=False)

    print(f'filtered gene2aa rows: {gene2aa.shape[0]}')

    # Embed amino acid sequences
    aas = gene2aa.sequence.values
    print('embedding amino acid sequences...')
    embedder = AA2EMB(model_name=args.model_name, 
                      batch_size=args.batch_size, 
                      max_len=args.max_len,
                      repr=args.repr)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    outputs = embedder.embed(aas, device=device).cpu().numpy()
    print(f'embeddings shape: {outputs.shape}')

    aas_dict = {'amino_acids': aas, 'embeddings': outputs, 'meta_df': gene2aa}
    torch.save(aas_dict, f'{args.out}/meta/aas_dict.pt')

    print('summary:')
    print('-' * 100)
    print(f'# genes embedded: {gene2aa.shape[0]}')
    print(f'embedding shape: {outputs.shape}')
    print('-' * 100)

    print(f'saved to: {args.out}/meta')
    print()


