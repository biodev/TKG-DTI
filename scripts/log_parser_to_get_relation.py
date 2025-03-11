#!/usr/bin/env python3

import argparse
import os
import re
import glob

def parse_remove_relation(log_file):
    """
    Parses a single log file, looking for:
      - 'remove_relation_idx=...' in the Namespace line
      - 'WARNING: removing relation: ...'
    Returns (idx, relation_str) if found, otherwise None.
    """
    remove_idx = None
    relation_str = None

    with open(log_file, 'r') as f:
        for line in f:
            # Look for remove_relation_idx in the Namespace line
            if 'Namespace(' in line and 'remove_relation_idx=' in line:
                match = re.search(r'remove_relation_idx=(\d+)', line)
                if match:
                    remove_idx = match.group(1)

            # Look for the WARNING line
            if 'WARNING: removing relation:' in line:
                # Example line:
                # "WARNING: removing relation: ('dbgap_subject', 'mut_missense_variant_deleterious_fwd', 'gene')"
                rel_match = re.search(r"WARNING: removing relation:\s*(.*)", line)
                if rel_match:
                    relation_str = rel_match.group(1).strip()
    
    if remove_idx is not None and relation_str is not None:
        return (remove_idx, relation_str)
    else:
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Parse log files to extract remove_relation_idx and the corresponding relation."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing the log.*.out files (searched recursively)."
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default=None,
        help="If provided, write the unique mappings to this file instead of printing to console."
    )
    
    args = parser.parse_args()
    root_dir = args.root

    # Dictionary: idx -> set of relations
    idx_to_relations = {}

    # Recursively look for log files named 'log.*.out'
    pattern = os.path.join(root_dir, '**', 'log.*.out')
    for logfile in glob.iglob(pattern, recursive=True):
        parsed = parse_remove_relation(logfile)
        if parsed:
            idx, relation = parsed
            idx_to_relations.setdefault(idx, set()).add(relation)

    # Prepare lines for output
    output_lines = []
    for idx in sorted(idx_to_relations.keys(), key=lambda x: int(x)):
        # Each idx may have multiple relations
        for rel in sorted(idx_to_relations[idx]):
            output_lines.append(f"{idx} -> {rel}")

    # Either print to console or write to file
    if args.outfile:
        with open(args.outfile, 'w') as f:
            for line in output_lines:
                f.write(line + '\n')
    else:
        for line in output_lines:
            print(line)

if __name__ == "__main__":
    main()
