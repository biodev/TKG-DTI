#!/usr/bin/env python3

import re
import glob

def parse_remove_relation(log_file):
    """
    Parses a single log file, looking for:
      - 'remove_relation_idx=...' in the Namespace line
      - 'WARNING: removing relation: ...' in subsequent lines
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
                # e.g. line: "WARNING: removing relation: ('dbgap_subject', 'mut_missense_variant_deleterious_fwd', 'gene')"
                rel_match = re.search(r"WARNING: removing relation:\s*(.*)", line)
                if rel_match:
                    # The capturing group should contain something like
                    # "('dbgap_subject', 'mut_missense_variant_deleterious_fwd', 'gene')"
                    relation_str = rel_match.group(1).strip()
    
    # Only return a valid tuple if both pieces of information were found
    if remove_idx is not None and relation_str is not None:
        return (remove_idx, relation_str)
    else:
        return None


def main():
    # Dictionary to store idx -> set of relations (in case the same idx appears with multiple relations)
    idx_to_relations = {}

    # Adjust the glob pattern (e.g. "logs/*.out") or directory path as needed
    for logfile in glob.glob("log.*.out"):
        parsed = parse_remove_relation(logfile)
        if parsed:
            idx, relation = parsed
            idx_to_relations.setdefault(idx, set()).add(relation)
    
    # Print results
    # If you want to write to a file, just open and write instead of printing
    print("Mapping of remove_relation_idx -> relation(s):")
    for idx in sorted(idx_to_relations.keys(), key=lambda x: int(x)):
        for rel in idx_to_relations[idx]:
            print(f"{idx} -> {rel}")


if __name__ == "__main__":
    main()
