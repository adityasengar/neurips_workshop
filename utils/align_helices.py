#!/usr/bin/env python

import re

try:
    from Bio import pairwise2
    from Bio.pairwise2 import format_alignment
except ImportError:
    print("Error: Biopython is not installed.")
    print("Please install it using: pip install biopython")
    exit()

try:
    import openpyxl
except ImportError:
    print("Error: openpyxl is not installed.")
    print("Please install it using: pip install openpyxl")
    exit()

# 3-letter to 1-letter amino acid code mapping
aa_codes = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def parse_d2_res(filename="d2_res.txt"):
    """
    Parses the d2_res.txt file to extract helix sequences, BW notation to residue ID mappings,
    and the ordered list of BW notations for each helix.
    """
    helices = {}
    bw_to_resid = {}
    helix_bw_notations = {} # New: To store ordered BW notations per helix
    current_helix = None
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                current_helix = None # Reset on empty lines
                continue

            # Helix names are TMx or H8
            if line.startswith("TM") or line.startswith("H8"):
                current_helix = line.split()[0]
                helices[current_helix] = ""
                helix_bw_notations[current_helix] = [] # Initialize list for this helix
            elif current_helix:
                # Residue lines have at least 3 parts: GPCRdb(A), BW, ResidueInfo
                parts = line.split()
                if len(parts) >= 3 and re.match(r'(\d+x\d+|\d+)', parts[0]):
                    bw_notation = parts[1]
                    res_info = parts[-1] # This should be the last part for residue info
                    # Extract residue ID (e.g., '31' from 'R31')
                    res_id_match = re.search(r'\d+', res_info)
                    if res_id_match:
                        residue_id = res_id_match.group(0)
                        bw_to_resid[bw_notation] = residue_id
                    
                    # Add BW notation to the current helix's list
                    helix_bw_notations[current_helix].append(bw_notation)

                    # For helix sequence, take the first letter of the last column
                    helices[current_helix] += res_info[0]

    return helices, bw_to_resid, helix_bw_notations

def parse_pdb(filename="heavy_chain.pdb"):
    """
    Parses a PDB file to extract the amino acid sequence and a mapping of sequence index to PDB residue number.
    """
    sequence = ""
    pdb_res_numbers = [] # To store PDB residue numbers corresponding to sequence indices
    last_res_num = None
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                res_num = int(line[22:26])
                if res_num != last_res_num:
                    res_name = line[17:20].strip()
                    if res_name in aa_codes:
                        sequence += aa_codes[res_name]
                        pdb_res_numbers.append(res_num)
                    last_res_num = res_num
    return sequence, pdb_res_numbers

import argparse

def parse_xlsx_bw_notations(filename):
    """
    Parses an XLSX file to extract BW notations.
    Assumes BW notations are in the second column (index 1) and skips the first row (header).
    """
    bw_notations = []
    try:
        workbook = openpyxl.load_workbook(filename)
        sheet = workbook.active # Get the active sheet
        # Start from the second row (min_row=2) to skip header, and read up to the second column (max_col=2)
        for row in sheet.iter_rows(min_row=2, max_col=2, values_only=True):
            if len(row) > 1 and row[1] is not None: # Check if the second column exists and is not None
                bw_notation = str(row[1]).strip()
                # Validate if it looks like a BW notation (e.g., "X.YY")
                if re.match(r'\d+\.\d+', bw_notation):
                    bw_notations.append(bw_notation)
    except Exception as e:
        print(f"Error parsing XLSX file {filename}: {e}")
    return bw_notations

def main():
    """
    Main function to perform sequence alignment.
    """
    parser = argparse.ArgumentParser(description="Align helix sequences from d2_res.txt with a PDB file's sequence.")
    parser.add_argument("--d2_res_file", type=str, default="d2_res.txt",
                        help="Path to the d2_res.txt file containing helix sequences.")
    parser.add_argument("--pdb_file", type=str, default="heavy_chain.pdb",
                        help="Path to the PDB file containing the full protein sequence.")
    parser.add_argument("--bw_notations", nargs='*', default=[],
                        help="List of BW notations (e.g., '3.50 3.51') to find corresponding residue IDs.")
    parser.add_argument("--xlsx_file", type=str, default=None,
                        help="Path to an XLSX file containing BW notations in the first column.")
    args = parser.parse_args()

    d2_res_file = args.d2_res_file
    pdb_file = args.pdb_file
    
    bw_notations_to_find = []

    if args.bw_notations:
        bw_notations_to_find.extend(args.bw_notations)

    if args.xlsx_file:
        xlsx_bw_notations = parse_xlsx_bw_notations(args.xlsx_file)
        bw_notations_to_find.extend(xlsx_bw_notations)

    # 1. Parse the helix sequences, BW to residue ID mapping, and helix BW notations from d2_res.txt
    helix_sequences, bw_to_direct_resid_map, helix_bw_notations = parse_d2_res(d2_res_file)
    if not helix_sequences and not bw_to_direct_resid_map:
        print(f"Could not parse any helix sequences or BW notations from {d2_res_file}")
        return

    # 2. Parse the full sequence and PDB residue numbers from heavy_chain.pdb
    pdb_sequence, pdb_res_numbers = parse_pdb(pdb_file)
    if not pdb_sequence:
        print(f"Could not parse sequence from {pdb_file}")
        return

    print(f"PDB Sequence length: {len(pdb_sequence)}")
    print("-" * 30)

    aligned_helix_pdb_ranges = {} # To store the aligned PDB residue ranges for each helix

    # 3. Align each helix sequence with the PDB sequence
    for helix_name, helix_seq in helix_sequences.items():
        print(f"Aligning {helix_name} (length {len(helix_seq)}):")
        print(f"Sequence: {helix_seq}")

        # Using local alignment to find the best match for the helix in the larger sequence
        alignments = pairwise2.align.localms(pdb_sequence, helix_seq, 2, -1, -0.5, -0.1)

        if alignments:
            best_alignment = alignments[0]
            print(format_alignment(*best_alignment))

            pdb_start_idx = best_alignment[3]
            pdb_end_idx = best_alignment[4]

            if pdb_res_numbers and len(pdb_res_numbers) > pdb_end_idx:
                aligned_pdb_start_res = pdb_res_numbers[pdb_start_idx]
                aligned_pdb_end_res = pdb_res_numbers[pdb_end_idx - 1] # end_idx is exclusive
                aligned_helix_pdb_ranges[helix_name] = (aligned_pdb_start_res, aligned_pdb_end_res)
                print(f"Aligned PDB Residue Range: {aligned_pdb_start_res}-{aligned_pdb_end_res}")
            else:
                print("Could not determine PDB residue range for alignment.")
        else:
            print("No alignment found.")
        print("-" * 30)

    # Process BW notations if provided, using aligned PDB residue ranges
    if bw_notations_to_find:
        print("\n--- BW Notation to Aligned PDB Residue ID Mapping ---")
        for bw_notation in bw_notations_to_find:
            found_in_helix = False
            for helix_name, bw_list in helix_bw_notations.items():
                if bw_notation in bw_list:
                    found_in_helix = True
                    if helix_name in aligned_helix_pdb_ranges:
                        offset_in_helix = bw_list.index(bw_notation)
                        aligned_start_res = aligned_helix_pdb_ranges[helix_name][0]
                        calculated_pdb_resid = aligned_start_res + offset_in_helix
                        print(f"BW Notation {bw_notation} (from {helix_name}): Aligned PDB Residue ID {calculated_pdb_resid}")
                    else:
                        print(f"BW Notation {bw_notation} (from {helix_name}): Helix not aligned, cannot determine aligned PDB Residue ID.")
                    break
            if not found_in_helix:
                print(f"BW Notation {bw_notation}: Not found in {d2_res_file}")
        print("-----------------------------------------------------")

if __name__ == "__main__":
    main()







