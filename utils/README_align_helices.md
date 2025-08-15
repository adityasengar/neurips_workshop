  GNU nano 5.6.1                                                                                                                            README.md                                                                                                                                      
# Helix Alignment Script

This script, `align_helices.py`, is designed to align helix sequences (e.g., TM1-TM7, H8) from a source file (either a `d2_res.txt` or an Excel `.xlsx` file) with a full protein sequence extracted from a PDB file. It uses Biopython's pairwise alignment functionality to find the bes>

## Features

-   Parses helix sequences and BW (Ballesteros-Weinstein) notations from `d2_res.txt` or `.xlsx` files.
-   Extracts the full amino acid sequence and PDB residue numbers from a `.pdb` file.
-   Performs local sequence alignment between each helix and the full PDB sequence.
-   Outputs the aligned PDB residue ranges for each helix.
-   Can optionally find corresponding PDB residue IDs for specific BW notations.

## Requirements

-   Biopython (`pip install biopython`)
-   openpyxl (`pip install openpyxl`) (only if using .xlsx source files)

## Usage

```bash
python align_helices.py [OPTIONS]
```

### Arguments

-   `--source_file` (str, default: `d2_res.txt`):
    Path to the source file containing helix sequences and BW notations.
    Supported formats: `.txt` (assumed to be `d2_res.txt` format) or `.xlsx`.

-   `--pdb_file` (str, default: `heavy_chain.pdb`):
    Path to the PDB file containing the full protein sequence.

-   `--bw_notations` (nargs='*', default: `[]`):
    A list of specific BW notations (e.g., `'3.50'`, `'3.51'`) for which to find corresponding aligned PDB residue IDs.

-   `--xlsx_file` (str, default: `None`):
    Path to an *additional* XLSX file containing BW notations to find. This is separate from `--source_file`.

### Example

To align helices from `residue_table_a1r.xlsx` with the sequence in `A1R_1000ns_cMD_noP2_chainA.pdb`:

```bash
python align_helices.py --source_file residue_table_a1r.xlsx --pdb_file A1R_1000ns_cMD_noP2_chainA.pdb
```

## Output Example

When run, the script will print information similar to the following for each helix:

```
PDB Sequence length: 293
------------------------------
Aligning TM1 (length 36):
Sequence: SAFQAAYIGIEVLIALVSVPGNVLVIWAVKVNQALR
2 SAFQAAYIGIEVLIALVSVPGNVLVIWAVKVNQALR
  ||||||||||||||||||||||||||||||||||||
1 SAFQAAYIGIEVLIALVSVPGNVLVIWAVKVNQALR
  Score=72

Aligned PDB Residue Range: 6-41
------------------------------
... (similar output for other helices) ...

--- BW Notation to Aligned PDB Residue ID Mapping ---
BW Notation 3.50 (from TM3): Aligned PDB Residue ID 80
-----------------------------------------------------
```

The output shows the alignment of each helix sequence against the PDB sequence, including the alignment score and the calculated PDB residue range for the aligned helix. If `bw_notations` or `xlsx_file` arguments are provided, it will also attempt to map those BW notations to align>


