# `align_helices.py` - Protein Helix Alignment and BW Notation Mapping

This script is designed to align protein helix sequences (typically from a `d2_res.txt` file) with a full protein sequence (from a PDB file) and to map Ballesteros-Weinstein (BW) notations to their corresponding PDB residue IDs based on these alignments.

## Features

*   **Helix Sequence Extraction**: Parses helix sequences (e.g., TM1, TM2, H8) from a `d2_res.txt` file.
*   **PDB Sequence Extraction**: Extracts the full amino acid sequence from a given PDB file.
*   **Sequence Alignment**: Uses Biopython's local alignment to find the best match for each helix sequence within the full PDB sequence, reporting the aligned PDB residue range.
*   **BW Notation Mapping**:
    *   Directly maps BW notations (e.g., `3.50`, `6.30`) to their corresponding residue IDs as defined in `d2_res.txt`.
    *   **Crucially**, it can also calculate the PDB residue ID for a given BW notation *relative to the aligned PDB residue range* of the helix it belongs to. This provides a more accurate mapping in the context of the aligned structure.
*   **Flexible Input**: Accepts BW notations directly via command-line arguments or by parsing an Excel (`.xlsx`) file.

## Dependencies

This script requires the following Python libraries:

*   `biopython`
*   `openpyxl`

You can install them using pip:

```bash
pip install biopython openpyxl
```

## Usage

```bash
python align_helices.py [OPTIONS]
```

### Arguments

*   `--d2_res_file <path/to/d2_res.txt>`: Path to the `d2_res.txt` file containing helix sequences and BW notations.
    *   Default: `d2_res.txt`
*   `--pdb_file <path/to/heavy_chain.pdb>`: Path to the PDB file containing the full protein sequence.
    *   Default: `heavy_chain.pdb`
*   `--bw_notations <BW_NOTATION_1> <BW_NOTATION_2> ...`: A space-separated list of BW notations for which to find the aligned PDB residue IDs.
*   `--xlsx_file <path/to/residue_table.xlsx>`: Path to an Excel (`.xlsx`) file containing BW notations. The script assumes BW notations are in the **second column** (column B) and skips the first row (header).

### Examples

1.  **Run with default files and show helix alignments:**
    ```bash
    python align_helices.py
    ```

2.  **Specify custom `d2_res.txt` and PDB files:**
    ```bash
    python align_helices.py --d2_res_file my_d2_data.txt --pdb_file my_protein.pdb
    ```

3.  **Find aligned PDB residue IDs for specific BW notations:**
    ```bash
    python align_helices.py --bw_notations 3.50 6.30 7.53
    ```
    This will output:
    ```
    --- BW Notation to Aligned PDB Residue ID Mapping ---
    BW Notation 3.50 (from TM3): Aligned PDB Residue ID 101
    BW Notation 6.30 (from TM6): Aligned PDB Residue ID 203
    BW Notation 7.53 (from TM7): Aligned PDB Residue ID 261
    -----------------------------------------------------
    ```

4.  **Read BW notations from an Excel file:**
    ```bash
    python align_helices.py --xlsx_file residue_table.xlsx
    ```

5.  **Combine explicit BW notations with those from an Excel file:**
    ```bash
    python align_helices.py --bw_notations 1.50 --xlsx_file residue_table.xlsx
    ```

## Input File Formats

### `d2_res.txt`

A plain text file where each helix section starts with a line like `TMx` or `H8`. Subsequent lines for that helix contain residue information, typically with the BW notation in the second column and the residue (e.g., `R31`) in the last column.

Example snippet:
```
GPCRdb(A)       BW      D2 receptor Human
TM1             
1x29    1.29    R31
1x30    1.30    P32
...
TM3             
3x49    3.49    D131
3x50    3.50    R132
3x51    3.51    Y133
...
```

### `residue_table.xlsx`

An Excel file where BW notations are expected to be in the **second column** (column B). The first row is assumed to be a header and will be skipped.

Example (assuming column B contains the BW notations):

| Column A | Column B | Column C |
| :------- | :------- | :------- |
| Header1  | BW_Number | Header3  |
| Data1    | 1.29     | Data3    |
| Data4    | 3.50     | Data6    |
| ...	   | ...      | ...	 |


