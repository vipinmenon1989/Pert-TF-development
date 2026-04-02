import anndata as ad
import pandas as pd
import argparse
import os
import sys

def extract_cell_metadata(input_file, column_name, output_file, sep=',', filter_pattern=None):
    """
    Loads an .h5ad file, extracts the index (Cell ID) and a specific 
    observation column (Cell Type), and saves to a text file.
    Optionally filters rows where the column value matches a pattern.
    """
    
    # 1. Validation
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    print(f"Loading {input_file}...")
    
    try:
        # Load the AnnData object
        adata = ad.read_h5ad(input_file)
    except Exception as e:
        print(f"Error reading .h5ad file: {e}")
        sys.exit(1)

    # 2. Check for column existence
    if column_name not in adata.obs.columns:
        print(f"Error: Column '{column_name}' not found in adata.obs.")
        print("Available columns:", list(adata.obs.columns))
        sys.exit(1)

    # 3. Extraction
    # adata.obs is a pandas DataFrame. 
    # The index usually contains the Cell IDs (barcodes).
    print(f"Extracting cell IDs and '{column_name}'...")
    
    # Create a new dataframe with just the index and the target column
    # We reset_index() to make the Cell ID a proper column for export
    df_export = adata.obs[[column_name]].reset_index()
    
    # Rename the index column to 'cell_id' for clarity in the output
    df_export.columns = ['cell_id', column_name]

    # 4. Optional Filtering (Applied to the Content Column)
    if filter_pattern:
        print(f"Filtering rows where '{column_name}' contains pattern: '{filter_pattern}'")
        initial_count = len(df_export)
        
        # Filter rows where the METADATA COLUMN (not ID) contains the pattern
        # We convert to string first to handle categorical or object types safely
        df_export = df_export[df_export[column_name].astype(str).str.contains(filter_pattern, na=False)]
        
        final_count = len(df_export)
        print(f"Kept {final_count} out of {initial_count} cells.")

    # 5. Save to file
    try:
        df_export.to_csv(output_file, index=False, sep=sep)
        print(f"Success. Data saved to '{output_file}'.")
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Argument parsing for CLI usage
    parser = argparse.ArgumentParser(
        description="Extract Cell ID and a specific metadata column from an .h5ad file."
    )
    
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        help="Path to input .h5ad file."
    )
    parser.add_argument(
        "-c", "--column", 
        required=True, 
        help="The column name in adata.obs containing cell types (e.g., 'leiden', 'cell_type', 'louvain')."
    )
    parser.add_argument(
        "-o", "--output", 
        required=True, 
        help="Path to output file (e.g., output.csv or output.txt)."
    )
    parser.add_argument(
        "--delimiter", 
        default=",", 
        help="Delimiter for the output file. Default is comma (CSV). Use '\\t' for tab-separated."
    )
    parser.add_argument(
        "--filter", 
        help="Optional: Only save rows where the selected COLUMN contains this string (e.g., 'ESC_DE')."
    )

    args = parser.parse_args()

    # handle escaped tab characters from command line
    delimiter = args.delimiter.replace('\\t', '\t')

    extract_cell_metadata(args.input, args.column, args.output, delimiter, args.filter)
