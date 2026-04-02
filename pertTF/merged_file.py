import pandas as pd
import argparse
import sys
import os

def main():
    # 1. Parse command line arguments
    parser = argparse.ArgumentParser(description="Merge two CSV files based on 'cell_id'.")
    parser.add_argument("csv_file_1", help="Path to the first CSV file")
    parser.add_argument("csv_file_2", help="Path to the second CSV file")
    
    args = parser.parse_args()

    # 2. Validation: Check if files exist
    if not os.path.exists(args.csv_file_1):
        sys.exit(f"Error: File '{args.csv_file_1}' not found.")
    if not os.path.exists(args.csv_file_2):
        sys.exit(f"Error: File '{args.csv_file_2}' not found.")

    # 3. Read Data
    try:
        df1 = pd.read_csv(args.csv_file_1)
        df2 = pd.read_csv(args.csv_file_2)
    except Exception as e:
        sys.exit(f"Error reading CSV files: {e}")

    # 4. Validation: Check for 'cell_id' column
    if 'cell_id' not in df1.columns:
        sys.exit(f"Error: 'cell_id' column missing from {args.csv_file_1}")
    if 'cell_id' not in df2.columns:
        sys.exit(f"Error: 'cell_id' column missing from {args.csv_file_2}")

    print(f"File 1 rows: {len(df1)}")
    print(f"File 2 rows: {len(df2)}")

    # 5. Merge Data
    # Defaults to 'inner' join (intersection). Change to how='left' or 'outer' if needed.
    merged_df = pd.merge(df1, df2, on='cell_id', how='inner')

    print(f"Merged rows: {len(merged_df)}")

    # 6. Save Output
    output_filename = '/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/CV_imporved_code/fold_1/best_epoch_e36/tail_end_gate/metadata_celltype_results_boxed_region.csv'
    merged_df.to_csv(output_filename, index=False)
    print(f"Successfully saved merged data to {output_filename}")

if __name__ == "__main__":
    main()
