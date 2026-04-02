import pandas as pd
import argparse
import sys
import os

def merge_and_keep_common(file1_path, file2_path, join_key, output_path):
    # 1. Validation
    if not os.path.exists(file1_path):
        sys.exit(f"Error: File not found: {file1_path}")
    if not os.path.exists(file2_path):
        sys.exit(f"Error: File not found: {file2_path}")

    print(f"Processing: {file1_path} + {file2_path}")

    # 2. Load Data
    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
    except Exception as e:
        sys.exit(f"Error reading CSV files: {e}")

    # 3. Identify Common Columns
    common_cols = set(df1.columns).intersection(set(df2.columns))
    
    if join_key not in common_cols:
        sys.exit(f"Error: Join key '{join_key}' is not present in both files.")

    # 4. Merge
    # how='inner' keeps only IDs present in BOTH files.
    merged_df = pd.merge(df1, df2, on=join_key, how='inner', suffixes=('_file1', '_file2'))

    # 5. Filter for Common Columns only
    # We reconstruct the column list to only keep the ID and the data that existed in both original files
    final_columns = [join_key]
    
    for col in common_cols:
        if col == join_key:
            continue
        final_columns.append(f"{col}_file1")
        final_columns.append(f"{col}_file2")

    result = merged_df[final_columns]

    # 6. Save
    result.to_csv(output_path, index=False)
    print(f"Success. Merged data saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two CSV files via CLI, keeping only common columns.")
    
    parser.add_argument("file1", help="Path to first CSV")
    parser.add_argument("file2", help="Path to second CSV")
    parser.add_argument("--key", default="ID", help="Column name to merge on (Default: ID)")
    parser.add_argument("--out", default="merged_output.csv", help="Output filename (Default: merged_output.csv)")

    args = parser.parse_args()

    merge_and_keep_common(args.file1, args.file2, args.key, args.out)
