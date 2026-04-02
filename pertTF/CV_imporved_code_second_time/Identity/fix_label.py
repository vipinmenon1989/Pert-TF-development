import scanpy as sc
import numpy as np

# 1. Load your regressed/preprocessed file
input_file = "preprocessed_regressed_for_retraining.h5ad"
ad = sc.read_h5ad(input_file)

# 2. Logic to isolate 'Ghost' cells (HPC-safe)
# We target cells labeled ESC that are physically near the DE cluster 
# or have the S-score signature you identified.
s_threshold = ad.obs['S_score'].quantile(0.90) # Top 10% of S-phase activity

# Identifying the Ghost cells
ghost_mask = (ad.obs["celltype_2"] == "ESC") & (ad.obs["S_score"] > s_threshold)

# 3. Apply the 'ESC(D3)' label
ad.obs["celltype_2"] = ad.obs["celltype_2"].astype(str)
ad.obs.loc[ghost_mask, "celltype_2"] = "ESC(D3)"

# 4. Cleanup and Save
ad.obs["celltype_2"] = ad.obs["celltype_2"].astype("category")
output_file = "final_training_data_with_D3_label.h5ad"
ad.write_h5ad(output_file)

print(f"[HPC SUCCESS] {ghost_mask.sum()} cells labeled as ESC(D3).")
