#!/bin/bash
#SBATCH --job-name=recover_umap
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --nodelist=ihc-h200-1
#SBATCH --output=/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/CV_imporved_code_second_time/Identity_complete/fold_1/recover_%j.out
#SBATCH --error=/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/CV_imporved_code_second_time/Identity_complete/fold_1/recover_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

# 1. Which fold do you want to recover? (Change this to 1, 2, 3, etc.)
FOLD_TO_RUN=1

# 2. Path to your data (The exact file you used for training)
DATA_PATH="/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/CV_imporved_code_second_time/Identity_complete/preprocessed_hvg3000_thresh0.08.h5ad"

# 3. Path to the folder CONTAINING your 'fold_1', 'fold_2' folders
# (Based on your previous paths, it seems your results are here)
BASE_OUTDIR="/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/CV_imporved_code_second_time/Identity_complete/"

# ------------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------------
source ~/.bashrc
conda activate pertTF_flashv2_1

echo "[slurm] Starting recovery for Fold ${FOLD_TO_RUN}..."
echo "[slurm] Data: ${DATA_PATH}"
echo "[slurm] Base Dir: ${BASE_OUTDIR}"

python3  complete_umap.py \
    --data "${DATA_PATH}" \
    --outdir "${BASE_OUTDIR}" \
    --fold ${FOLD_TO_RUN} \
    --n-splits 5

echo "[slurm] Done."
