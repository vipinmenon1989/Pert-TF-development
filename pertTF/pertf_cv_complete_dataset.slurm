#!/bin/bash
#SBATCH --job-name=perttf_full        # CHANGED: "cv" -> "full" to match the script
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --nodelist=ihc-l40s-1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=150G
#SBATCH --time=74:00:00
#SBATCH --output=/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/CV_imporved_code_third_time/Identity_complete_petrub_classifier_1/full_train_%j.out  # CHANGED: Removed %a
#SBATCH --error=/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/CV_imporved_code_third_time/Identity_complete_petrub_classifier_1/full_train_%j.err   # CHANGED: Removed %a

set -euo pipefail

# 1. Setup Environment
cd /local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/
source /local/projects-t3/lilab/vmenon/anaconda3/etc/profile.d/conda.sh
conda activate /autofs/projects-t3/lilab/vmenon/conda_envs/pertTF_flashv2_1

# 2. Define Variables
OUTDIR="/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/CV_imporved_code_third_time/Identity_complete_petrub_classifier_1"
PREPROCESSED_H5AD="$OUTDIR/preprocessed_hvg3000_thresh0.08.h5ad"

# 3. Export for Python Script
export OUTDIR PREPROCESSED_H5AD
export SEED=42
export MAX_EPOCHS=100
export MAX_CELLS="none"
# Note: N_SPLITS and FOLD_ID are not needed for train_full.py

# 4. Performance Tuning
export OMP_NUM_THREADS=18
export MKL_NUM_THREADS=18
export OPENBLAS_NUM_THREADS=18
export NUMEXPR_NUM_THREADS=18

# 5. Logging (WandB)
export WANDB_MODE=online
export WANDB_API_KEY="3f42c1f651e5c0658618383b0a787f06656bd550"
export WANDB_PROJECT="PertTF_Production" # CHANGED: Separate project for full run to avoid mixing with CV plots

# 6. GPU Checks
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[slurm] ERROR: SLURM failed to assign a GPU." >&2
    exit 1
fi

echo "[slurm] Job ID: $SLURM_JOB_ID"
echo "[slurm] Node: $(hostname)"
echo "[slurm] GPU: $CUDA_VISIBLE_DEVICES"

# 7. Execution
echo "[slurm] Starting Full Training..."
python train_full.py

echo "[slurm] Job Complete."