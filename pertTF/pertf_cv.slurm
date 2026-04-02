#!/bin/bash
#SBATCH --job-name=perttf_cv
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --nodelist=ihc-h200-1
#SBATCH --array=1-5
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=150G
#SBATCH --time=74:00:00
#SBATCH --output=/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/CV_imporved_code_second_time/Identity_complete/pertf_fold_%x_%A_%a.out
#SBATCH --error=/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/CV_imporved_code_second_time/Identity_complete/pertf_fold_%x_%A_%a.err

set -euo pipefail

cd /local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/

source /local/projects-t3/lilab/vmenon/anaconda3/etc/profile.d/conda.sh
conda activate /autofs/projects-t3/lilab/vmenon/conda_envs/pertTF_flashv2_1

OUTDIR="/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/CV_imporved_code_second_time/Identity_complete"
PREPROCESSED_H5AD="$OUTDIR/preprocessed_hvg3000_thresh0.08.h5ad"

export OUTDIR PREPROCESSED_H5AD
export N_SPLITS=5
export SEED=42
export MAX_EPOCHS=100
export MAX_CELLS="none"
export FOLD_ID="$SLURM_ARRAY_TASK_ID"

# (optional but recommended)
export OMP_NUM_THREADS=18
export MKL_NUM_THREADS=18
export OPENBLAS_NUM_THREADS=18
export NUMEXPR_NUM_THREADS=18

export WANDB_MODE=online
export WANDB_API_KEY="3f42c1f651e5c0658618383b0a787f06656bd550"
export WANDB_PROJECT="PertTF"
export WANDB_RUN_GROUP="cv_${SLURM_JOB_ID}"

# ---------------- GPU Allocation (GRES) ----------------

# SLURM automatically sets CUDA_VISIBLE_DEVICES based on the --gres request.
# We explicitly set the physical ID for logging, if needed by downstream scripts.

if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    echo "[slurm] ERROR: SLURM failed to assign a GPU to this task (Check GRES status)." >&2
    exit 1
fi

export PHYSICAL_GPU_ID="$CUDA_VISIBLE_DEVICES"

echo "[slurm] job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID node=$(hostname)"
echo "[slurm] GRES Allocated GPU(s): $CUDA_VISIBLE_DEVICES"
echo "[slurm] Starting fold script on assigned GPU."

# ---------------- Run fold ----------------
python pertf_fold_2.py

# ---------------- Cleanup ----------------
echo "[slurm] done."
