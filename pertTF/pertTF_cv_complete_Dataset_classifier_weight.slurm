#!/bin/bash
#SBATCH --job-name=perttf_sweep
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1          # Requests 1 dedicated GPU per task
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=150G
#SBATCH --time=74:00:00
#SBATCH --array=0-4           # Run 5 jobs total
#SBATCH --output=/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/CV_imporved_code_third_time/sweep_%A_%a.out
#SBATCH --error=/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/CV_imporved_code_third_time/sweep_%A_%a.err

# NOTE: Removed "--nodelist=ihc-l40s-1". 
# This allows Slurm to put Job 1 on Node A and Job 2 on Node B if needed.

set -euo pipefail
export PYTHONUNBUFFERED=1

# 1. Setup Environment (Original FAST configuration)
source /local/projects-t3/lilab/vmenon/anaconda3/etc/profile.d/conda.sh
conda activate /autofs/projects-t3/lilab/vmenon/conda_envs/pertTF_flashv2_1

# 2. Define Weights
WEIGHTS=(6.0 7.0 8.0 9.0)
export PERT_LOSS_WEIGHT=${WEIGHTS[$SLURM_ARRAY_TASK_ID]}

# 3. Define Variables
export OUTDIR="/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/CV_imporved_code_third_time/Identity_complete_petrub_classifier_sweep"
mkdir -p "$OUTDIR"

export PREPROCESSED_H5AD="/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/CV_imporved_code_third_time/Identity_complete_petrub_classifier_1/preprocessed_hvg3000_thresh0.08.h5ad"
export SEED=42
export MAX_EPOCHS=100
export WANDB_PROJECT="PertTF_Weight_Sweep"

# 4. Performance Tuning
export OMP_NUM_THREADS=18
export MKL_NUM_THREADS=18

# 5. GPU & Node Verification ( The "Proof" )
echo "------------------------------------------------"
echo "Job ID: $SLURM_ARRAY_JOB_ID | Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Assigned GPU: $CUDA_VISIBLE_DEVICES"
echo "Weight: $PERT_LOSS_WEIGHT"
# This prints the specific UUID of the GPU to prove they are different
nvidia-smi --query-gpu=uuid,name --format=csv,noheader
echo "------------------------------------------------"

# 6. Execution
python train_full_petrubation_classifier_weight.py
