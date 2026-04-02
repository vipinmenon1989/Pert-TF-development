#!/bin/bash
#SBATCH --job-name=cv_pert_tf_10000
#SBATCH --account=ihc
#SBATCH --partition=ihc 
#SBATCH --gres=gpu:5          # or whichever partition you use
#SBATCH --nodelist=ihc-l40s-1
#SBATCH --time=720:00:00          # 30 days
#SBATCH --mem=600G
#SBATCH --cpus-per-task=100
#SBATCH --output=//autofs/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/cvPertfscore_3000_final_gene_5fold_pert.log
#SBATCH --error=/autofs/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/cvPertfscore_3000_final_gene_5fold_pert.err
cd /autofs/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/

export MAX_FOLD_GPUS=5
source /local/projects-t3/lilab/vmenon/anaconda3/etc/profile.d/conda.sh
conda activate /autofs/projects-t3/lilab/vmenon/conda_envs/pertTF_flashv2_1

echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

python cv_Pert-TF_3000_genes.py

