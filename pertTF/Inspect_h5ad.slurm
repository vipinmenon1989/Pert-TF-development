#!/bin/bash
#SBATCH --job-name=Inspect
#SBATCH --account=ihc
#SBATCH --partition=ihc 
#SBATCH --gres=gpu:5          # or whichever partition you use
#SBATCH --nodelist=ihc-grid-1-1-1
#SBATCH --time=720:00:00          # 30 days
#SBATCH --mem=600G
#SBATCH --output=/autofs/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Inspect.log
#SBATCH --error=/autofs/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Inspect.err
cd /autofs/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/

source /local/projects-t3/lilab/vmenon/anaconda3/etc/profile.d/conda.sh
conda activate /autofs/projects-t3/lilab/vmenon/conda_envs/scanpy_vipin

python check_h5ad.py object_integrated_assay3_annotated_final.cleaned.h5ad

