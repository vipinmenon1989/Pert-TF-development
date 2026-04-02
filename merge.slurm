#!/bin/bash
#SBATCH --job-name=pert_tf
#SBATCH --account=ihc
#SBATCH --partition=ihc         # or whichever partition you use
#SBATCH --nodelist=ihc-h200-1
#SBATCH --time=36:00:00          # 30 days
#SBATCH --mem=350G
#SBATCH --output=/local/projects-t3/lilab/vmenon/Pert-TF-model/Pertfscore.log
#SBATCH --error=/local/projects-t3/lilab/vmenon/Pert-TF-model/Pertfscore.err

cd /local/projects-t3/lilab/vmenon/Pert-TF-model/

source /local/projects-t3/lilab/vmenon/anaconda3/etc/profile.d/conda.sh
conda activate scgpt_menon

python merge_previous_dataset.py

