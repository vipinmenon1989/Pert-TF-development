#!/bin/bash
#SBATCH --job-name=cv_pert_tf_10000
#SBATCH --account=ihc
#SBATCH --partition=ihc 
#SBATCH --nodelist=ihc-grid-1-1-1
#SBATCH --time=720:00:00          # 30 days
#SBATCH --mem=100G
#SBATCH --output=/autofs/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/valid_plot_time_identity.log
#SBATCH --error=/autofs/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/valid_plot_time_identity.err

cd /autofs/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Identity_model/
source /local/projects-t3/lilab/vmenon/anaconda3/etc/profile.d/conda.sh
conda activate /autofs/projects-t3/lilab/vmenon/conda_envs/pertTF_flashv2_1

python validation_time.py
 
