#!/bin/bash
#SBATCH --job-name=cv_pert_tf_10000
#SBATCH --account=ihc
#SBATCH --partition=ihc         # or whichever partition you use
#SBATCH --nodelist=ihc-grid-1-1-1
#SBATCH --time=720:00:00          # 30 days
#SBATCH --mem=600G
#SBATCH --output=/autofs/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/valid_plot_pert.log
#SBATCH --error=/autofs/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/valid_plot_pert.err

cd /local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Pert_model/
source /local/projects-t3/lilab/vmenon/anaconda3/etc/profile.d/conda.sh
conda activate /autofs/projects-t3/lilab/vmenon/conda_envs/pertTF_flashv2_1

python validation_time.py
# python validation_check.py
# python validation_datamaker.py
#python validation_data_checker.py
