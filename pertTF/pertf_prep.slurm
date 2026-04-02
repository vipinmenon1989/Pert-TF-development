#!/bin/bash
#SBATCH --job-name=perttf_prep
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --nodelist=ihc-h200-1
#SBATCH --cpus-per-task=100
#SBATCH --mem=550G
#SBATCH --time=78:00:00
#SBATCH --output=/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/CV_imporved_code_second_time/prep_%x_%j.out
#SBATCH --error=/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/CV_imporved_code_second_time/prep_%x_%j.err

set -euo pipefail

cd /local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/

source /local/projects-t3/lilab/vmenon/anaconda3/etc/profile.d/conda.sh
conda activate /autofs/projects-t3/lilab/vmenon/conda_envs/pertTF_flashv2_1

RAW_H5AD="/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/object_integrated_assay3_annotated_final.modified.cleaned.updated.celltype2_only.h5ad"
OUTDIR="/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/CV_imporved_code_second_time/"

python pertf_prep_2.py \
  --raw-h5ad "$RAW_H5AD" \
  --outdir "$OUTDIR" \
  --seed 42 \
  --keep-wt-frac 0.1 \
  --n-bins 51 \
  --n-hvg 3000

echo "DONE. OUTDIR=$OUTDIR"
echo "Preprocessed: $OUTDIR/preprocessed_hvg3000.h5ad"
