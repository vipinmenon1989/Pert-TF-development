import copy
import json
import time
import os
import logging
import numpy as np
import resource
import psutil
import sys, scipy.sparse as sp
os.environ["WANDB_API_KEY"]= "3f42c1f651e5c0658618383b0a787f06656bd550"
os.environ["KMP_WARNINGS"] = "off"
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')
import torch
from anndata import AnnData
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
from tqdm import tqdm
import gseapy as gp
from torchtext.vocab import Vocab
#from torchtext._torchtext import Vocab 
from collections import Counter, defaultdict
import scgpt as scg
from scgpt.tasks import GeneEmbedding
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import cm
import csv
from perttf.model.pertTF import PerturbationTFModel
from perttf.model.config_gen import generate_config
from perttf.model.train_data_gen import produce_training_datasets
from perttf.model.train_function_vipin import train, wrapper_train, eval_testdata
import wandb, random
from sklearn.model_selection import KFold
from typing import Optional
import optuna
from pathlib import Path
import json
import multiprocessing as mp
from perttf.utils.safe_config import SafeConfig
from scipy.stats import pearsonr, spearmanr

adata_eval = sc.read_h5ad("/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Pert_model/fold_1/best_epoch_e44/adata_best_validation.h5ad")

# 1. get true next expression (target)
true_next = adata_eval.layers["X_binned_next"]
if hasattr(true_next, "toarray"):
    true_next = true_next.toarray()
true_next = np.asarray(true_next, dtype=float)

# 2. get predicted next expression
# this depends on how wrapper_train stored it.
# check available keys:
print([k for k in adata_eval.obsm.keys() if "expr" in k or "mlm" in k or "mvc" in k])

# Suppose you see something like "mvc_expr_pred" or "expr_pred".
pred_next = adata_eval.obsm["expr_pred"]  # or "mvc_expr_pred", etc.
pred_next = np.asarray(pred_next, dtype=float)

# Now compute per-cell Pearson then average:
cell_pearsons = []
for i in range(true_next.shape[0]):
    r, _ = pearsonr(true_next[i, :], pred_next[i, :])
    cell_pearsons.append(r)
mean_cell_pearson = float(np.nanmean(cell_pearsons))

print("mean per-cell Pearson:", mean_cell_pearson)

# Per-gene Spearman:
gene_spearmans = []
for j in range(true_next.shape[1]):
    rho, _ = spearmanr(true_next[:, j], pred_next[:, j])
    gene_spearmans.append(rho)
mean_gene_spearman = float(np.nanmean(gene_spearmans))

print("mean per-gene Spearman:", mean_gene_spearman)
