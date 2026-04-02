import torch, re
sd = torch.load("/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Pert_model/fold_1/best_model.pt", map_location="cpu")
has_next = any(re.search(r"next|NEXT|pertNEXT|genotype_next", k) for k in sd.keys())
print("Has NEXT head? ->", has_next)
