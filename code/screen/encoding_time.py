import numpy as np
import matplotlib.pyplot as plt
import torch 
from tqdm import tqdm
from scipy import stats
from statsmodels.stats.multitest import multipletests


device = "cuda:3"
correlation_matrix = torch.load('/srv/nfs-data/sisko/matteoc/meg/brain_data/00_brain_corr_tot.pt').to(device)
y_val_true = torch.load('/srv/nfs-data/sisko/matteoc/meg/brain_data/00_brain_y_val_true.pt').to(device)
predict = torch.load('/srv/nfs-data/sisko/matteoc/meg/brain_data/00_brain_y_val_pred.pt').to(device)

n_permutations = 100
pvals_perm = torch.zeros_like(correlation_matrix, device=device)
pred_corr_matrix = torch.zeros_like(correlation_matrix, device=device)

# Assuming the first dimension is now for subjects
for subj in range(correlation_matrix.shape[0]):  # Loop over subjects
    print('SUBJECT: ', subj)
    for ch in tqdm(range(correlation_matrix.shape[1])):
        for tp in range(correlation_matrix.shape[2]):
            real_pc = correlation_matrix[subj, ch, tp]
            perm_pcs = []

            for _ in range(n_permutations):
                # Shuffle predictions along the time dimension for each subject and channel
                permuted_pred = predict[subj, :, ch, tp][torch.randperm(predict.shape[1])]
                
                # Convert tensors to CPU and numpy for scipy.stats.pearsonr
                perm_pc, _ = stats.pearsonr(
                    y_val_true[subj, :, ch, tp].cpu().numpy(),
                    permuted_pred.cpu().numpy()
                )
                perm_pcs.append(perm_pc)

            # Calculate the p-value for the permutation test
            pval = torch.mean((torch.tensor(perm_pcs) >= real_pc.cpu()).float())
            pvals_perm[subj, ch, tp] = pval
            pred_corr_matrix[subj, ch, tp] = torch.mean(torch.tensor(perm_pcs))

torch.save(pvals_perm, '/srv/nfs-data/sisko/matteoc/meg/brain_data/00_brain_pval_null.pt')
torch.save(pred_corr_matrix, '/srv/nfs-data/sisko/matteoc/meg/brain_data/00_brain_corr_null.pt')

