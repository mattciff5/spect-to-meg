import mne
import h5py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn.plotting import plot_markers
from mne_bids import BIDSPath
from himalaya.backend import set_backend, get_backend
from himalaya.scoring import correlation_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import os
import tqdm
from himalaya.ridge import RidgeCV
from himalaya.kernel_ridge import KernelRidgeCV
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import fdrcorrection


patients = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
models = ['clap', 'w2v', 'gpt2', 'clap_512', 'stft']
n_permutations = 500
device_id = 4

brain_data_path = "/srv/nfs-data/sisko/matteoc/meg/save_brains"
stimulus_data_path = "/srv/nfs-data/sisko/matteoc/meg/save_stimulus"
results_path = "/srv/nfs-data/sisko/matteoc/meg/results"

inner_cv = KFold(n_splits=5, shuffle=False)
ridge_model = make_pipeline(
    RidgeCV(alphas=[0.01, 1, 10, 20, 1e2, 1e3], fit_intercept=True, cv=inner_cv)
)

def batch_pearson_corr(x, y, dim=1):
    """Pearson correlation per batch lungo dim specificata"""
    x_centered = x - x.mean(dim=dim, keepdim=True)
    y_centered = y - y.mean(dim=dim, keepdim=True)
    numerator = (x_centered * y_centered).sum(dim=dim)
    denominator = torch.sqrt((x_centered**2).sum(dim=dim) * (y_centered**2).sum(dim=dim))
    return numerator / (denominator + 1e-8)

def train_encoding(X, Y, epochs_shape, device_id=device_id):
    corrs = []
    preds = []
    kfold = KFold(2, shuffle=False)
    for train_index, test_index in tqdm.tqdm(kfold.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        scaler = StandardScaler()
        Y_train = scaler.fit_transform(Y_train)
        Y_test = scaler.transform(Y_test)

        torch.cuda.set_device(device_id)
        backend = set_backend("torch_cuda")

        X_train = backend.asarray(X_train).to(f'cuda:{device_id}')
        Y_train = backend.asarray(Y_train).to(f'cuda:{device_id}')
        X_test = backend.asarray(X_test).to(f'cuda:{device_id}')
        Y_test = backend.asarray(Y_test).to(f'cuda:{device_id}')

        ridge_model.fit(X_train, Y_train)
        Y_preds = ridge_model.predict(X_test)
        corr = correlation_score(Y_test, Y_preds).reshape(epochs_shape)

        if "torch" in get_backend().__name__:
            corr = corr.numpy(force=True)
        
        del X_train, Y_train, X_test, Y_test
        torch.cuda.empty_cache()

        corrs.append(corr)
        preds.append(Y_preds)

    return np.stack(corrs), np.concatenate(preds, axis=0).reshape(-1, *epochs_shape)


# Iterazione su ogni modello e soggetto
for model_name in models:
    stimulus_file = f"{model_name}_tensor.pt"
    audio_data = torch.load(os.path.join(stimulus_data_path, stimulus_file)).float()
    n_samples = audio_data.shape[0] // 2
    audio_data = audio_data[:n_samples]
    alpha_fdr = None
    for subj in tqdm.tqdm(patients):
        print(f"Processing subject {subj} - model {model_name}")
        brain_file = f"brain_tensor_subject_{subj}.pt"
        brain_data = torch.load(os.path.join(brain_data_path, brain_file)).float()
        if brain_data.shape[0] == n_samples:
            brain_data = brain_data
        else:
            brain_data = (brain_data[:n_samples] + brain_data[n_samples:]) / 2

        X = audio_data.reshape(audio_data.shape[0], -1)
        Y = brain_data.reshape(brain_data.shape[0], -1)
        epochs_shape = brain_data.shape[1:]
        corrs_embedding, pred_brains = train_encoding(X, Y, epochs_shape)

        save_dir = os.path.join(results_path, model_name)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"corrs_embedding_{subj}.npy"), corrs_embedding)

        n_samples, n_channels, n_timepoints = pred_brains.shape
        pred_brains_torch = torch.tensor(pred_brains, device=device_id)
        brain_data_torch = torch.tensor(brain_data, device=device_id)

        null_distribution_tp = torch.zeros((n_permutations, n_timepoints), device=device_id)
        null_distribution_ch = torch.zeros((n_permutations, n_channels), device=device_id)

        for perm in tqdm.tqdm(range(n_permutations), desc="Permutation runs"):
            shuffled_idx = torch.randperm(n_samples, device=device_id)
            y_true_perm = brain_data_torch[shuffled_idx]

            for tp in range(n_timepoints):
                corrs = batch_pearson_corr(pred_brains_torch[:, :, tp], y_true_perm[:, :, tp], dim=1)
                null_distribution_tp[perm, tp] = corrs.mean()

            for ch in range(n_channels):
                corrs_ch = batch_pearson_corr(pred_brains_torch[:, ch, :], y_true_perm[:, ch, :], dim=1)
                null_distribution_ch[perm, ch] = corrs_ch.mean()

        null_distribution_tp = null_distribution_tp.cpu().numpy()
        null_distribution_ch = null_distribution_ch.cpu().numpy()

        real_corr_tp = corrs_embedding.mean((0,1))
        print("Time correlation max:", real_corr_tp.max())
        real_corr_ch = corrs_embedding.mean((0,2))
        p_values_tp = np.mean(null_distribution_tp >= real_corr_tp[None, :], axis=0)  
        p_values_ch = np.mean(null_distribution_ch >= real_corr_ch[None, :], axis=0) 

        if model_name in ['gpt2', 'clap_512', 'clap']:
            alpha_fdr = 0.01
        else:
            alpha_fdr = 0.02
        significant_tp, pvals_corrected_tp = fdrcorrection(p_values_tp, alpha=alpha_fdr+0.03)
        significant_ch, pvals_corrected_ch = fdrcorrection(p_values_ch, alpha=alpha_fdr)
        np.save(os.path.join(save_dir, f"significant_tp_{subj}.npy"), significant_tp)
        np.save(os.path.join(save_dir, f"significant_ch_{subj}.npy"), significant_ch)

        del brain_data, pred_brains, corrs_embedding, pred_brains_torch, brain_data_torch
        torch.cuda.empty_cache()
