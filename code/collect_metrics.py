import os
import mne
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from collect_data import num_channel

freq_bands = {
    'delta': [0, 2],
    'theta': [2, 4],
    'alpha': [4, 7],
    'beta': [7, 16],
}

def get_correlation(meg_tensor_test, pred_meg_y):
    real_target = []
    for channel in range(num_channel):
        y_test = meg_tensor_test[:, channel, :, :].reshape(meg_tensor_test.shape[0], -1)
        real_target.append(y_test)
    pred_meg_y = pred_meg_y.permute(1, 0, 2)
    real_target = torch.tensor(np.array(real_target))
    real_target = real_target.permute(1, 0, 2)
    correlations = np.array([np.corrcoef(real_target[:,i].reshape(-1), pred_meg_y[:,i].reshape(-1))[0, 1] 
                             for i in range(num_channel)])
    return pred_meg_y, real_target, correlations


def get_topomap(raw, correlations, vlim, cmap='RdBu_r', sphere=0.13, extrapolate='local', size=8.5):
    meg_indices = mne.pick_types(raw.info, meg=True)
    meg_channel_positions = np.array([raw.info['chs'][i]['loc'][:2] for i in meg_indices])
    print('meg_channel_positions.shape: ', meg_channel_positions.shape)

    correlations = np.array(correlations).reshape(-1)
    print('correlations.shape: ', correlations.shape)
    fig, ax = plt.subplots()
    topomap = mne.viz.plot_topomap(correlations, meg_channel_positions, ch_type='meg',
                                names=raw.info['ch_names'], sphere=sphere,
                                image_interp='cubic', extrapolate=extrapolate,
                                border='mean', size=size, cmap=cmap, axes=ax, 
                                vlim=vlim, show=False)
    cbar = plt.colorbar(topomap[0], ax=ax, fraction=0.02, pad=0.1)   
    cbar.set_label('Correlation')
    fig.set_size_inches(10, 8)  
    plt.show()


def get_pixel_corr(real_target, pred_meg_y):
    real_target_reshaped = real_target.view(real_target.shape[0], real_target.shape[1], -1).cpu().numpy()
    pred_meg_y_reshaped = pred_meg_y.view(pred_meg_y.shape[0], pred_meg_y.shape[1], -1).cpu().numpy()
    pixel_corr = np.array([
            np.corrcoef(real_target_reshaped[:, i, j], pred_meg_y_reshaped[:, i, j])[0, 1]
            for i in range(num_channel)
            for j in range(real_target_reshaped.shape[2])
            ])
    pixel_corr = pixel_corr.reshape(num_channel, real_target_reshaped.shape[-1])
    return pixel_corr


def kld_compute(q: torch.Tensor, p_prior: torch.Tensor) -> torch.Tensor:
    """
    Computes the Kullback-Leibler Divergence (KLD) loss.
    Args:
        q (Tensor): The approximate posterior probability distribution.
        p_prior (Tensor): The prior probability distribution.
    Returns:
        Tensor: The KLD loss.
    """
    log_q = torch.log(q + 1e-20)
    kld = (torch.sum(q * (log_q - torch.log(p_prior + 1e-20)), dim=-1)).sum()
    return kld


def bands_metrics(real_target, pred_meg_y, freq_bands):
    metrics_by_band = {}

    for band_name, band_range in freq_bands.items():
        band_corr_values = []
        
        for i in range(num_channel):
            real_band_data = real_target[:, :, band_range[0]:band_range[1], :]
            pred_band_data = pred_meg_y[:, :, band_range[0]:band_range[1], :]

            real_band_data = real_band_data.reshape(real_band_data.shape[0], real_band_data.shape[1], -1)
            pred_band_data = pred_band_data.reshape(pred_band_data.shape[0], pred_band_data.shape[1], -1)

            pearson_corr = np.corrcoef(real_band_data[:,i,:].reshape(-1), pred_band_data[:,i,:].reshape(-1))[0,1]
            modified_r2 = np.abs(pearson_corr) * pearson_corr
            mse = mean_squared_error(real_band_data[:,i,:], pred_band_data[:,i,:])
            mae = mean_absolute_error(real_band_data[:,i,:], pred_band_data[:,i,:])
            mae_norm = float(mae/abs(pred_band_data[:,i,:].mean()))

            metrics_by_band.setdefault(band_name, []).append({
                'channel': i,
                'pearson_corr': pearson_corr,
                'modified_r2': modified_r2,
                'mse': mse,
                'mae': mae,
                'mae_norm': mae_norm,
            })

    return metrics_by_band


def get_kullback_vect(pred_meg_y, real_target):
    
    """
    This if I want to plot: 
        fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
        N_pred, bins_pred, patches_pred = axs[0].hist(pred_meg_y[202, 0], bins=16, density=True)
        N_real, bins_real, patches_real = axs[1].hist(real_target[202, 0], bins=16, density=True)
    
    """
    kld_vector = []
    for ch in range(num_channel):
        single_ch_every_smp = []
        for smp in range(real_target.shape[0]):
            N_pred, bins_pred = np.histogram(pred_meg_y[smp, ch], bins=16, density=True)
            N_real, bins_real = np.histogram(real_target[smp, ch], bins=16, density=True)
            kl_div = kld_compute(torch.from_numpy(N_real), torch.from_numpy(N_pred))
            single_ch_every_smp.append(kl_div)
        mean_value = sum(single_ch_every_smp) / len(single_ch_every_smp)
        kld_vector.append(mean_value)
    kld_vector = np.array(kld_vector).reshape(-1)
    
    return kld_vector



