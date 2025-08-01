import os
import pandas as pd 
import mne_bids
import mne
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch 
from tqdm import tqdm
import ast


shift_value = 1e13
duration = 2   # seconds
n_fft_meg = 512 
hop_len_meg = n_fft_meg // 2  
n_fft_speech = 512 
hop_len_speech = n_fft_speech // 2
act_subjects = 8
num_models = 4
sampling_audio = 16000
sampling_meg = 1000
freq_cut = 30
num_channel = 208     

'''
 I task si riferiscono alle 4 storie:
 --> task 0: lw1
 --> task 1: cable_spool_fort
 --> task 2: easy_money
 --> task 3: the_black_willow
'''

meg_path = '/data01/data/MEG'
patient = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
session = ['0', '1']
task = {'lw1': 0.0, 'cable_spool_fort': 1.0, 'easy_money': 2.0, 'the_black_willow': 3.0}
task_list = ['lw1', 'cable_spool_fort', 'easy_money', 'the_black_willow']
lw1 = ['0.0', '1.0', '2.0', '3.0']
cable_spool_fort = ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0']
easy_money = ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0']
the_black_willow = ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0', '11.0']
tasks_with_sound_ids = {
    'lw1': lw1,
    'cable_spool_fort': cable_spool_fort,
    'easy_money': easy_money,
    'the_black_willow': the_black_willow
}


def get_bids_raw(meg_path, subject, session, task):
    bids_path = mne_bids.BIDSPath(
        subject = subject,
        session = session, 
        task = task, 
        datatype = "meg", 
        root = meg_path,
    )
    raw = None
    try:
        raw = mne_bids.read_raw_bids(bids_path, verbose=False)
        raw.load_data().filter(0.5, 30.0, n_jobs=1) 
        raw = raw.pick_types(
            meg=True, misc=False, eeg=False, eog=False, ecg=False
        )
    except FileNotFoundError:
        print("missing", subject, session, task)
        pass
    return raw



def get_meg_from_raw_epochs(epochs):
    data_meg = epochs.get_data()
    data_meg = data_meg * shift_value
    tensor_meg = torch.tensor(data_meg)
    return tensor_meg


def get_epochs(raw, story_uid, sound_id, decim=1):
    meta = list()
    for annot in raw.annotations:
        d = eval(annot.pop("description"))
        for k, v in annot.items():
            assert k not in d.keys()
            d[k] = v
        meta.append(d)
    meta = pd.DataFrame(meta)
    meta["intercept"] = 1.0
    meta=meta[(meta["kind"]=="word") & (meta["story_uid"]==story_uid) & 
              (meta["sound_id"]==sound_id)]  
    # if meta.empty:
        # raise ValueError(f"No matching meta entries found for story_uid: {story_uid} and sound_id: {sound_id}")

    # events/epochs
    events = np.c_[meta.onset * raw.info["sfreq"], np.ones((len(meta), 2))].astype(int)
    epochs = mne.Epochs(
        raw,
        events,
        tmin=-2.0,
        tmax=duration,      # time_window hyperparam
        decim=decim,        # how many points define the temporal window --> 3 s * 1000 sr / 1 tp 
        baseline=(-0.2, 0.0),
        metadata=meta,
        preload=True,
        event_repeated="merge",
        verbose=False
    )
    # threshold
    th = np.percentile(np.abs(epochs._data), 95)
    epochs._data[:] = np.clip(epochs._data, -th, th)
    epochs.apply_baseline()
    return epochs


def build_phrase_dataset(X, y, context_length=20, movement=1, words_after=5):   
    """
    Builds a dataset for training a machine learning model.

    Parameters:
        X (array-like): The input data.
        y (array-like): The target data.
        context_length (int, optional): The length of the input sequences. Defaults to 20.
        movement (int, optional): The step size for creating overlapping sequences. Defaults to 1.
        words_after (int, optional): The number of words to include after the context length. Defaults to 5.

    Returns:
        tuple: A tuple containing the input sequences and output sentences as NumPy arrays.
    """

    X_sent = []
    y_sent = []
    
    indices = np.arange(context_length, len(y), movement)
    for i in tqdm(indices):
        sentence = ' '.join(y[i-context_length:i+words_after])
        y_sent.append(sentence)
        # X_sent.append(X[i-context_length:i+words_after])
        X_sent.append(X[i])

    for j in range(len(X_sent)):
        matrix = X_sent[j]
        original_shape = matrix.shape
        target_shape = (context_length+words_after, num_channel, original_shape[-1])
        if original_shape[0] < target_shape[0]:
            padding =  [(0, target_dim - original_dim) for target_dim, original_dim in zip(target_shape, original_shape)]
            X_sent[j] = np.pad(matrix, padding, mode='constant', constant_values=0)

    for z in range(len(y_sent)):
        sentence = y_sent[z]
        if len(sentence) < (context_length+words_after):
            y_sent[z] += [''] * (context_length+words_after - len(sentence))

    return np.stack(X_sent), np.stack(y_sent)


def generate_sent_matrix(X, y, context_length=20, movement=1, words_after=5, num_channel=208):
    y_sent = []
    X_sent = []

    total_len = len(y)
    indices = np.arange(0, total_len, movement)

    for i in tqdm(indices):
        start = max(0, i - context_length)
        end = min(total_len, i + words_after + 1)  # +1 per includere la parola corrente

        sentence = list(y[start:end])

        if i - context_length < 0:
            pad_left = [''] * (context_length - i)
            sentence = pad_left + sentence

        if i + words_after + 1 > total_len:
            pad_right = [''] * ((i + words_after + 1) - total_len)
            sentence = sentence + pad_right

        full_sentence = ' '.join(sentence).strip()
        y_sent.append(full_sentence)

        matrix = X[i]
        original_shape = matrix.shape
        target_shape = (context_length + words_after + 1, num_channel, original_shape[-1])

        if original_shape[0] < target_shape[0]:
            pad_amount = target_shape[0] - original_shape[0]
            padding = [(0, pad_amount), (0, 0), (0, 0)]
            matrix = np.pad(matrix, padding, mode='constant', constant_values=0)

        X_sent.append(matrix)

    return np.stack(X_sent), np.array(y_sent)


def get_topomap(raw, correlations, vlim, cmap='RdBu_r', sphere=0.13, extrapolate='local', 
                image_interp='cubic', threshold=None, size=8.5, label_to_set='Correlation'):
    meg_indices = mne.pick_types(raw.info, meg=True)
    meg_channel_positions = np.array([raw.info['chs'][i]['loc'][:2] for i in meg_indices])
    print('meg_channel_positions.shape: ', meg_channel_positions.shape)
    correlations = np.array(correlations).reshape(-1)
    print('correlations.shape: ', correlations.shape)
    if threshold is not None:
        correlations = np.where(correlations > threshold, correlations, np.nan)
    fig, ax = plt.subplots()
    topomap = mne.viz.plot_topomap(correlations, meg_channel_positions, ch_type='meg',
                                names=raw.info['ch_names'], sphere=sphere,
                                image_interp=image_interp, extrapolate=extrapolate,
                                border='mean', size=size, cmap=cmap, axes=ax, 
                                vlim=vlim, show=False)
    cbar = plt.colorbar(topomap[0], ax=ax, fraction=0.02, pad=0.1)   
    cbar.set_label(label_to_set)
    fig.set_size_inches(10, 8)  
    plt.show()