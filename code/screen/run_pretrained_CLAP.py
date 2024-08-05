import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mne_bids
from transformers import AutoFeatureExtractor, ClapModel, ClapProcessor, ClapAudioModel, ClapTextModel
from transformers import AutoProcessor, AutoTokenizer
import torch
from datasets import load_dataset
from collect_data import *
from collect_metrics import *
from sklearn.linear_model import Ridge, RidgeCV
import torchaudio


stimuli_feature = "audio"    # audio or text
subjects = ['01', '06', '07', '08', '09']
# subjects = ['01']
tasks_with_sound_ids = {
            'lw1': lw1,
            'cable_spool_fort': cable_spool_fort,
            'easy_money': easy_money,
            'the_black_willow': the_black_willow
}
stimuli_path = meg_path + '/stimuli/audio'
wav_files_duration = {}
for filename in os.listdir(stimuli_path):
    if filename.endswith('.wav'): 
        file_path = os.path.join(stimuli_path, filename)
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        wav_files_duration[filename.rstrip('.wav')] = duration
# print('WAVE FILES DURATION: ',wav_files_duration)
print('WAVE FILES WITH\ NUMBERS: ',task)
wav_list_without_numb = list(task.keys())

audio_feat_train = torch.load('/data01/data/MEG/collect_data/audio/clap_audio/audio_feat_train.pt')
audio_feat_test = torch.load('/data01/data/MEG/collect_data/audio/clap_audio/audio_feat_test.pt')

for subject in subjects: 
    megsp_path = os.path.join(meg_path, 'collect_data/megsp')
    megsp_list = os.listdir(megsp_path)
    print('NUM_SUBJECT: ', subject)
    megsp_list_session_0 = [f for f in megsp_list if f.startswith(subject) and f.split('_')[1] == '0']
    megsp_list_session_1 = [f for f in megsp_list if f.startswith(subject) and f.split('_')[1] == '1']
    meg_0_tensor_train, meg_0_tensor_valid, meg_0_tensor_test = get_splitted_tensor(megsp_list_session_0, megsp_path)
    meg_1_tensor_train, meg_1_tensor_valid, meg_1_tensor_test = get_splitted_tensor(megsp_list_session_1, megsp_path)
    meg_tensor_train = torch.cat((meg_0_tensor_train, meg_1_tensor_train), 0)
    meg_tensor_valid = torch.cat((meg_0_tensor_valid, meg_1_tensor_valid), 0)
    meg_tensor_test = torch.cat((meg_0_tensor_test, meg_1_tensor_test), 0)
    print('DIMENSION_MEG_TENSOR_TRAIN: ', meg_tensor_train.shape)
    print('DIMENSION_MEG_TENSOR_VALID: ', meg_tensor_valid.shape)
    print('DIMENSION_MEG_TENSOR_TEST: ', meg_tensor_test.shape)

    pred_target = []
    mse_scores = []
    real_target = []
    audio_train = audio_feat_train.detach().numpy()
    print('reshape audio train pre', audio_train.shape)
    audio_train = audio_train.reshape(audio_train.shape[0], -1)
    audio_test = audio_feat_test.detach().numpy()
    print('reshape audio test pre', audio_test.shape)
    audio_test = audio_test.reshape(audio_test.shape[0], -1)
    for channel in tqdm.trange(num_channel):   
        y_train = meg_tensor_train[:, channel, :, :].reshape(meg_tensor_train.shape[0], -1)
        y_test = meg_tensor_test[:, channel, :, :].reshape(meg_tensor_test.shape[0], -1)
        # model = Ridge(alpha=5000, max_iter=1000)
        model = RidgeCV(alphas=[5000])
        model.fit(audio_train, y_train)
        y_pred = model.predict(audio_test)
        mse = mean_squared_error(y_test, y_pred)
        pred_target.append(y_pred)
        real_target.append(y_test)
        mse_scores.append(mse)

    print('pred_target_shape: ', torch.tensor(pred_target).shape)
    save_pred_target = os.path.join(meg_path, 'collect_data/results_'+subject+'/meg_prediction_ridge_audio_CLAP_'+subject+'.pt')
    torch.save(torch.tensor(pred_target), save_pred_target)






