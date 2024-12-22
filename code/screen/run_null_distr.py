import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collect_data import *
from collect_metrics import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

subjects_list = ['01', '02', '04', '05', '06', '07', '08', '09']
stimuli_path = meg_path + '/stimuli/audio'
megsp_path = os.path.join(meg_path, 'collect_data/megsp')
megsp_list = os.listdir(megsp_path)
extr_path = meg_path + "/collect_data"
subjects_metrics_null = {}
model_list = ['ridge_']
# model_list = ['ridge_text_clip_', 'ridge_text_gpt_', 'ridge_', 'ridge_w2v_']

for model in model_list:
    print('MODEL: ', model)
    list_subj_null = []
    for repo in tqdm(os.listdir(extr_path)[3:-3]):
        subject = repo[-2:]
        print('SOGGETTO: ', subject)
        if subject == '10':
            break
        # subj_path = extr_path + '/' + repo + '/'
        subj_path = '/srv/nfs-data/sisko/matteoc/meg/audio_stft'
        megsp_list_session_0 = [f for f in megsp_list if f.startswith(subject) and f.split('_')[1] == '0']
        megsp_list_session_1 = [f for f in megsp_list if f.startswith(subject) and f.split('_')[1] == '1']
        meg_0_tensor_train, meg_0_tensor_valid, meg_0_tensor_test = get_splitted_tensor(megsp_list_session_0, megsp_path)
        meg_1_tensor_train, meg_1_tensor_valid, meg_1_tensor_test = get_splitted_tensor(megsp_list_session_1, megsp_path)
        meg_tensor_test = torch.cat((meg_0_tensor_test, meg_1_tensor_test), 0)
        if (model == 'ridge_text_clip_' or model == 'ridge_text_gpt_'):
            meg_tensor_test = meg_tensor_test[19:-1]

        distr_list = []
        null_distr_iter = 100
        for trial in range(null_distr_iter):
            pred_meg_y = torch.load(os.path.join(subj_path, 'meg_prediction_'+model+subject+'.pt')) 
            pred_meg_y = pred_meg_y.permute(1, 0, 2)
            pred_meg_y = pred_meg_y.reshape(pred_meg_y.shape[0], pred_meg_y.shape[1], 16, 26)
            # flattened_tensor = pred_meg_y
            flattened_tensor = pred_meg_y.reshape(pred_meg_y.shape[0], pred_meg_y.shape[1], -1)
            num_elements = flattened_tensor.shape[-1]
            random_indices = np.random.permutation(num_elements)
            shuffled_tensor = flattened_tensor[:, :, random_indices]
            shuffled_tensor = shuffled_tensor.reshape(pred_meg_y.shape)
            # subjects_metrics_null['subject_'+subject] = bands_metrics(meg_tensor_test, shuffled_tensor, freq_bands)
            get_metrics = bands_metrics(meg_tensor_test, shuffled_tensor, freq_bands)
            distr_list.append(get_metrics)

        mod_r2_null = np.empty((num_channel,null_distr_iter))
        for i in range(null_distr_iter):
            temp_list = distr_list[i]['complete']
            for j, h in enumerate(temp_list):
                mod_r2_null[j,i] = h['pearson_corr']
        
        list_subj_null.append(mod_r2_null)
    np.save('/data01/data/MEG/collect_data/results/null_distr/null_distr_pc_100_'+model+'nfft.npy', list_subj_null)


       


            