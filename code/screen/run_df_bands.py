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
subjects_metrics = {}
model_list = ['ridge_text_clip_', 'ridge_text_gpt_', 'ridge_', 'ridge_w2v_']

for model in model_list:
    print('MODEL: ', model)
    data_frame = []
    for repo in tqdm(os.listdir(extr_path)[3:-3]):
        subject = repo[-2:]
        print('SOGGETTO: ', subject)
        subj_path = extr_path + '/' + repo + '/'
        megsp_list_session_0 = [f for f in megsp_list if f.startswith(subject) and f.split('_')[1] == '0']
        megsp_list_session_1 = [f for f in megsp_list if f.startswith(subject) and f.split('_')[1] == '1']
        meg_0_tensor_train, meg_0_tensor_valid, meg_0_tensor_test = get_splitted_tensor(megsp_list_session_0, megsp_path)
        meg_1_tensor_train, meg_1_tensor_valid, meg_1_tensor_test = get_splitted_tensor(megsp_list_session_1, megsp_path)
        meg_tensor_test = torch.cat((meg_0_tensor_test, meg_1_tensor_test), 0)
        if (model == 'ridge_text_clip_' or model == 'ridge_text_gpt_'):
            meg_tensor_test = meg_tensor_test[19:-1]

        pred_meg_y = torch.load(os.path.join(subj_path, 'meg_prediction_'+model+subject+'.pt')) 
        pred_meg_y = pred_meg_y.permute(1, 0, 2)
        pred_meg_y = pred_meg_y.reshape(pred_meg_y.shape[0], pred_meg_y.shape[1], 16, 26)
        subjects_metrics['subject_'+subject] = bands_metrics(meg_tensor_test, pred_meg_y, freq_bands_tot, nmi_compute=True)

    for subject, metrics in subjects_metrics.items():
        for band, channels in metrics.items():
            for channel_data in channels:
                row = {'Subject': subject, 'Band': band, 'Channel': channel_data['channel']}
                row.update(channel_data)
                del row['channel']  
                data_frame.append(row)
    df = pd.DataFrame(data_frame)
    df.to_csv('/data01/data/MEG/collect_data/results/csv_subj_bands/csv_'+model+'.csv', index=False)