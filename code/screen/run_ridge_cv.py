import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collect_data import *
from collect_metrics import *
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt

megsp_path = os.path.join(meg_path, 'collect_data/megsp')
audio_path = os.path.join(meg_path, 'collect_data/audio')
megsp_list = os.listdir(megsp_path)
audio_list = os.listdir(audio_path)

subjects = patient
for select_subj in subjects:
    print('NUM_SUBJECT: ', select_subj)
    megsp_list_session_0 = [f for f in megsp_list if f.startswith(select_subj) and f.split('_')[1] == '0']
    megsp_list_session_1 = [f for f in megsp_list if f.startswith(select_subj) and f.split('_')[1] == '1']

    audio_tensor_train, audio_tensor_valid, audio_tensor_test = get_splitted_tensor(audio_list, audio_path)
    audio_tensor_train = torch.cat((audio_tensor_train, audio_tensor_train), 0)
    audio_tensor_valid = torch.cat((audio_tensor_valid, audio_tensor_valid), 0)
    audio_tensor_test = torch.cat((audio_tensor_test, audio_tensor_test), 0)
    print('DIMENSION_AUDIO_TENSOR_TRAIN: ', audio_tensor_train.shape)
    print('DIMENSION_AUDIO_TENSOR_VALID: ', audio_tensor_valid.shape)
    print('DIMENSION_AUDIO_TENSOR_TEST: ', audio_tensor_test.shape)

    meg_0_tensor_train, meg_0_tensor_valid, meg_0_tensor_test = get_splitted_tensor(megsp_list_session_0, megsp_path)
    meg_1_tensor_train, meg_1_tensor_valid, meg_1_tensor_test = get_splitted_tensor(megsp_list_session_1, megsp_path)
    meg_tensor_train = torch.cat((meg_0_tensor_train, meg_1_tensor_train), 0)
    meg_tensor_valid = torch.cat((meg_0_tensor_valid, meg_1_tensor_valid), 0)
    meg_tensor_test = torch.cat((meg_0_tensor_test, meg_1_tensor_test), 0)
    print('DIMENSION_MEG_TENSOR_TRAIN: ', meg_tensor_train.shape)
    print('DIMENSION_MEG_TENSOR_VALID: ', meg_tensor_valid.shape)
    print('DIMENSION_MEG_TENSOR_TEST: ', meg_tensor_test.shape)

    device = torch.device("cuda:{}".format(1) if torch.cuda.is_available() else "cpu")
    print("DEVICE", device)

    pred_target = []
    cv_scores = []
    cv_alphas = []
    audio_train = audio_tensor_train.reshape(audio_tensor_train.shape[0], -1)
    audio_test = audio_tensor_test.reshape(audio_tensor_test.shape[0], -1)
    for channel in tqdm(range(num_channel)):    # 10 canali --> tempo +/- 12 minuti
        y_train = meg_tensor_train[:, channel, :, :].reshape(meg_tensor_train.shape[0], -1)
        y_test = meg_tensor_test[:, channel, :, :].reshape(meg_tensor_test.shape[0], -1)
        model_cv = RidgeCV(alphas=[1, 10, 500, 5000])
        model_cv.fit(audio_train, y_train)
        score = model_cv.score(audio_train, y_train)
        y_pred = model_cv.predict(audio_test)
        pred_target.append(y_pred)
        cv_scores.append(score)
        cv_alphas.append(model_cv.alpha_)


    save_pred_target = os.path.join(meg_path, 'collect_data/results_'+select_subj+'/meg_prediction_ridgeCV_'+select_subj+'.pt')
    torch.save(torch.tensor(pred_target), save_pred_target)
    save_mse = os.path.join(meg_path, 'collect_data/results_'+select_subj+'/meg_score_ridgeCV_'+select_subj+'.pt')
    torch.save(torch.tensor(cv_scores), save_mse)
    save_alphas = os.path.join(meg_path, 'collect_data/results_'+select_subj+'/meg_alphas_ridgeCV_'+select_subj+'.pt')
    torch.save(torch.tensor(cv_alphas), save_alphas)