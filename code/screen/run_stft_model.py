import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import multiprocessing
from collect_data import *
from collect_metrics import *
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import tqdm as tqdm
import matplotlib.pyplot as plt
from scipy.stats import entropy
# from ridge_gpu import *

megsp_path = os.path.join(meg_path, 'collect_data/megsp')
audio_path = os.path.join(meg_path, 'collect_data/audio_nfft')
megsp_list = os.listdir(megsp_path)
audio_list = os.listdir(audio_path)

path_to_save = '/srv/nfs-data/sisko/matteoc/meg/audio_stft'

subjects = ['01','02','04','05','06','07']
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

    # evice = torch.device("cuda:{}".format(1) if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print("DEVICE", device)

    pred_target = []
    mse_scores = []
    real_target = []
    # audio_train = audio_tensor_train.reshape(audio_tensor_train.shape[0], -1).to(device=device)
    # audio_test = audio_tensor_test.reshape(audio_tensor_test.shape[0], -1).to(device=device)
    audio_train = audio_tensor_train.reshape(audio_tensor_train.shape[0], -1)
    audio_test = audio_tensor_test.reshape(audio_tensor_test.shape[0], -1)

    # def process_channel(channel, audio_train, audio_test, meg_tensor_train, meg_tensor_test, pred_target, real_target, mse_scores):
    #     y_train = meg_tensor_train[:, channel, :, :].reshape(meg_tensor_train.shape[0], -1)
    #     y_test = meg_tensor_test[:, channel, :, :].reshape(meg_tensor_test.shape[0], -1)
    #     model = Ridge(alpha=5000, max_iter=1000)
    #     model.fit(audio_train, y_train)
    #     y_pred = model.predict(audio_test)
    #     mse = mean_squared_error(y_test, y_pred)
    #     pred_target.append(y_pred)
    #     real_target.append(y_test)
    #     mse_scores.append(mse)

    # processes = []
    # for channel in tqdm.trange(num_channel):
    #     p = multiprocessing.Process(target=process_channel, args=(channel, audio_train, audio_test, meg_tensor_train, meg_tensor_test, pred_target, real_target, mse_scores))
    #     processes.append(p)
    #     p.start()

    # for p in tqdm.trange(processes):
    #     p.join()

    for channel in tqdm.trange(num_channel):    
        y_train = meg_tensor_train[:, channel, :, :].reshape(meg_tensor_train.shape[0], -1)
        y_test = meg_tensor_test[:, channel, :, :].reshape(meg_tensor_test.shape[0], -1)
        model = Ridge(alpha=5000, max_iter=1000)
        # model = Ridge(alpha=5000, fit_intercept=True, device=device)
        # model = LinearRegression(C=5000,lr=1e-3, penalty='l2', n_iter=1000, fit_intercept=True)
        model.fit(audio_train, y_train)
        y_pred = model.predict(audio_test)
        mse = mean_squared_error(np.nan_to_num(y_test), np.nan_to_num(y_pred))
        pred_target.append(y_pred)
        real_target.append(y_test)
        mse_scores.append(mse)

    # def fit_and_predict(num_channel, audio_train, meg_tensor_train, meg_tensor_test, audio_test):     
    #     y_train = meg_tensor_train[:, num_channel, :, :].reshape(meg_tensor_train.shape[0], -1).to(device=device)
    #     y_test = meg_tensor_test[:, num_channel, :, :].reshape(meg_tensor_test.shape[0], -1)
    #     model = LinearRegression(C=5000, lr=1e-3, penalty='l2', n_iter=500, fit_intercept=True)
    #     model.fit(audio_train, y_train)
    #     y_pred = model.predict(audio_test)
    #     mse = mean_squared_error(np.nan_to_num(y_test), np.nan_to_num(y_pred))
    #     return model, y_pred, mse
    
    # n_jobs = 208
    # results = Parallel(n_jobs=n_jobs)(delayed(fit_and_predict)(
    #     num_channel, audio_train, meg_tensor_train, meg_tensor_test, audio_test) for num_channel in range(meg_tensor_train.shape[1]))
    # meg_channel_models, meg_channel_ypred, meg_channel_mse = zip(*results)
    # print(meg_channel_models)

    # save_mse = os.path.join(meg_path, 'collect_data/results_'+select_subj+'/meg_mse_ridge_'+select_subj+'.pt')
    # torch.save(torch.tensor(mse_scores), save_mse)
    save_pred_target = os.path.join(path_to_save, 'results_'+select_subj+'/meg_prediction_ridge_'+select_subj+'.pt')
    torch.save(torch.tensor(pred_target), save_pred_target)