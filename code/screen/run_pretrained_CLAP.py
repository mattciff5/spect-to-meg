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
from sklearn.linear_model import Ridge
import torchaudio


stimuli_feature = "text"    # audio or text
subjects = ['01', '02', '04', '05', '06', '07', '08', '09']
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
print('WAVE FILES DURATION: ',wav_files_duration)
print('WAVE FILES WITH\ NUMBERS: ',task)
wav_list_without_numb = list(task.keys())

device = torch.device("cuda:{}".format(1) if torch.cuda.is_available() else "cpu")
print("DEVICE", device)

if stimuli_feature == "text":
    print('STIMULI_FEATURES: ', stimuli_feature)
    for subject in subjects:
        # no 03
        epochs_list = []
        for sess in session:
            print('---------', str(sess), '----------')
            for task in [0, 1, 2, 3]:
                print('---------', str(task), '----------')
                bids_path = mne_bids.BIDSPath(
                    subject=subject,
                    session=str(sess),
                    task=str(task),
                    datatype="meg",
                    root=meg_path,
                )
                try:
                    raw = mne_bids.read_raw_bids(bids_path)
                except FileNotFoundError:
                    print("missing", subject, sess, task)
                    pass
                raw = raw.pick_types(
                    meg=True, misc=False, eeg=False, eog=False, ecg=False
                )
                raw.load_data().filter(0.5, 30.0, n_jobs=1)
                if task == 0:
                    for sound_id in lw1:
                        epochs = get_epochs(raw, float(task), float(sound_id))
                        epochs_list.append(epochs)
                if task == 1:
                    for sound_id in cable_spool_fort:
                        epochs = get_epochs(raw, float(task), float(sound_id))
                        epochs_list.append(epochs)
                if task == 2:
                    for sound_id in easy_money:
                        epochs = get_epochs(raw, float(task), float(sound_id))
                        epochs_list.append(epochs)
                if task == 3:
                    for sound_id in the_black_willow:
                        epochs = get_epochs(raw, float(task), float(sound_id))
                        epochs_list.append(epochs)

        if subject == '01':
            train_ratio = 0.7
            val_ratio = 0.1
            tensor_list_train = []
            tensor_list_valid = []
            tensor_list_test = []
            for epoch in epochs_list:
                total_samples = len(epoch)
                train_size = int(total_samples * train_ratio)
                val_size = int(total_samples * val_ratio)
                train_tensor = epoch[:train_size]
                val_tensor = epoch[train_size:train_size + val_size]
                test_tensor = epoch[train_size + val_size:]
                tensor_list_train.append(train_tensor)
                tensor_list_valid.append(val_tensor)
                tensor_list_test.append(test_tensor)
            concat_epochs_train = mne.concatenate_epochs(tensor_list_train)
            concat_epochs_valid = mne.concatenate_epochs(tensor_list_valid)
            concat_epochs_test = mne.concatenate_epochs(tensor_list_test)

            X_train = concat_epochs_train.get_data()
            y_train = concat_epochs_train.metadata.word.to_numpy()
            X_sent_train, y_sent_train = build_phrase_dataset(X_train, y_train)
            X_test = concat_epochs_test.get_data()
            y_test = concat_epochs_test.metadata.word.to_numpy()
            X_sent_test, y_sent_test = build_phrase_dataset(X_test, y_test)
            X_valid = concat_epochs_valid.get_data()
            y_valid = concat_epochs_valid.metadata.word.to_numpy()
            X_sent_valid, y_sent_valid = build_phrase_dataset(X_valid, y_valid)

            #CLAP TEXT
            text_feat_train = []
            text_feat_test = []
            model = ClapTextModel.from_pretrained("laion/clap-htsat-fused").to("cuda:4")
            tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-fused")
            with torch.no_grad():
                inputs = tokenizer(list(y_sent_train), padding=True, return_tensors="pt")
                outputs = model(**inputs.to('cuda:4'))
                last_hidden_state = outputs.last_hidden_state.cpu()
                text_feat_train.append(last_hidden_state)
            text_feat_train=torch.stack(text_feat_train).squeeze()
            with torch.no_grad():
                inputs = tokenizer(list(y_sent_test), padding=True, return_tensors="pt")
                outputs = model(**inputs.to('cuda:4'))
                last_hidden_state = outputs.last_hidden_state.cpu()
                text_feat_test.append(last_hidden_state)
            text_feat_test=torch.stack(text_feat_test).squeeze()
            print('text_feat_train shape --> CLAP: ', text_feat_train.shape)
            print('text_feat_test shape --> CLAP: ', text_feat_test.shape)
            torch.save(text_feat_train, '/data01/data/MEG/collect_data/audio/clap_text/text_feat_train.pt')
            torch.save(text_feat_test, '/data01/data/MEG/collect_data/audio/clap_text/text_feat_test.pt')
        else:
            text_feat_train = torch.load('/data01/data/MEG/collect_data/audio/clap_text/text_feat_train.pt')
            text_feat_test = torch.load('/data01/data/MEG/collect_data/audio/clap_text/text_feat_test.pt')
            print('text_feat_train shape --> CLAP: ', text_feat_train.shape)
            print('text_feat_test shape --> CLAP: ', text_feat_test.shape)

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

        meg_tensor_train = meg_tensor_train[19:-1]
        meg_tensor_test = meg_tensor_test[19:-1]
        print('new_dim_meg_tensor_train: ', meg_tensor_train.shape)
        print('new_dim_meg_tensor_test: ', meg_tensor_test.shape)

        pred_target_text = []
        mse_scores_text = []
        real_target_text = []
        text_train = text_feat_train.detach().numpy()
        print('reshape text_train pre', text_train.shape)
        text_train = text_train.reshape(text_train.shape[0], -1)
        text_test = text_feat_test.detach().numpy()
        print('reshape text_test pre', text_test.shape)
        text_test = text_test.reshape(text_test.shape[0], -1)
        for channel in tqdm.trange(num_channel):   
            y_train = meg_tensor_train[:, channel, :, :].reshape(meg_tensor_train.shape[0], -1)
            y_test = meg_tensor_test[:, channel, :, :].reshape(meg_tensor_test.shape[0], -1)
            model = Ridge(alpha=5000, max_iter=1000)
            model.fit(text_train, y_train)
            y_pred = model.predict(text_test)
            mse = mean_squared_error(y_test, y_pred)
            pred_target_text.append(y_pred)
            real_target_text.append(y_test)
            mse_scores_text.append(mse)

        save_pred_target = os.path.join(meg_path, 'collect_data/results_'+subject+'/meg_prediction_ridge_text_CLAP_'+subject+'.pt')
        torch.save(torch.tensor(pred_target_text), save_pred_target)
        save_mse = os.path.join(meg_path, 'collect_data/results_'+subject+'/meg_mse_ridge_text_CLAP_'+subject+'.pt')
        torch.save(torch.tensor(mse_scores_text), save_mse)

else: 
    print('STIMULI_FEATURES else: ', stimuli_feature)
    for subject in subjects: 
        print('PATIENT: ', subject)
        audio_input = []
        for fiaba in range(4):
            audio_name = wav_list_without_numb[fiaba]   # to change story
            print('AUDIO_NAME: ', audio_name)
            selected_sound_ids = tasks_with_sound_ids[audio_name]
            for i in range(len(session)):
                print("SESSION: ", session[i])
                story_uid = int(task[audio_name])
                print("STORY_UID_OR_TASK: ", story_uid)
                raw = get_bids_raw(meg_path, subject, session[i], str(story_uid))
                for z, sound_id in enumerate(selected_sound_ids):
                    print("SOUND_ID: ", float(sound_id))
                    epochs_data = get_epochs(raw, float(story_uid), float(sound_id))
                    if (i == 0):
                        audio_path = stimuli_path + '/' + audio_name + '_' + str(z) + '.wav'
                        data_audio_chunks = []
                        epoch_spectr = get_meg_from_raw_epochs(epochs_data)
                        for j in range(epoch_spectr.shape[0]):
                            start = epochs_data[j]._metadata["start"].item()
                            duration = 3
                            y, sr = librosa.load(audio_path, sr=sampling_audio, offset=start, duration=duration)
                            if (y.shape[0] < duration*sampling_audio):   
                                # make padding         
                                pad_width = duration*sampling_audio - y.shape[0]
                                y = np.pad(y, (0, pad_width), mode='constant', constant_values=0)
                            data_audio_chunks.append(y)
                        audio_tensor_chunk = torch.tensor(data_audio_chunks)
                        audio_input.append(audio_tensor_chunk)
                        print('AUDIO_SPECTR_SHAPE: ', audio_tensor_chunk.shape)
        tensor_list_train = []
        tensor_list_valid = []
        tensor_list_test = []
        for file_tensor in audio_input:
            train_tensor, val_tensor, test_tensor = split_tensor(file_tensor)
            tensor_list_train.append(train_tensor)
            tensor_list_valid.append(val_tensor)
            tensor_list_test.append(test_tensor)
        audio_tensor_train = torch.cat(tensor_list_train, dim=0)
        audio_tensor_valid = torch.cat(tensor_list_valid, dim=0)
        audio_tensor_test = torch.cat(tensor_list_test, dim=0)

        audio_tensor_train = torch.cat((audio_tensor_train, audio_tensor_train), 0)
        audio_tensor_valid = torch.cat((audio_tensor_valid, audio_tensor_valid), 0)
        audio_tensor_test = torch.cat((audio_tensor_test, audio_tensor_test), 0)
        print('DIMENSION_AUDIO_TENSOR_TRAIN: ', audio_tensor_train.shape)
        print('DIMENSION_AUDIO_TENSOR_VALID: ', audio_tensor_valid.shape)
        print('DIMENSION_AUDIO_TENSOR_TEST: ', audio_tensor_test.shape)

        if subject == '12':
            #CLAP AUDIO
            audio_feat_train = []
            audio_feat_test = []
            model = ClapAudioModel.from_pretrained("laion/clap-htsat-fused").to(device="cuda")
            processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_audio, new_freq=48000)

            with torch.no_grad():
                for wv in tqdm.tqdm(list(zip(audio_tensor_train))):
                    wv = resampler(wv[0])
                    inputs = processor(audios=wv, return_tensors="pt", sampling_rate=48_000)
                    outputs = model(**inputs.to('cuda'))
                    last_hidden_state = outputs.last_hidden_state.to('cpu')
                    audio_feat_train.append(last_hidden_state)
            audio_feat_train=torch.stack(audio_feat_train).squeeze()
            with torch.no_grad():
                for wv in tqdm.tqdm(list(zip(audio_tensor_test))):
                    wv = resampler(wv[0])
                    inputs = processor(audios=wv, return_tensors="pt", sampling_rate=48_000)
                    outputs = model(**inputs.to('cuda'))
                    last_hidden_state = outputs.last_hidden_state.to('cpu')
                    audio_feat_test.append(last_hidden_state)
            audio_feat_test=torch.stack(audio_feat_test).squeeze()
            print('audio_feat_train shape --> CLAP: ', audio_feat_train.shape)
            print('audio_feat_test shape --> CLAP: ', audio_feat_test.shape)
            torch.save(audio_feat_train, '/data01/data/MEG/collect_data/audio/clap_audio/audio_feat_train.pt')
            torch.save(audio_feat_test, '/data01/data/MEG/collect_data/audio/clap_audio/audio_feat_test.pt')
        else:
            audio_feat_train = torch.load('/data01/data/MEG/collect_data/audio/clap_audio/audio_feat_train.pt')
            audio_feat_test = torch.load('/data01/data/MEG/collect_data/audio/clap_audio/audio_feat_test.pt')

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
            model = Ridge(alpha=5000, max_iter=1000)
            model.fit(audio_train, y_train)
            y_pred = model.predict(audio_test)
            mse = mean_squared_error(y_test, y_pred)
            pred_target.append(y_pred)
            real_target.append(y_test)
            mse_scores.append(mse)

        save_pred_target = os.path.join(meg_path, 'collect_data/results_'+subject+'/meg_prediction_ridge_CLAPwv_'+subject+'.pt')
        torch.save(torch.tensor(pred_target), save_pred_target)
        save_mse = os.path.join(meg_path, 'collect_data/results_'+subject+'/meg_mse_ridge_CLAPwv_'+subject+'.pt')
        torch.save(torch.tensor(mse_scores), save_mse)






