import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
import mne
import torchaudio
from param_funct import *
from transformers import AutoProcessor, Wav2Vec2Model


home_path_meg = '/data01/data/MEG'
stimuli_path = home_path_meg + '/stimuli/audio'
meg_path = '/srv/nfs-data/sisko/matteoc/meg'
meg_path_data = os.path.join(meg_path, 'MEG_DATA/osfstorage')
save_brain_dir = os.path.join(meg_path, 'save_brains')
save_stimulus_dir = os.path.join(meg_path, 'save_stimulus')
# patients = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
patients = ['12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
sub_decim = 10
session = ['0', '1']

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft_speech, hop_length=hop_len_speech).to(device)
db_transform = torchaudio.transforms.AmplitudeToDB().to(device)


for subject in tqdm(patients):
    print('PATIENT:', subject)
    brain_signal_data = []
    audio_spect_data = []
    audio_w2v_data = []

    for sess in range(len(session)):
        print("SESSION:", session[sess])
        for story in task_list:
            print('AUDIO_NAME:', story)
            selected_sound_ids = tasks_with_sound_ids[story]
            story_uid = int(task[story])
            print("STORY_UID:", story_uid)
            raw = get_bids_raw(meg_path_data, subject, session[sess], str(story_uid)) # change path
            if raw is None:
                print(f"Skipping missing data for patient {subject}, session {session[sess]}")
                continue
            for z, sound_id in enumerate(selected_sound_ids):
                print("SOUND_ID:", float(sound_id))
                epochs_data = get_epochs(raw, float(story_uid), float(sound_id), sub_decim)
                epoch_signal = get_meg_from_raw_epochs(epochs_data)
                print('MEG_SHAPE:', epoch_signal.shape)
                brain_signal_data.append(epoch_signal)

                if subject == '01':
                    data_audio_chunks = []
                    data_audio_spect = []
                    audio_path = f"{stimuli_path}/{story}_{z}.wav"
                    waveform, sr = torchaudio.load(audio_path)
                    if sr != sampling_audio:
                        waveform = torchaudio.functional.resample(waveform, sr, sampling_audio)
                    waveform = waveform.squeeze(0).to(device)

                    for j in range(epoch_signal.shape[0]):
                        start = epochs_data[j]._metadata["start"].item()
                        sample_start = round(start * sampling_audio)
                        sample_end = round((start + duration) * sampling_audio)
                        y = waveform[sample_start:sample_end]
                        expected_len = int(duration * sampling_audio)
                        if y.shape[0] < expected_len:
                            pad_len = expected_len - y.shape[0]
                            y = torch.nn.functional.pad(y, (0, pad_len), value=0.0)
                        elif y.shape[0] > expected_len:
                            y = y[:expected_len]
                        data_audio_chunks.append(y)
                        spec = spectrogram_transform(y.unsqueeze(0))
                        spec_db = db_transform(spec)
                        data_audio_spect.append(spec_db.squeeze(0).cpu())
                    audio_tensor_chunk = torch.stack(data_audio_chunks)
                    audio_tensor_spect = torch.stack(data_audio_spect)

                    inputs_w2v = processor(audio_tensor_chunk.cpu(), sampling_rate=sampling_audio, return_tensors="pt")
                    w2v_input = inputs_w2v.input_values.squeeze(0).to(device)
                    with torch.no_grad():
                        outputs = model(w2v_input)
                    last_hidden_w2v = outputs.last_hidden_state.cpu()

                    print('AUDIO_SPECT_SHAPE:', audio_tensor_spect.shape)
                    print('AUDIO_W2V:', last_hidden_w2v.shape)
                    audio_spect_data.append(audio_tensor_spect)
                    audio_w2v_data.append(last_hidden_w2v)

    # Salva tensore neurale per soggetto
    brain_signal_tensor = torch.cat(brain_signal_data, dim=0)
    torch.save(brain_signal_tensor, os.path.join(save_brain_dir, f"brain_tensor_subject_{subject}.pt"))
    print(f"Saved brain tensor for subject {subject} → {brain_signal_tensor.shape}")

    if subject == '01':
        audio_spect_tensor = torch.cat(audio_spect_data, dim=0)
        audio_w2v_tensor = torch.cat(audio_w2v_data, dim=0)
        torch.save(audio_spect_tensor, os.path.join(save_stimulus_dir, "stft_tensor.pt"))
        torch.save(audio_w2v_tensor, os.path.join(save_stimulus_dir, "w2v_tensor.pt"))
        print(f"Saved audio spectrogram tensor → {audio_spect_tensor.shape}")
        print(f"Saved wav2vec2 embedding tensor → {audio_w2v_tensor.shape}")
