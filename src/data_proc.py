import torch
import numpy as np
import pickle
import random
import os
from utils import ls, preprocess_wav, melspectrogram
import torchaudio
from torchaudio import transforms
import torch

from params import *

def amp_to_db(x):
    return 20 * torch.log10(torch.maximum(x, torch.tensor([1e-5])))

def normalize(S):
    return torch.clip((S - min_level_db) / -min_level_db, 0, 1)

class DataProc(torch.utils.data.Dataset):

    def __init__(self, args, split):
        self.args = args
        # read the number of folders started with spkr_ under /workspace/data/kinwai/tt-vae-gan/voice_conversion/data/data_urmp
        data_urmp_path = args.dataset
        spkr_folders = [folder for folder in os.listdir(data_urmp_path) if folder.startswith('spkr_')]
        self.target_sr = 16000
        self.segment_size = 25600
        self.mel_specgram = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=fmin,
            n_mels=num_mels,
            power=2,
            )
        # in this verion data_dict is a dictionary for .wav filenames
        self.data_dict = {}
        # read the number of .wav files under each spkr_ folder
        for i in range(len(spkr_folders)):
            self.data_dict[i] = []
            for wav in os.listdir(os.path.join(data_urmp_path, spkr_folders[i])):
                if wav.endswith('.wav'):
                    self.data_dict[i].append(
                        os.path.join(
                            data_urmp_path,
                            spkr_folders[i],
                            wav)
                            )
        

    def __len__(self):
        total_tracks = 0
        # count the total number of .wav files inside self.data_dict
        for i in range(len(self.data_dict.keys())):
            total_tracks += len(self.data_dict[i])
        return int(total_tracks)# *self.args.n_cpu

    def __getitem__(self, item):
        rslt = []
        n_spkrs = len(self.data_dict.keys())

        for i in range(0, n_spkrs):
            # chose random item based on prop distribution (length of each sample)
            random_track_idx = random.randint(0, len(self.data_dict[i])-1)
            trackname = self.data_dict[i][random_track_idx]
            # waveform = preprocess_wav(trackname, source_sr=16000)
            info = torchaudio.info(trackname)
            audio_len = info.num_frames
            sr = info.sample_rate
            sr_ratio = sr / self.target_sr

               
            # sample 25600 frames from the waveform
            actual_segment_size = int(self.segment_size*sr_ratio)
            if audio_len > actual_segment_size:
                start = random.randint(0, audio_len-actual_segment_size)   
            waveform, sr = torchaudio.load(
                trackname,
                frame_offset=start,
                num_frames=actual_segment_size)

            # resample the waveform to 16000 if it is not already
            if sr != self.target_sr:
                waveform = torchaudio.functional.resample(
                    waveform,
                    sr,
                    self.target_sr
                    )                             

            waveform = waveform[0]
            rslt.append(self.random_sample(i,waveform))

        # prepares a random sample per speaker
        samples = {}
        for i in range(0, n_spkrs): samples[i] = rslt[i]
        return samples

    def augment(self,data,sample_rate=16000, pitch_shift=0.5):
        if pitch_shift == 0 : return data
        return librosa.effects.pitch_shift(data, sample_rate, pitch_shift)

    def random_sample(self,i,waveform):
        n_samples = num_samples
        data = self.mel_specgram(waveform)
        data = amp_to_db(data)-ref_level_db
        data = normalize(data)        
        assert data.shape[1] >= n_samples
        rand_i = random.randint(0,data.shape[1]-n_samples)
        data = data[:,rand_i:rand_i+n_samples]
        return data.unsqueeze(0)
