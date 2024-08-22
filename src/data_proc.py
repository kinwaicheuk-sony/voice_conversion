import torch
import numpy as np
import pickle
import random
import os
from utils import ls, preprocess_wav, melspectrogram

from params import num_samples

class DataProc(torch.utils.data.Dataset):

    def __init__(self, args, split):
        self.args = args
        # read the number of folders started with spkr_ under /workspace/data/kinwai/tt-vae-gan/voice_conversion/data/data_urmp
        data_urmp_path = args.dataset
        spkr_folders = [folder for folder in os.listdir(data_urmp_path) if folder.startswith('spkr_')]

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
        return int(total_tracks)*self.args.n_cpu

    def __getitem__(self, item):
        rslt = []
        n_spkrs = len(self.data_dict.keys())

        for i in range(0, n_spkrs):
            # chose random item based on prop distribution (length of each sample)
            random_track_idx = random.randint(0, len(self.data_dict[i])-1)
            trackname = self.data_dict[i][random_track_idx]
            waveform = preprocess_wav(trackname, source_sr=16000)
            rslt.append(self.random_sample(i,waveform))

        # prepares a random sample per speaker
        samples = {}
        for i in range(0, n_spkrs): samples[i] = np.array(rslt)[i,:]
        return samples

    def augment(self,data,sample_rate=16000, pitch_shift=0.5):
        if pitch_shift == 0 : return data
        return librosa.effects.pitch_shift(data, sample_rate, pitch_shift)

    def random_sample(self,i,waveform):
        n_samples = num_samples
        data = melspectrogram(waveform)
        assert data.shape[1] >= n_samples
        rand_i = random.randint(0,data.shape[1]-n_samples)
        data = data[:,rand_i:rand_i+n_samples]
        return np.array([data])
