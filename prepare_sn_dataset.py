from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import torchaudio
from transformers import AutoProcessor


class SNDataset(Dataset):
    def __init__(self,
                 root_pth_speech,
                 root_pth_noise,
                 ):
        self.root_pth_speech = root_pth_speech
        self.root_pth_noise = root_pth_noise
        self.speech_files = []
        self.noise_files=[]

        for filename in os.listdir(self.root_pth_speech):
            self.speech_files.append(filename)
        for filename in os.listdir(self.root_pth_noise):
            self.noise_files.append(filename)
        self.files=self.speech_files+self.noise_files
        self.labels=[1]*len(self.speech_files)+[1]*len(self.noise_files)


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file = self.files[idx]
        label=self.labels[idx]
        if label==1:
            root_pth=self.root_pth_speech
        else:   
            root_pth=self.root_pth_noise
        wav, sr = torchaudio.load(f"{os.path.join(root_pth, file)}")
        
        target_length = int(self.window_length * sr)
        
        num_windows = waveform.shape[1] // target_length
        windows = []
        
        for i in range(num_windows):
            start = i * target_length
            end = (i + 1) * target_length
            window = wav[:, start:end]
            
            # 预处理：将窗口长度不足5秒的部分用zero padding
            if window.shape[1] < target_length:
                padding = torch.zeros(1, target_length - window.shape[1])
                window = torch.cat((window, padding), dim=1)
            
            windows.append(window)

        # inputs = self.processor(wav,
        #                         sampling_rate=16000,
        #                         return_tensors="pt",
        #                         padding=True)

        inputs = inputs['input_values'].reshape(-1)

        return inputs, torch.FloatTensor(label)
