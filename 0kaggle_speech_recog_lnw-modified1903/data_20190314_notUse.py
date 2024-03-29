"""
model data (trainer.py use this)

origin 머신러닝탐구생활 03구글 음성인식경진대회 baseline code pytorch
modifier naewon liu (lnw)
date 201902



"""
import torch
import numpy as np
from torch.utils.data import Dataset
import librosa
from glob import glob
import random

#lnw
data_path = '/home/neoadmin/kaggle_speech_recog/competitions_data'

# 음성 파일의 sample rate은 1초 = 16000으로 지정한다
SR = 16000

#lnw add. 파일안에서 읽을 사이즈   400ms = 0.4second
#lnw add. 200ms 만큼 쉬프트해서 다시 넣어줄 크기 200ms=0.2second
readSize = SR * 0.4
shiftSize = SR * 0.2


# 경진대회 전용 SpeechDataset 클래스를 정의한다
class SpeechDataset(Dataset):
    def __init__(self, mode, label_to_int, wav_list, label_list=None):
        self.mode = mode
        self.label_to_int = label_to_int
        self.wav_list = wav_list
        self.label_list = label_list
        self.sr = SR
        self.n_silence = int(len(wav_list) * 0.1)

        # 배경 소음 데이터를 미리 읽어온다
        # lnw data path modified
        #self.background_noises = [librosa.load(x, sr=self.sr)[0] for x in glob("~/.kaggle/competitions/tensorflow-speech-recognition-challenge/train/audio/_background_noise_/*.wav")]
        self.background_noises = [librosa.load(x, sr=self.sr)[0] for x in glob(data_path+"/train/audio/_background_noise_/*.wav")]

    def get_one_word_wav(self, idx):
        # idx 번째 음성 파일을 1초만큼 읽어온다
        wav = librosa.load(self.wav_list[idx], sr=self.sr)[0]
        if len(wav) < self.sr:
            wav = np.pad(wav, (0, self.sr - len(wav)), 'constant')
        return wav[:self.sr]

    # lnw add. get wavRaedlist (divided by readSize) 
    def get_one_word_wavReadlist(self, idx):
        # idx 번째 음성 파일을 readSize 만큼 읽어와 리스트를  던져준다
        # 한번 다  넣어준후 다시한번 shiftSize 200ms 이동한 곳에서부터 다시한번 넣어준다 
        wav = librosa.load(self.wav_list[idx], sr=self.sr)[0]
        wavReadlist = []
        sizeCount= int(len(wav)/readSize)
        for fi in range(sizeCount):
            startIndex = fi*readSize
            endIndex = (fi+1)*readSize
            if endIndex <= len(wav) : 
                wavReadlist.append(wav[startIndex:endIndex) 
        for fi in range(sizeCount):
            startIndex = fi*readSize+shiftSize
            endIndex = (fi+1)*readSize+shiftSize
            if endIndex <= len(wav) :
                wavReadlist.append(wav[startIndex:endIndex]) 
        return wavReadlist


    def get_one_noise(self):
        # 배경 소음 데이터 중 랜덤하게 1초를 읽어온다
        selected_noise = self.background_noises[random.randint(0, len(self.background_noises) - 1)]
        start_idx = random.randint(0, len(selected_noise) - 1 - self.sr)
        return selected_noise[start_idx:(start_idx + self.sr)]

    def get_mix_noises(self, num_noise=1, max_ratio=0.1):
        # num_noise 만큼의 배경 소음을 합성한다
        result = np.zeros(self.sr)
        for _ in range(num_noise):
            result += random.random() * max_ratio * self.get_one_noise()
        return result / num_noise if num_noise > 0 else result

    def get_silent_wav(self, num_noise=1, max_ratio=0.5):
        # 배경 소음 데이터를 silence로 가정하고 불러온다
        return self.get_mix_noises(num_noise=num_noise, max_ratio=max_ratio)

    def __len__(self):
        # 교차검증 모드일 경우에는 ‘silence’를 추가한 만큼이 데이터 크기이고, Test 모드일 경우에는 제공된 테스트 데이터가 전부이다
        if self.mode == 'test':
            return len(self.wav_list)
        else:
            return len(self.wav_list) + self.n_silence

    # lnw modified name for bakcup
    def __getitem_org__(self, idx):
        # idx번째 음성 데이터 하나를 반환한다
        if idx < len(self.wav_list):
            # 전처리는 mel spectrogram으로 지정한다
            # (옵션) 여기서 Data Augmentation을 수행할 수 있다.
            wav_numpy = preprocess_mel(self.get_one_word_wav(idx))
            wav_tensor = torch.from_numpy(wav_numpy).float()
            wav_tensor = wav_tensor.unsqueeze(0)

            # 음성 스펙트로그램(spec), 파일 경로(id)와 정답값(label)을 반환한다
            if self.mode == 'test':
                return {'spec': wav_tensor, 'id': self.wav_list[idx]}
            else:
                label = self.label_to_int.get(self.label_list[idx], len(self.label_to_int))
                return {'spec': wav_tensor, 'id': self.wav_list[idx], 'label': label}
        else:
            # 배경 소음을 반환한다
            wav_numpy = preprocess_mel(self.get_silent_wav(
                num_noise=random.choice([0, 1, 2, 3]),
                max_ratio=random.choice([x / 10. for x in range(20)])))
            wav_tensor = torch.from_numpy(wav_numpy).float()
            wav_tensor = wav_tensor.unsqueeze(0)
            return {'spec': wav_tensor, 'id': 'silence', 'label': len(self.label_to_int) + 1}


    #lnw add/modified  
    def __getitem__(self, idx):
        # idx번째 음성 데이터 리스트를 반환한다
        if idx < len(self.wav_list):
            # 전처리는 mel spectrogram으로 지정한다
            # (옵션) 여기서 Data Augmentation을 수행할 수 있다.
          for fi in self.get_one_word_wavReadlist(idx):
            wav_numpy = preprocess_mel(fi)
            wav_tensor = torch.from_numpy(wav_numpy).float()
            wav_tensor = wav_tensor.unsqueeze(0)

            # 음성 스펙트로그램(spec), 파일 경로(id)와 정답값(label)을 반환한다
            if self.mode == 'test':
                return {'spec': wav_tensor, 'id': self.wav_list[idx]}
            else:
                label = self.label_to_int.get(self.label_list[idx], len(self.label_to_int))
                return {'spec': wav_tensor, 'id': self.wav_list[idx], 'label': label}
        else:
            # 배경 소음을 반환한다
            wav_numpy = preprocess_mel(self.get_silent_wav(
                num_noise=random.choice([0, 1, 2, 3]),
                max_ratio=random.choice([x / 10. for x in range(20)])))
            wav_tensor = torch.from_numpy(wav_numpy).float()
            wav_tensor = wav_tensor.unsqueeze(0)
            return {'spec': wav_tensor, 'id': 'silence', 'label': len(self.label_to_int) 



# mel spectrogram 전처리 함수이다
def preprocess_mel(data, n_mels=40):
    spectrogram = librosa.feature.melspectrogram(data, sr=SR, n_mels=n_mels, hop_length=160, n_fft=480, fmin=20, fmax=4000)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram

