
import librosa
import numpy as np

SR = 48000

def preprocess_mel(data, n_mels=40):
    spectrogram = librosa.feature.melspectrogram(data, sr=SR, n_mels=n_mels, hop_length=160, n_fft=480, fmin=20, fmax=4000)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


wav = librosa.load("./u-96000.wav",sr=SR,mono=False)[0]
wav=wav.ravel()
spect =  preprocess_mel(wav)

#wav2 = librosa.load("./u-96000mono.wav",sr=SR)[0]
#spect2 =  preprocess_mel(wav2)