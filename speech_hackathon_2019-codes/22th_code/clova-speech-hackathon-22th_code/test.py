import wavio
import torch
import numpy as np
from specaugment import spec_augment_pytorch, melscale_pytorch
import matplotlib.pyplot as plt

PAD = 0
N_FFT = 512
SAMPLE_RATE = 16000

def trim(data, threshold_attack=0.01, threshold_release=0.05, attack_margin=5000, release_margin=5000):
    data_size = len(data)
    cut_head = 0
    cut_tail = data_size

    plt.subplot(5,1,1)
    plt.plot(data)

    # Square
    w = np.power(np.divide(data, np.max(data)), 2)

    plt.subplot(5,1,2)
    plt.plot(w)

    # Gaussian kernel
    sig = 20000
    time = np.linspace(-40000, 40000)
    kernel = np.exp(-np.square(time)/2/sig/sig)

    # Smooth and normalize
    w = np.convolve(w, kernel, mode='same')
    w = np.divide(w, np.max(w))

    plt.subplot(5,1,3)
    plt.plot(w)


    # Detect crop sites
    for sample in range(data_size):
        sample_num = sample
        sample_amp = w[sample_num]
        if sample_amp > threshold_attack:
            cut_head = np.max([sample_num - attack_margin, 0])
            break

    for sample in range(data_size):
        sample_num = data_size-sample-1
        sample_amp = w[sample_num]
        if sample_amp > threshold_release:
            cut_tail = np.min([sample_num + release_margin, data_size])
            break

    print(cut_head)
    print(cut_tail)
    plt.subplot(5,1,4)
    plt.plot(data[cut_head:cut_tail])

    data_copy = data[cut_head:cut_tail]
    del w, time, kernel, data

    plt.subplot(5,1,5)
    plt.plot(data_copy)
    #plt.show()

    return data_copy


def get_spectrogram_feature(filepath, train_mode=False):
    (rate, width, sig) = wavio.readwav(filepath)
    wavio.writewav24("test.wav", rate=rate, data=sig)
    sig = sig.ravel()
    sig = trim(sig)

    stft = torch.stft(torch.FloatTensor(sig),
                      N_FFT,
                      hop_length=int(0.01*SAMPLE_RATE),
                      win_length=int(0.030*SAMPLE_RATE),
                      window=torch.hamming_window(int(0.030*SAMPLE_RATE)),
                      center=False,
                      normalized=False,
                      onesided=True)

    stft = (stft[:,:,0].pow(2) + stft[:,:,1].pow(2)).pow(0.5)

    amag = stft.clone().detach()

    amag = amag.view(-1, amag.shape[0], amag.shape[1])  # reshape spectrogram shape to [batch_size, time, frequency]
    mel = melscale_pytorch.mel_scale(amag, sample_rate=SAMPLE_RATE, n_mels=N_FFT//2+1)  # melspec with same shape

    plt.subplot(1,2,1)
    plt.imshow(mel.transpose(1,2).squeeze(), cmap='jet')

    p = 1  # always augment
    randp = np.random.uniform(0, 1)
    do_aug = p > randp
    if do_aug & train_mode:  # apply augment
        print("augment image")
        mel = spec_augment_pytorch.spec_augment(mel, time_warping_para=80, frequency_masking_para=54,
                                                time_masking_para=50, frequency_mask_num=1, time_mask_num=1)
    feat = mel.view(mel.shape[1], mel.shape[2])  # squeeze back to [frequency, time]
    feat = feat.transpose(0, 1).clone().detach()

    plt.subplot(1,2,2)
    plt.imshow(feat, cmap='jet')
    plt.show()  # display it

    del stft, amag, mel
    return feat


filepath = "./sample_dataset/train/train_data/wav_007.wav"
feat = get_spectrogram_feature(filepath, train_mode=True)

filepath = "./sample_dataset/train/train_data/wav_002.wav"
feat = get_spectrogram_feature(filepath, train_mode=True)

filepath = "./sample_dataset/train/train_data/wav_006.wav"
feat = get_spectrogram_feature(filepath, train_mode=True)

filepath = "./sample_dataset/train/train_data/wav_016.wav"
feat = get_spectrogram_feature(filepath, train_mode=True)

filepath = "./sample_dataset/train/train_data/wav_040.wav"
feat = get_spectrogram_feature(filepath, train_mode=True)