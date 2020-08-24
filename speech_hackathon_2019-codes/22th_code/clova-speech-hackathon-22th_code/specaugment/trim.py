import wave
import numpy as np
import matplotlib.pyplot as plt


def trim(data, cfg_trim):
    
    threshold_attack = cfg_trim["threshold_attack"]
    threshold_release = cfg_trim["threshold_release"]
    attack_margin = cfg_trim["attack_margin"]
    release_margin = cfg_trim["release_margin"]

    data_size = len(data)
    cut_head = 0
    cut_tail = data_size

    # Square
    w = np.power(np.divide(data, np.max(data)), 2)

    # Gaussian kernel
    sig = 20000
    time = np.linspace(-40000, 40000)
    kernel = np.exp(-np.square(time)/2/sig/sig)

    # Smooth and normalize
    w = np.convolve(w, kernel, mode='same')
    w = np.divide(w, np.max(w))

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

    #print("trimmed audio length = ", cut_tail-cut_head+1)

    data_copy = data[cut_head:cut_tail]
    del w, time, kernel, data

    return data_copy
