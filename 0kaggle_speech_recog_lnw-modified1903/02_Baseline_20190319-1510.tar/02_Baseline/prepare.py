# -*- encoding : utf-8 -*-

# 10개의 label과 데이터 경로를 지정한다
labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
data_path = '~/.kaggle/competitions/tensorflow-speech-recognition-challenge' 

from glob import glob
import random
import os
import numpy as np

SEED = 2018

# 리스트를 랜덤하게 셔플하는 함수이다
def random_shuffle(lst):
    random.seed(SEED)
    random.shuffle(lst)
    return lst

# 텍스트 파일을 저장할 폴더를 생성한다.
if not os.path.exists('input'):
    os.mkdir('input')

# 훈련 데이터 전체를 먼저 trn_all.txt에 저장한다
trn_all = []
trn_all_file = open('input/trn_all.txt', 'w')
# 제공된 훈련 데이터 경로를 모두 읽어온다
files = glob(data_path + '/train/audio/*/*.wav')
for f in files:
    # 배경 소음은 skip한다
    if '_background_noise_' in f:
        continue

    # 정답값(label)과 화자(speaker)정보를 파일명에서 추출한다
    label = f.split('/')[-2]
    speaker = f.split('/')[-1].split('_')[0]
    if label not in labels:
        # 10개의 label외 데이터는 20%의 확률로 unknown으로 분류하여 추가한다
        label = 'unknown'
        if random.random() < 0.2:
            trn_all.append((label, speaker, f))
            trn_all_file.write('{},{},{}\n'.format(label, speaker, f))
    else:
        trn_all.append((label, speaker, f))
        trn_all_file.write('{},{},{}\n'.format(label, speaker, f))
trn_all_file.close()


# 훈련 데이터를 화자 기반 9:1 비율로 분리한다
uniq_speakers = list(set([speaker for (label, speaker, path) in trn_all]))
random_shuffle(uniq_speakers)
cutoff = int(len(uniq_speakers) * 0.9)
speaker_val = uniq_speakers[cutoff:]

# 교차 검증용 파일을 생성한다
trn_file = open('input/trn.txt', 'w')
val_file = open('input/val.txt', 'w')
for (label, speaker, path) in trn_all:
    if speaker not in speaker_val:
        trn_file.write('{},{},{}\n'.format(label, speaker, path))
    else:
        val_file.write('{},{},{}\n'.format(label, speaker, path))
trn_file.close()
val_file.close()

# 테스트 데이터에 대해서도 텍스트 파일을 생성한다
tst_all_file = open('input/tst.txt', 'w')
files = glob(data_path + '/test/audio/*.wav')
for f in files:
    tst_all_file.write(',,{}\n'.format(f))
tst_all_file.close()
