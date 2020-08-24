"""
model trainer
"""
from torch.autograd import Variable
from data import SpeechDataset
from torch.utils.data import DataLoader
import torch
from time import time
from torch.nn import Softmax
import numpy as np
import pandas as pd
import os
from random import choice
from resnet import ResModel
from tqdm import tqdm
#lnw
from datetime import datetime

#lnw start time
lstart_time = datetime.now()
print("Start time : "+str(lstart_time))

def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_time(now, start):
    time_in_min = int((now - start) / 60)
    return time_in_min

#  기본 설정값을 지정한다
mGPU = False  # multi-GPU를 사용할 경우에는 True로 지정한다

mode = 'predict' # 검증 모드(val:라벨있는 경우) or 예측 모드(predict:라벨정답없i경우)

model_name = 'model/model_resnet.pth'  # 모델 결과물을 저장할 때 모델 이름을 지정한다

# ResNet 모델을 활성화한다
loss_fn = torch.nn.CrossEntropyLoss()
model = ResModel
speechmodel = torch.nn.DataParallel(model()) if mGPU else model()
speechmodel = speechmodel.cuda()

# SpeechDataset을 활성화한다
labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
label_to_int = dict(zip(labels, range(len(labels))))
int_to_label = dict(zip(range(len(labels)), labels))
int_to_label.update({len(labels): 'unknown', len(labels) + 1: 'silence'})

# 모드에 따라 학습 및 검증에 사용할 파일을 선택한다
tst = 'input/val.txt' if mode == 'val' else 'input/tst.txt'


start_time = time()

# val mode
if mode == 'val' :
        print("Doing validation and save..." )

       
        # 검증 데이터를 불러온다
        softmax = Softmax()
        tst_list = [line.strip() for line in open(tst, 'r').readlines()]
        wav_list = [line.split(',')[-1] for line in tst_list]
        label_list = [line.split(',')[0] for line in tst_list]
        cvdataset = SpeechDataset(mode='test', label_to_int=label_to_int, wav_list=wav_list)
        cvloader = DataLoader(cvdataset, 1, shuffle=False)

        # 모델을 불러와 .eval() 함수로 검증 준비를 한다
        speechmodel = torch.nn.DataParallel(model()) if mGPU else model()
        speechmodel.load_state_dict(torch.load('{}_cv'.format(model_name)))
        speechmodel = speechmodel.cuda()
        speechmodel.eval()

        # 검증 데이터를 batch_size만큼씩 받아오며 예측값을 저장한다
        fnames, preds = [], []
        for batch_idx, batch_data in enumerate(tqdm(cvloader)):
            spec = Variable(batch_data['spec'].cuda())
            fname = batch_data['id']
            y_pred = softmax(speechmodel(spec))
            preds.append(y_pred.data.cpu().numpy())
            fnames += fname

        preds = np.vstack(preds)
        preds = [int_to_label[x] for x in np.argmax(preds, 1)]
        fnames = [fname.split('/')[-2] for fname in fnames]
        num_correct = 0
        for true, pred in zip(fnames, preds):
            if true == pred:
                num_correct += 1


        # 검증 데이터의 정확률을 기록한다
        print("v accuracy:", 100. * num_correct / len(preds), get_time(time(), start_time))

        # 테스트 파일 명과 예측값을 sub 폴더 아래 저장한다. 캐글에 직접 업로드 할 >수 있는 파일 포맷이다.
        create_directory("sub")
        pd.DataFrame({'fname': fnames, 'label': preds}).to_csv("sub/{}.csv".format(model_name.split('/')[-1]), index=False)




# predict
if mode == 'predict' :

     # 테스트 데이터에 대한 예측값을 파일에 저장한다
     print("Doing prediction and save file")
     softmax = Softmax()

     # 테스트 데이터를 불러온다
     tst = [line.strip() for line in open(tst, 'r').readlines()]
     wav_list = [line.split(',')[-1] for line in tst]
     testdataset = SpeechDataset(mode='test', label_to_int=label_to_int, wav_list=wav_list)
     testloader = DataLoader(testdataset, 1, shuffle=False)


     # 모델을 불러온다
     speechmodel = torch.nn.DataParallel(model()) if mGPU else model()
     speechmodel.load_state_dict(torch.load(model_name))
     speechmodel = speechmodel.cuda()
     speechmodel.eval()
    
     test_fnames, test_labels = [], []
     pred_scores = []

     # 테스트 데이터에 대한 예측값을 계산한다
     for batch_idx, batch_data in enumerate(tqdm(testloader)):
         spec = Variable(batch_data['spec'].cuda())
         fname = batch_data['id']
         y_pred = softmax(speechmodel(spec))
         pred_scores.append(y_pred.data.cpu().numpy())
         test_fnames += fname

     # 가장 높은 확률값을 가진 예측값을 label 형태로 저장한다
     final_pred = np.vstack(pred_scores)
     final_labels = [int_to_label[x] for x in np.argmax(final_pred, 1)]
     test_fnames = [x.split("/")[-1] for x in test_fnames]

     # 테스트 파일 명과 예측값을 sub 폴더 아래 저장한다. 캐글에 직접 업로드 할 수 있는 파일 포맷이다.
     create_directory("sub")
     pd.DataFrame({'fname': test_fnames, 'label': final_labels}).to_csv("sub/{}.csv".format(model_name.split('/')[-1]), index=False)



#lnw end time, duration
end_time = datetime.now()
print("End time : "+str(end_time))
print('Duration: {}'.format(end_time - lstart_time))


