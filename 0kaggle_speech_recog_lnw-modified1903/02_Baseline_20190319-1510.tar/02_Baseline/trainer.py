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

def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_time(now, start):
    time_in_min = int((now - start) / 60)
    return time_in_min

# 학습을 위한 기본 설정값을 지정한다
BATCH_SIZE = 32  # 데이터 묶음에 해당하는 batch_size는 GPU 메모리에 알맞게 지정한다
mGPU = False  # multi-GPU를 사용할 경우에는 True로 지정한다
epochs = 20  # 모델이 훈련 데이터를 학습하는 횟수를 지정한다
mode = 'cv' # 교차 검증 모드(cv) or 테스트 모드(test)
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
trn = 'input/trn.txt' if mode == 'cv' else 'input/trn_all.txt'
tst = 'input/val.txt' if mode == 'cv' else 'input/tst.txt'


trn = [line.strip() for line in open(trn, 'r').readlines()]
wav_list = [line.split(',')[-1] for line in trn]
label_list = [line.split(',')[0] for line in trn]
# 학습용 SpeechDataset을 불러온다
traindataset = SpeechDataset(mode='train', label_to_int=label_to_int, wav_list=wav_list, label_list=label_list)

start_time = time()
for e in range(epochs):
    print("training epoch ", e)
    # learning_rate를 epoch마다 다르게 지정한다
    learning_rate = 0.01 if e < 10 else 0.001
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, speechmodel.parameters()), lr=learning_rate, momentum=0.9, weight_decay=0.00001)
    # 모델을 학습하기 위하여 .train() 함수를 실행한다
    speechmodel.train()

    total_correct = 0
    num_labels = 0
    trainloader = DataLoader(traindataset, BATCH_SIZE, shuffle=True)
    # 학습을 수행한다
    for batch_idx, batch_data in enumerate(tqdm(trainloader)):
        # batch_size 만큼의 음성 데이터(spec)와 정답값(label)을 받아온다
        spec = batch_data['spec']
        label = batch_data['label']
        spec, label = Variable(spec.cuda()), Variable(label.cuda())
        # 현재 모델의 예측값(y_pred)을 계산한다
        y_pred = speechmodel(spec)
        _, pred_labels = torch.max(y_pred.data, 1)
        correct = (pred_labels == label.data).sum()
        # 정답과 예측값간의 차이(loss)를 계산한다 
        loss = loss_fn(y_pred, label)

        total_correct += correct
        num_labels += len(label)
    
        optimizer.zero_grad()
        # loss를 기반으로 back-propagation을 수행한다
        loss.backward()
        # 모델 파라미터를 업데이트한다. (실질적 학습)
        optimizer.step()
    
    # 훈련 데이터에서의 정확률을 기록한다
    print("training accuracy:", 100. * total_correct / num_labels, get_time(time(), start_time))

    # 교차 검증 모드의 경우, 검증 데이터에 대한 정확률을 기록한다
    if mode == 'cv':
        # 현재 학습 중인 모델을 임시로 저장한다
        torch.save(speechmodel.state_dict(), '{}_cv'.format(model_name))
        
        # 검증 데이터를 불러온다
        softmax = Softmax()
        tst_list = [line.strip() for line in open(tst, 'r').readlines()]
        wav_list = [line.split(',')[-1] for line in tst_list]
        label_list = [line.split(',')[0] for line in tst_list]
        cvdataset = SpeechDataset(mode='test', label_to_int=label_to_int, wav_list=wav_list)
        cvloader = DataLoader(cvdataset, BATCH_SIZE, shuffle=False)

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
        print("cv accuracy:", 100. * num_correct / len(preds), get_time(time(), start_time))

# 학습이 완료된 모델을 저장한다
create_directory("model")
torch.save(speechmodel.state_dict(), model_name)

# 테스트 데이터에 대한 예측값을 파일에 저장한다
print("doing prediction...")
softmax = Softmax()

# 테스트 데이터를 불러온다
tst = [line.strip() for line in open(tst, 'r').readlines()]
wav_list = [line.split(',')[-1] for line in tst]
testdataset = SpeechDataset(mode='test', label_to_int=label_to_int, wav_list=wav_list)
testloader = DataLoader(testdataset, BATCH_SIZE, shuffle=False)

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
