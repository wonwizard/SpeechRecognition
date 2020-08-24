import yaml
from util.prepare_dataset import load_dataset,create_dataloader
from util.functions import log_parser,batch_iterator, collapse_phn
from model.las_model import Listener,Speller
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import time
import os.path as path
import os as os
import matplotlib.pyplot as plt



# Load example config file for experiment
config_path = 'config/las_example_config.yaml'
conf = yaml.load(open(config_path,'r'))


listener_model_path = path.join("checkpoint3"+"listener")
speller_model_path = path.join("checkpoint3"+"speller")



import editdistance as ed
from glob import glob
import re as re
import os as os
import json as _json
import os.path as path
import soundfile as sf
import utils as utils
import random as _random
import utils.hangul as _hangul

from random import shuffle as shuffle
from importlib import import_module as import_module
from functools import wraps
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from six.moves import xrange as range

from abc import ABC, abstractmethod

import pandas as pd
import random as random
import tensorflow as tf
from tqdm import tqdm


import pickle as _pickle
import os.path as _path
from random import shuffle as _shuffle
import numpy as _np

learning_rate=0.0001
n_classes=56

batch_size=4
feat_len =13
max_label_len=26   #max_label_len+2해서 sos eos 추가
max_time_step=448
pre_trained=True

ctc_labels = _hangul.JAMOS
labels = [" "] + ctc_labels

def load_data(data_dir):

  assert _path.isdir(data_dir)
  load_rawdata = getattr(import_module("utils.data"),"opendict")
  preprocess_audio = getattr(import_module("utils.audio"), "mfcc")
  preprocess_text = getattr(import_module("utils.hangul"), "jamo_token")
  cache_path = _path.join(data_dir, "opendict" + "_cache.pkl")

  print("Loading cached data.")
  with open(cache_path, "rb") as f:
      cache = _pickle.load(f)
  return [[audio, preprocess_text(text)] for (audio, text) in cache]


data=load_data('/home/suji93/data/')
data = list(filter(lambda x: x[0] is not None,data))

#max_len=max(list(map(lambda x:len(x[1]), data)))+2


def batch_generator(data, batch_size,idx=0, num_workers=1, bin_size=100, bin_num=10):

  tmp = []
  bin_idx = 0
  for _ in range(bin_num):
    tmp.append(list(filter(lambda x: bin_idx <= len(x) and len(x) < bin_idx+bin_size,data)))
    bin_idx += bin_size
  order = list(range(0, len(data), batch_size))
  while True:
    shuffle(order)
    for i in range(bin_num):
      shuffle(tmp[i])
    data = sum(tmp, [])
    for i in order:
      if i % num_workers == idx:
        yield list(zip(*data[i:(i+batch_size)]))
        
_random.seed(29520)
_random.shuffle(data)
        
n_train = int(len(data) * 0.85/batch_size) * batch_size
n_valid=n_train+int(len(data)*0.05)
n_test=n_valid+int(len(data)*0.1)
test_set=data[n_valid:n_test]
valid_set=data[n_train:n_valid]
batchs = {
      "train": batch_generator(data[:n_train], batch_size),
      "valid": batch_generator(data[n_train:n_valid], batch_size),
      "test": batch_generator(data[n_valid:n_test], batch_size)
  }




def OneHotEncode(Y,max_label_len,max_idx):
    new_y = np.zeros((len(Y),max_label_len,max_idx))
    for idx,label_seq in enumerate(Y):
        cnt = 1
        for label in label_seq:
            new_y[idx,cnt,label] = 1.0
            cnt += 1
        new_y[idx,cnt,1] = 1.0 # <eos>
    return new_y




#######training set##############

ctc_labels = _hangul.JAMOS
labels = [" "] + ctc_labels
jamo2index = {k:(v+2) for v,k in enumerate(labels)}
index2jamo = {(v+2):k for v,k in enumerate(labels)}

def encode(string):
    return list(map(lambda c: jamo2index[c], string))

def decode(string):
    return list(map(lambda c: index2jamo[c], string))



if pretrained:
    listener=torch.load(listener_model_path)
    speller=torch.load(speller_model_path)
    
else:
    listener = Listener(**conf['model_parameter'])
    speller = Speller(**conf['model_parameter'])
optimizer = torch.optim.Adam([{'params':listener.parameters()}, {'params':speller.parameters()}], 
                              lr=conf['training_parameter']['learning_rate'])




def preprocess(data):
    audio=data[0]
    label=data[1]
    
    #zero padding
    input_len_feed = np.asarray(list(map(len, audio)))
    inputs_feed = np.zeros((len(audio), max_time_step, feat_len), np.float32)
    
    #one-hot
    preprocess_text=[encode(label[i]) for i in range(len(label))]
    label=OneHotEncode(preprocess_text,max_label_len,max_idx=n_classes)
    for i, l in enumerate(input_len_feed):
      inputs_feed[i, :l, :] = audio[i]
    return (inputs_feed, label)




   

def LetterErrorRate2(pred_y,true_y):
    ed_accumalate = []
    for p,t in zip(pred_y,true_y):
        compressed_t = [w for w in t if (w!=1 and w!=0)]
        compressed_p = []
        for p_w in p:
            if p_w == 0:
                continue
            if p_w == 1:
                break
            compressed_p.append(p_w)
            
        ed_accumalate.append(ed.eval(compressed_p[1:-1],compressed_t[1:-1])/len(compressed_t))
    return ed_accumalate   

def model(batch_audio, batch_text, is_training):
        
             
     batch_input = Variable(torch.from_numpy(batch_audio)).type(torch.FloatTensor).cuda()
     batch_label = Variable(torch.from_numpy(np.array(batch_text)), requires_grad=False).cuda()
     objective = nn.CrossEntropyLoss(ignore_index=0).cuda()
     #objective = nn.NLLLoss(ignore_index=0)
     

     optimizer.zero_grad()
     listner_feature = listener(batch_input)
 
     if is_training:
         raw_pred_seq, attention_record = speller(listner_feature,ground_truth=batch_label,teacher_force_rate=tf_rate)
            
     else:
         raw_pred_seq, attention_record = speller(listner_feature,ground_truth=None,teacher_force_rate=0)
            
     #output_class_dim=56
     pred_y = torch.cat([torch.unsqueeze(each_y,1) for each_y in raw_pred_seq],1).view(-1,n_classes+1)
     true_y = torch.max(batch_label,dim=2)[1].view(-1)

     loss = objective(pred_y,true_y)
     
     if is_training:
         loss.backward()
         optimizer.step()
     

        
     batch_loss = loss.cpu().data.numpy()
     batch_ler = LetterErrorRate(torch.max(pred_y,dim=1)[1].cpu().data.numpy().reshape(batch_size,max_label_len),
                                    true_y.cpu().data.numpy().reshape(batch_size,max_label_len))
     
    
     pred=torch.max(pred_y,dim=1)[1].cpu().data.numpy().reshape(batch_size,max_label_len)
     true=true_y.cpu().data.numpy().reshape(batch_size,max_label_len)
    
     batch_ler=sum(batch_ler)/len(batch_ler)
     return batch_loss,batch_ler, pred, true


###############training##################
    


best_ler = 1.0
traing_log = open('log'+'train.log','w')

num_epochs=10
for epoch in range(num_epochs):

    epoch_head = time.time()
    loss = 0.0
    ler = 0.0
    val_loss = 0.0
    val_ler = 0.0

    # Teacher forcing rate linearly decay
    tf_rate = 0.8 - (0.8-0.0)
    

    for _ in range(int(n_train/batch_size)):
        
        audio_train,label_train=preprocess(list(next(batchs["train"])))
        train_loss,train_ler,tr_pred_y, tr_true_y=model(audio_train, label_train,True)
        
        loss += train_loss
        ler+= train_ler
        
    tr_loss=loss/int(n_train/batch_size)
    tr_ler=ler/int(n_train/batch_size)
    training_time = float(time.time()-epoch_head)
        
    print('Train Error Rate',tr_ler)
    print('Train loss',tr_loss)
    

        
    audio_valid,label_valid=preprocess(list(next(batchs["valid"])))
    val_loss,val_ler,val_pred_y, val_true_y=model(audio_valid, label_valid,False)

        
    # Checkpoint
    
    print('valid ler',val_ler)
    print('valid loss',val_loss)
    print('val prediction', val_pred_y)
    print('val true', val_true_y)
    torch.save(listener, listener_model_path)
    torch.save(speller, speller_model_path)

###################test####################################



dfa = {0:[-1,-1, 1, 1,-1, 1,-1,-1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0],
  1:[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
  2:[-1, 3, 3, 3, 3, 3, 3, 3, 3,-1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,-1, 3, 3, 3, 3, 3,-1, 3, 3, 3, 3, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 2],
  3:[ 0,-1, 1, 1,-1, 1,-1,-1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 3]}


def accepts(transitions,initial,accepting,s):
    state = initial
    for c in s:
        state = transitions[state][c]
        if state==-1:
            break
    return state in accepting


class WER:
    def __init__(self, true_data, pred_data):
        self.data1 = true_data
        self.data2 = pred_data
    
    def compress(self,label):
        i=0
        val=[]
        while label[i]!=1:
            val.append(label[i])
            i+=1
        return val[1:]

    def sep(self,text):   
        mo=[]
        new=[]
        for i in text:
            if i!=2:
                new.append(i)
            else:
                mo.append(new)
                new=[]
        mo.append(new)
        return mo
    
    def count2(self, text):
        count=0
        for i in text:
            if i==2:
                count+=1        
        return count
    
    
    def wer(self):
        text1=self.compress(self.data1)
        text2=self.compress(self.data2)
        
        min_num=min(self.count2(text1),self.count2(text2)) 
                
        score=0
        i=0
        #2의 개수 
        while i<=min_num:
            
            if self.sep(text1)[i]!=self.sep(text2)[i]:
                score+=1
            else:
                score+=0
            i+=1
        total_wer=score/len(text1)
        return total_wer

    


ler=0.0
for _ in range(int(n_test/batch_size)):
    
    audio_test,label_test=preprocess(list(next(batchs["test"])))
    te_ler,te_pred_y, te_true_y=model(audio_test, label_test,False)

    ler+= te_ler
    
    wer=0
    for i in range(len(te_pred_y)):
        if accepts(dfa,0,{3},te_pred_y[i])==True:
            wer+=WER(te_true_y[i],te_pred_y[i])
            
test_wer=wer/int(n_test/batch_size)
print('test_wer',wer)
    
test_ler=ler/int(n_test/batch_size)
print('Testing Error Rate',test_ler)



