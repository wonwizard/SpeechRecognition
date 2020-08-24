# -*- encoding : utf-8 -*-
#
# data prepare
# lnw modified
# 기존 라벨중  변경
# 특정크기만큼 voice noise files 잘라 저장 
# 특정 트기만큼 쉬프트후 재저장 
# 리스트 랜덤 순서 정열
#
# usage
#trndata1_path = data_path+'/train/xxx' 에 학습할 보이스 파일들 넣어둠 라벨은 이 디렉토리에 있는것들은 모두 같은 지정된 라벨로  파일로 저장함 
#valdata1_path = data_path+'/val/voice'  에 검증할 보이스 파일들 넣어둠
#testt_path = data_path+'/test'   에 라벨없는 테스트할 파일들 넣어둠
# data_path+'/input'가 있으면 모든 프로세스 하고 이미 있으면 재준비안함  재준비하게 하려면  이 디렉토리 및 파일들 모두 삭제 
# 검증용  별도 준비못해다던지 훈련과 검증용 분리용 필요할 경우 막은 것 열어서 사용
# python3 pre_pare.py
#


# setting 
# 10개의 label과 데이터 경로를 지정한다
# lnw modified.  변경  .  라벨순서와 트레인/검수  디렉토리순서가 맞아야 라벨순서대로 라젤링이 자동됨 
# train 파일 train 파일  에서의 라벨도 변경 필요 
labels = ['vocchild', 'vocfemale', 'vocmale', 'vocold', 'drone', '6', '7', '8', '9', '10']
#lnw modified 
#data_path = '~/.kaggle/competitions/tensorflow-speech-recognition-challenge' 
data_path = '/home/neoadmin/kaggle_speech_recog/competitions_data'
#lnw add
trndata1_path = data_path+'/train/vocchild'
trndata2_path = data_path+'/train/vocfemale'
trndata3_path = data_path+'/train/vocmale'
trndata4_path = data_path+'/train/vocold'
trndata5_path = data_path+'/train/drone'
#trndata6_path = data_path+'/train/06'
#trndata7_path = data_path+'/train/07'
#trndata8_path = data_path+'/train/08'
#trndata9_path = data_path+'/train/09'
#trndata10_path = data_path+'/train/10'
valdata1_path = data_path+'/val/vocchild'
valdata2_path = data_path+'/val/vocfemale'
valdata3_path = data_path+'/val/vocmale'
valdata4_path = data_path+'/val/vocold'
valdata5_path = data_path+'/val/drone'
#valdata6_path = data_path+'/val/06'
#valdata7_path = data_path+'/val/07'
#valdata8_path = data_path+'/val/08'
#valdata9_path = data_path+'/val/09'
#valdata10_path = data_path+'/val/10'
prepared_trndata1_path= data_path+'/input/trn01'
prepared_trndata2_path= data_path+'/input/trn02'
prepared_trndata3_path= data_path+'/input/trn03'
prepared_trndata4_path= data_path+'/input/trn04'
prepared_trndata5_path= data_path+'/input/trn05'
#prepared_trndata6_path= data_path+'/input/trn06'
#prepared_trndata7_path= data_path+'/input/trn07' 
#prepared_trndata8_path= data_path+'/input/trn08'
#prepared_trndata9_path= data_path+'/input/trn09' 
#prepared_trndata10_path= data_path+'/input/trn10' 
prepared_valdata1_path= data_path+'/input/val01'
prepared_valdata2_path= data_path+'/input/val02'
prepared_valdata3_path= data_path+'/input/val03'
prepared_valdata4_path= data_path+'/input/val04'
prepared_valdata5_path= data_path+'/input/val05'
#prepared_valdata6_path= data_path+'/input/val06'
#prepared_valdata7_path= data_path+'/input/val07'
#prepared_valdata8_path= data_path+'/input/val08'
#prepared_valdata9_path= data_path+'/input/val09'
#prepared_valdata10_path= data_path+'/input/val10'
test_path = data_path+'/test'
prepared_test_path= data_path+'/input/test'

#lnw add. slice file (milleseconds). data.py 파일안에 설정도 변경시 변경필요
sliceTime = 1000
shiftTime = 400

from glob import glob
import random
import os
import numpy as np
#lnw
from datetime import datetime
import wave
import sys


#lnw start time 
start_time = datetime.now()
print("Start time : "+str(start_time))

SEED = 2018
# 리스트를 랜덤하게 셔플하는 함수이다
def random_shuffle(lst):
    random.seed(SEED)
    random.shuffle(lst)
    return lst

#lnw add wave file slice 함수 
def wavSlice(infileName,insaveDir,insliceTime):
   #fileName='test2.wav'
   # slice seconds, 400ms = 400
   #sliceTime = 400
   #print('slice infileName:',infileName)

   wavfile = wave.open(infileName)
   frameRate = wavfile.getframerate()
   # frames per ms
   fpms = frameRate / 1000 
   width = wavfile.getsampwidth()
   # modified 20190330
   #numFrames = wavfile.getnframes()
   numFrames = wavfile.getnframes()*width

   insliceTime = insliceTime*width
   frames = wavfile.readframes(numFrames)
   sliceNumFrames = int(fpms * insliceTime)

   wavfile.close()

   x=0
   while x+sliceNumFrames<=numFrames:
       curFrames= frames[x:x+sliceNumFrames]
       x=x+sliceNumFrames
       infileName=infileName.split('/')[-1]
       writeFilname = insaveDir+'/'+infileName[:-4]+'-'+str(x)+infileName[-4:]
       #print('writeFilename:',writeFilname)
       write = wave.open(writeFilname,'w')
       write.setparams((wavfile.getnchannels(), width, frameRate, sliceNumFrames, wavfile.getcomptype(), wavfile.getcompname()))
       write.writeframes(curFrames)
       write.close()

   #XXXXms shift resave
   x=int(fpms*shiftTime) # fpms : frames per ms
   while x+sliceNumFrames<=numFrames:
       curFrames= frames[x:x+sliceNumFrames]
       x=x+sliceNumFrames
       infileName=infileName.split('/')[-1]
       writeFilname = insaveDir+'/'+infileName[:-4]+'-'+str(x)+infileName[-4:]
       write = wave.open(writeFilname,'w')
       write.setparams((wavfile.getnchannels(), width, frameRate, sliceNumFrames, wavfile.getcomptype(), wavfile.getcompname()))
       write.writeframes(curFrames)
       write.close()


# 텍스트 파일을 저장할 폴더를 생성한다.
if not os.path.exists('input'):
    os.mkdir('input')

# lnw add input slice data dir 생성
if not os.path.exists(data_path+'/input'):
    os.mkdir(data_path+'/input')
else :
    print (data_path+'/input',' exist... Stoped. If you want process, del dir and files.') 
    sys.exit()
    
# lnw add mkae  slice files directory 
if not os.path.exists(prepared_trndata1_path):
    os.mkdir(prepared_trndata1_path)
if not os.path.exists(prepared_trndata2_path):
    os.mkdir(prepared_trndata2_path)
if not os.path.exists(prepared_trndata3_path):
    os.mkdir(prepared_trndata3_path)
if not os.path.exists(prepared_trndata4_path):
    os.mkdir(prepared_trndata4_path)
if not os.path.exists(prepared_trndata5_path):
    os.mkdir(prepared_trndata5_path)
#if not os.path.exists(prepared_trndata6_path):
#    os.mkdir(prepared_trndata6_path)
#if not os.path.exists(prepared_trndata7_path):
#    os.mkdir(prepared_trndata7_path)
#if not os.path.exists(prepared_trndata8_path):
#    os.mkdir(prepared_trndata8_path)
#if not os.path.exists(prepared_trndata9_path):
#    os.mkdir(prepared_trndata9_path)
#if not os.path.exists(prepared_trndata10_path):
#    os.mkdir(prepared_trndata10_path)
if not os.path.exists(prepared_valdata1_path):
    os.mkdir(prepared_valdata1_path)
if not os.path.exists(prepared_valdata2_path):
    os.mkdir(prepared_valdata2_path)
if not os.path.exists(prepared_valdata3_path):
    os.mkdir(prepared_valdata3_path)
if not os.path.exists(prepared_valdata4_path):
    os.mkdir(prepared_valdata4_path)
if not os.path.exists(prepared_valdata5_path):
    os.mkdir(prepared_valdata5_path)
#if not os.path.exists(prepared_valdata6_path):
#    os.mkdir(prepared_valdata6_path)
#if not os.path.exists(prepared_valdata7_path):
#    os.mkdir(prepared_valdata7_path)
#if not os.path.exists(prepared_valdata8_path):
#    os.mkdir(prepared_valdata8_path)
#if not os.path.exists(prepared_valdata9_path):
#    os.mkdir(prepared_valdata9_path)
#if not os.path.exists(prepared_valdata10_path):
#    os.mkdir(prepared_valdata10_path)


# 훈련 데이터 전체 trn_all.txt에 저장한다
trn_all = []
trn_all_file = open('input/trn_all.txt', 'w')
val_all = []
val_all_file = open('input/val_all.txt', 'w')

# 제공된 훈련 데이터 경로를 모두 읽어온다
#files = glob(data_path + '/train/audio/*/*.wav')

#lnw add. read raw train datas
d01TrnFiles = glob(trndata1_path+'/*.wav')
d02TrnFiles = glob(trndata2_path+'/*.wav')
d03TrnFiles = glob(trndata3_path+'/*.wav')
d04TrnFiles = glob(trndata4_path+'/*.wav')
d05TrnFiles = glob(trndata5_path+'/*.wav')
#d06TrnFiles = glob(trndata6_path+'/*.wav')
#d07TrnFiles = glob(trndata7_path+'/*.wav')
#d08TrnFiles = glob(trndata8_path+'/*.wav')
#d09TrnFiles = glob(trndata9_path+'/*.wav')
#d10TrnFiles = glob(trndata10_path+'/*.wav')
d01ValFiles = glob(valdata1_path+'/*.wav')
d02ValFiles = glob(valdata2_path+'/*.wav')
d03ValFiles = glob(valdata3_path+'/*.wav')
d04ValFiles = glob(valdata4_path+'/*.wav')
d05ValFiles = glob(valdata5_path+'/*.wav')
#d06ValFiles = glob(valdata6_path+'/*.wav')
#d07ValFiles = glob(valdata7_path+'/*.wav')
#d08ValFiles = glob(valdata8_path+'/*.wav')
#d09ValFiles = glob(valdata9_path+'/*.wav')
#d10ValFiles = glob(valdata10_path+'/*.wav')

#lnw add. slice files 
for f in d01TrnFiles:
    #fileName = f.split('/')[-1] 
    fileName = f
    #print('d01TrnFile:',fileName)
    saveDir = prepared_trndata1_path 
    wavSlice(fileName,saveDir,sliceTime)
for f in d02TrnFiles:
    #fileName = f.split('/')[-1]
    fileName = f
    #print('d02TrnFile:',fileName)
    saveDir = prepared_trndata2_path
    wavSlice(fileName,saveDir,sliceTime)
for f in d03TrnFiles:
    #fileName = f.split('/')[-1]
    fileName = f
    #print('d03TrnFile:',fileName)
    saveDir = prepared_trndata3_path
    wavSlice(fileName,saveDir,sliceTime)
for f in d04TrnFiles:
    #fileName = f.split('/')[-1]
    fileName = f
    #print('d04TrnFile:',fileName)
    saveDir = prepared_trndata4_path
    wavSlice(fileName,saveDir,sliceTime)
for f in d05TrnFiles:
    #fileName = f.split('/')[-1]
    fileName = f
    #print('d05TrnFile:',fileName)
    saveDir = prepared_trndata5_path
    wavSlice(fileName,saveDir,sliceTime)
'''
for f in d06TrnFiles:
    #fileName = f.split('/')[-1]
    fileName = f
    #print('d06TrnFile:',fileName)
    saveDir = prepared_trndata6_path
    wavSlice(fileName,saveDir,sliceTime)
for f in d07TrnFiles:
    #fileName = f.split('/')[-1]
    fileName = f
    #print('d07TrnFile:',fileName)
    saveDir = prepared_trndata7_path
    wavSlice(fileName,saveDir,sliceTime)
for f in d08TrnFiles:
    #fileName = f.split('/')[-1]
    fileName = f
    #print('d08TrnFile:',fileName)
    saveDir = prepared_trndata8_path
    wavSlice(fileName,saveDir,sliceTime)
for f in d09TrnFiles:
    #fileName = f.split('/')[-1]
    fileName = f
    #print('d09TrnFile:',fileName)
    saveDir = prepared_trndata9_path
    wavSlice(fileName,saveDir,sliceTime)
for f in d10TrnFiles:
    #fileName = f.split('/')[-1]
    fileName = f
    #print('d10TrnFile:',fileName)
    saveDir = prepared_trndata10_path
    wavSlice(fileName,saveDir,sliceTime)
'''

for f in d01ValFiles:
    #fileName = f.split('/')[-1]
    fileName = f
    #print('d01ValFile:',fileName)
    saveDir = prepared_valdata1_path
    wavSlice(fileName,saveDir,sliceTime)
for f in d02ValFiles:
    #fileName = f.split('/')[-1]
    fileName = f
    #print('d02TrnFile:',fileName)
    saveDir = prepared_valdata2_path
    wavSlice(fileName,saveDir,sliceTime)
for f in d03ValFiles:
    #fileName = f.split('/')[-1]
    fileName = f
    #print('d03TrnFile:',fileName)
    saveDir = prepared_valdata3_path
    wavSlice(fileName,saveDir,sliceTime)
for f in d04ValFiles:
    #fileName = f.split('/')[-1]
    fileName = f
    #print('d04TrnFile:',fileName)
    saveDir = prepared_valdata4_path
    wavSlice(fileName,saveDir,sliceTime)
for f in d05ValFiles:
    #fileName = f.split('/')[-1]
    fileName = f
    #print('d05TrnFile:',fileName)
    saveDir = prepared_valdata5_path
    wavSlice(fileName,saveDir,sliceTime)
'''
for f in d06ValFiles:
    #fileName = f.split('/')[-1]
    fileName = f
    #print('d06TrnFile:',fileName)
    saveDir = prepared_valdata6_path
    wavSlice(fileName,saveDir,sliceTime)
for f in d07ValFiles:
    #fileName = f.split('/')[-1]
    fileName = f
    #print('d07TrnFile:',fileName)
    saveDir = prepared_valdata7_path
    wavSlice(fileName,saveDir,sliceTime)
for f in d08ValFiles:
    #fileName = f.split('/')[-1]
    fileName = f
    #print('d08TrnFile:',fileName)
    saveDir = prepared_valdata8_path
    wavSlice(fileName,saveDir,sliceTime)
for f in d09ValFiles:
    #fileName = f.split('/')[-1]
    fileName = f
    #print('d09TrnFile:',fileName)
    saveDir = prepared_valdata9_path
    wavSlice(fileName,saveDir,sliceTime)
for f in d10ValFiles:
    #fileName = f.split('/')[-1]
    fileName = f
    #print('d10TrnFile:',fileName)
    saveDir = prepared_valdata10_path
    wavSlice(fileName,saveDir,sliceTime)
'''

#lnw add. read sliced train and val datas for label file
d01TrnFiles = glob(prepared_trndata1_path+'/*.wav')
d02TrnFiles = glob(prepared_trndata2_path+'/*.wav')
d03TrnFiles = glob(prepared_trndata3_path+'/*.wav')
d04TrnFiles = glob(prepared_trndata4_path+'/*.wav')
d05TrnFiles = glob(prepared_trndata5_path+'/*.wav')
#d06TrnFiles = glob(prepared_trndata6_path+'/*.wav')
#d07TrnFiles = glob(prepared_trndata7_path+'/*.wav')
#d08TrnFiles = glob(prepared_trndata8_path+'/*.wav')
#d09TrnFiles = glob(prepared_trndata9_path+'/*.wav')
#d10TrnFiles = glob(prepared_trndata10_path+'/*.wav')
d01ValFiles = glob(prepared_valdata1_path+'/*.wav')
d02ValFiles = glob(prepared_valdata2_path+'/*.wav')
d03ValFiles = glob(prepared_valdata3_path+'/*.wav')
d04ValFiles = glob(prepared_valdata4_path+'/*.wav')
d05ValFiles = glob(prepared_valdata5_path+'/*.wav')
#d06ValFiles = glob(prepared_valdata6_path+'/*.wav')
#d07ValFiles = glob(prepared_valdata7_path+'/*.wav')
#d08ValFiles = glob(prepared_valdata8_path+'/*.wav')
#d09ValFiles = glob(prepared_valdata9_path+'/*.wav')
#d10ValFiles = glob(prepared_valdata10_path+'/*.wav')

for f in d01TrnFiles:
      #label = 'noise'
      label = labels[0]
      speaker = 'none'
      trn_all.append((label, speaker, f))
      trn_all_file.write('{},{},{}\n'.format(label, speaker, f))
for f in d02TrnFiles:
      label = labels[1]
      speaker = 'none'
      trn_all.append((label, speaker, f))
      trn_all_file.write('{},{},{}\n'.format(label, speaker, f))
for f in d03TrnFiles:
      label = labels[2]
      speaker = 'none'
      trn_all.append((label, speaker, f))
      trn_all_file.write('{},{},{}\n'.format(label, speaker, f))
for f in d04TrnFiles:
      label = labels[3]
      speaker = 'none'
      trn_all.append((label, speaker, f))
      trn_all_file.write('{},{},{}\n'.format(label, speaker, f))
for f in d05TrnFiles:
      label = labels[4]
      speaker = 'none'
      trn_all.append((label, speaker, f))
      trn_all_file.write('{},{},{}\n'.format(label, speaker, f))
'''
for f in d06TrnFiles:
      label = labels[5]
      speaker = 'none'
      trn_all.append((label, speaker, f))
      trn_all_file.write('{},{},{}\n'.format(label, speaker, f))
for f in d07TrnFiles:
      label = labels[6]
      speaker = 'none'
      trn_all.append((label, speaker, f))
      trn_all_file.write('{},{},{}\n'.format(label, speaker, f))
for f in d08TrnFiles:
      label = labels[7]
      speaker = 'none'
      trn_all.append((label, speaker, f))
      trn_all_file.write('{},{},{}\n'.format(label, speaker, f))
for f in d09TrnFiles:
      label = labels[8]
      speaker = 'none'
      trn_all.append((label, speaker, f))
      trn_all_file.write('{},{},{}\n'.format(label, speaker, f))
for f in d10TrnFiles:
      label = labels[9]
      speaker = 'none'
      trn_all.append((label, speaker, f))
      trn_all_file.write('{},{},{}\n'.format(label, speaker, f))
'''
for f in d01ValFiles:
      label = labels[0]
      speaker = 'none'
      val_all.append((label, speaker, f))
      val_all_file.write('{},{},{}\n'.format(label, speaker, f))
for f in d02ValFiles:
      label = labels[1]
      speaker = 'none'
      val_all.append((label, speaker, f))
      val_all_file.write('{},{},{}\n'.format(label, speaker, f))
for f in d03ValFiles:
      label = labels[2]
      speaker = 'none'
      val_all.append((label, speaker, f))
      val_all_file.write('{},{},{}\n'.format(label, speaker, f))
for f in d04ValFiles:
      label = labels[3]
      speaker = 'none'
      val_all.append((label, speaker, f))
      val_all_file.write('{},{},{}\n'.format(label, speaker, f))
for f in d05ValFiles:
      label = labels[4]
      speaker = 'none'
      val_all.append((label, speaker, f))
      val_all_file.write('{},{},{}\n'.format(label, speaker, f))
'''
for f in d06ValFiles:
      label = labels[5]
      speaker = 'none'
      val_all.append((label, speaker, f))
      val_all_file.write('{},{},{}\n'.format(label, speaker, f))
for f in d07ValFiles:
      label = labels[6]
      speaker = 'none'
      val_all.append((label, speaker, f))
      val_all_file.write('{},{},{}\n'.format(label, speaker, f))
for f in d08ValFiles:
      label = labels[7]
      speaker = 'none'
      val_all.append((label, speaker, f))
      val_all_file.write('{},{},{}\n'.format(label, speaker, f))
for f in d09ValFiles:
      label = labels[8]
      speaker = 'none'
      val_all.append((label, speaker, f))
      val_all_file.write('{},{},{}\n'.format(label, speaker, f))
for f in d10ValFiles:
      label = labels[9]
      speaker = 'none'
      val_all.append((label, speaker, f))
      val_all_file.write('{},{},{}\n'.format(label, speaker, f))
'''

trn_all_file.close()
val_all_file.close()


# 훈련 데이터를 화자 기반 9:1 비율로 분리한다 -> lnw 처음부터 훈련용과 검증용 별도 저장함
#uniq_speakers = list(set([speaker for (label, speaker, path) in trn_all]))
#random_shuffle(uniq_speakers)
#cutoff = int(len(uniq_speakers) * 0.9)
#speaker_val = uniq_speakers[cutoff:]

#lnw add random list sort
trn_all=random_shuffle(trn_all)
val_all=random_shuffle(val_all)

# lnw 검증용  별도 준비못해다던지 훈련과 검증용 분리용 필요할 경우 아래 막은 것 열어서 사용
cutoff = int(len(trn_all) * 0.9)
val_all=trn_all[cutoff:]
trn_all=trn_all[:cutoff]

# 교차 검증용 파일을 생성한다
trn_file = open('input/trn.txt', 'w')
val_file = open('input/val.txt', 'w')
#for (label, speaker, path) in trn_all:
#    if speaker not in speaker_val:
#        trn_file.write('{},{},{}\n'.format(label, speaker, path))
#    else:
#        val_file.write('{},{},{}\n'.format(label, speaker, path))
#lnw add
for (label, speaker, path) in trn_all:
    trn_file.write('{},{},{}\n'.format(label, speaker, path))
for (label, speaker, path) in val_all:
    val_file.write('{},{},{}\n'.format(label, speaker, path))

trn_file.close()
val_file.close()

# test data process
#lnw add. read raw train datas
testfiles = glob(test_path+'/*.wav')
if not os.path.exists(prepared_test_path):
    os.mkdir(prepared_test_path)
#lnw add. slice files
for f in testfiles:
    #fileName = f.split('/')[-1]
    fileName = f
    saveDir = prepared_test_path
    wavSlice(fileName,saveDir,sliceTime)
#lnw add. read sliced test data files
testFiles = glob(prepared_test_path+'/*.wav')
# lnw add. ramdom shuffle sort 은 필요없어서 안함
# 테스트 데이터에 대해서도 텍스트 파일을 생성한다
tst_all_file = open('input/tst.txt', 'w')
for f in testFiles:
    tst_all_file.write(',,{}\n'.format(f))
tst_all_file.close()


#lnw end time, duration
end_time = datetime.now()
print("End time : "+str(end_time))
print('Duration: {}'.format(end_time - start_time))

