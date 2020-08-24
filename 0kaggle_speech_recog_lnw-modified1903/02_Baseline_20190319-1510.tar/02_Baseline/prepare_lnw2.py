# -*- encoding : utf-8 -*-
#
# data prepare
# lnw modified
# 기존 라벨중 두개를  voice noise 로 변경
# 특정크기만큼 voice noise files 잘라 저장 
# 200ms 만큼 쉬프트후 재저장 
# 리스트 랜덤 순서 정열
#
# usage
#trndata1_path = data_path+'/train/voice' 에 학습할 보이스 파일들 넣어둠 라벨은 이 디렉토리에 있는것들은 모두 같은 지정된 라벨로  파일로 저장함 
#trndata2_path = data_path+'/train/noise' 에 학습할 노이즈 퍼일들 넣어둠
#valdata1_path = data_path+'/val/voice'  에 검증할 보이스 파일들 넣어둠
#valdata2_path = data_path+'/val/noise'  에 덤증할 노이즈 퍼일들 넣어둠
#testt_path = data_path+'/test'   에 라벨없는 테스트할 파일들 넣어둠
# data_path+'/input'가 있으면 모든 프로세스 하고 이미 있으면 재준비안함  재준비하게 하려면  이 디렉토리 및 파일들 모두 삭제 
# python3 pre_pare.py
#


# setting 
# 10개의 label과 데이터 경로를 지정한다
# lnw modified. 두개만  voice noise 로 변경
labels = ['voice', 'noise', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
#lnw modified 
#data_path = '~/.kaggle/competitions/tensorflow-speech-recognition-challenge' 
data_path = '/home/neoadmin/kaggle_speech_recog/competitions_data'
#lnw add
trndata1_path = data_path+'/train/voice'
trndata2_path = data_path+'/train/noise'
valdata1_path = data_path+'/val/voice'
valdata2_path = data_path+'/val/noise'
prepared_trndata1_path= data_path+'/input/trnvoice'
prepared_trndata2_path= data_path+'/input/trnnoise'
prepared_valdata1_path= data_path+'/input/valvoice'
prepared_valdata2_path= data_path+'/input/valnoise'
test_path = data_path+'/test'
prepared_test_path= data_path+'/input/test'
#lnw add. slice file (milleseconds). data.py 파일안에 설정도 변경시 변경필요
sliceTime = 400


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
   numFrames = wavfile.getnframes()
   width = wavfile.getsampwidth()

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

   #200ms shidt resave
   x=int(fpms*200) # fpms : frames per ms
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
    
# lnw add mkae noise slice files directory, voice slice files directory 
if not os.path.exists(prepared_trndata1_path):
    os.mkdir(prepared_trndata1_path)
if not os.path.exists(prepared_trndata2_path):
    os.mkdir(prepared_trndata2_path)
if not os.path.exists(prepared_valdata1_path):
    os.mkdir(prepared_valdata1_path)
if not os.path.exists(prepared_valdata2_path):
    os.mkdir(prepared_valdata2_path)

# 훈련 데이터 전체 trn_all.txt에 저장한다
trn_all = []
trn_all_file = open('input/trn_all.txt', 'w')
val_all = []
val_all_file = open('input/val_all.txt', 'w')

# 제공된 훈련 데이터 경로를 모두 읽어온다
#files = glob(data_path + '/train/audio/*/*.wav')

#lnw add. read raw train datas
voiceTrnFiles = glob(trndata1_path+'/*.wav')
noiseTrnFiles = glob(trndata2_path+'/*.wav')
voiceValFiles = glob(valdata1_path+'/*.wav')
noiseValFiles = glob(valdata2_path+'/*.wav')

#lnw add. slice files 
for f in voiceTrnFiles:
    #fileName = f.split('/')[-1] 
    fileName = f
    #print('voiceTrnFile:',fileName)
    saveDir = prepared_trndata1_path 
    wavSlice(fileName,saveDir,sliceTime)
for f in noiseTrnFiles:
    #fileName = f.split('/')[-1]
    fileName = f
    #print('noiseTrnFile:',fileName)
    saveDir = prepared_trndata2_path
    wavSlice(fileName,saveDir,sliceTime)
for f in voiceValFiles:
    #fileName = f.split('/')[-1]
    fileName = f
    #print('voiceValFile:',fileName)
    saveDir = prepared_valdata1_path
    wavSlice(fileName,saveDir,sliceTime)
for f in noiseValFiles:
    #fileName = f.split('/')[-1]
    fileName = f
    #print('noiseTrnFile:',fileName)
    saveDir = prepared_valdata2_path
    wavSlice(fileName,saveDir,sliceTime)

#lnw add. read sliced train and val datas for label file
voiceTrnFiles = glob(prepared_trndata1_path+'/*.wav')
noiseTrnFiles = glob(prepared_trndata2_path+'/*.wav')
voiceValFiles = glob(prepared_valdata1_path+'/*.wav')
noiseValFiles = glob(prepared_valdata2_path+'/*.wav')

for f in voiceTrnFiles:
      label = 'voice'
      speaker = 'none'
      trn_all.append((label, speaker, f))
      trn_all_file.write('{},{},{}\n'.format(label, speaker, f))
for f in noiseTrnFiles:
      label = 'noise'
      speaker = 'none'
      trn_all.append((label, speaker, f))
      trn_all_file.write('{},{},{}\n'.format(label, speaker, f))
for f in voiceValFiles:
      label = 'voice'
      speaker = 'none'
      val_all.append((label, speaker, f))
      val_all_file.write('{},{},{}\n'.format(label, speaker, f))
for f in noiseValFiles:
      label = 'noise'
      speaker = 'none'
      val_all.append((label, speaker, f))
      val_all_file.write('{},{},{}\n'.format(label, speaker, f))

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

