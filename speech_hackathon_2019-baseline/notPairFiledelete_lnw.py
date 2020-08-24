#-*- coding:utf-8 -*-
'''
같은 이름의 다른 확장자 (짝이 맞지 않은 것) 가진 파일 삭제 
txt 파일은 있으나 같은 이름의 wav 파일은 없는 것 삭제
 2019.9.24 Neoconvergence, Liu Nae Won

'''

import os
import glob

#os.listdir을 사용했을 경우 파일명만 가져오지만 glob.glob 사용한경우 디렉토리명까지 모두 가져온다.특정 확장자만 가져오기 가능, walk 는 하위 디렉토리까지 모두 가져온다
#os.listdir("./dir") ['a.txt','b.wav'] ,  glob.glob("./dir/*.txt")  ['./dir/a.txt']
#os.path.dirname(name), os.path.basename(name) ('c:\\temp\\python', 'data.txt') 폴더, 파일을 보여준다. 파일이 없거나.. 형식에 맞지 않으면 아무것도 안나온다.
#os.path.splitext(name) ('c:\\temp\\python\\data', '.txt') 확장자만 따로 떨어뜨린다.
#os.path.exists("c:\python32\python.exe')  True  폴더 또는 파일 존재 여부 확인


# process
txtFiles = glob.glob('./sample_dataset/train/train_data/*.txt')

for txtFileName in txtFiles:
    wavFileName = os.path.splitext(txtFileName)[0]+'.wav'
    #print(txtFileName)
    #print(wavFileName)
    if not os.path.exists(wavFileName):
        os.remove(txtFileName) 

# check
txtFiles = glob.glob('./sample_dataset/train/train_data/*.txt')
wavFiles = glob.glob('./sample_dataset/train/train_data/*.wav')

print("txt files length:",len(txtFiles),"wav files length:",len(wavFiles))

