#-*- coding:utf-8 -*-
'''

json file maker for Jasper

2019.9.11 Liu Nae Won


'''

import librosa
import os
import glob

sr = 16000   # wav sample rate

#### get files list 
path = "/home/neoadmin/stt/aihack_speech/baseline/sample_dataset/train/train_data/*.wav"
jsonpath= "/home/neoadmin/stt/aihack_speech/baseline/sample_dataset/train/train-wav.json"
file_list = glob.glob(path)
#print ('file_list:',file_list)

#### json file first line write
jsontxt ="["+"\n"
jfw = open(jsonpath,"w")
jfw.write(jsontxt)
jfw.close()

#### read file
for fname in file_list :

   #print("fname:",fname)
   y, sr = librosa.load(fname,sr=sr)
   duration = librosa.get_duration(y=y, sr=sr)
   num_samples = int(duration*sr)
   #print (fname,duration,num_samples)

   ftranfile=fname.split('.')[0]+".txt"
   #ftran=open(ftranfile)
   ftran=open(ftranfile,'rt',encoding='cp949')
   rline = ftran.readline()
   
   # ETRI (철자전사)/(발음전사)로 두번 적힌것 발음전사만 남기기 
   ts = 0
   while ts > -1 : 
      ts = rline.find(")/")
      if ts > -1 :
         ts2 = rline[:ts].find("(") 
         rline = rline[:ts2]+rline[ts+2:]
   # Etri 전사규칙 특수 기호 없애기
   rline = rline.replace("b/"," ")
   rline = rline.replace("l/"," ")
   rline = rline.replace("o/"," ")
   rline = rline.replace("n/"," ")
   rline = rline.replace("/","")
   rline = rline.replace("(","")
   rline = rline.replace(")","") 
   rline = rline.replace("*","")
   rline = rline.replace("+","") 
   # Etri 전사규칙에 영어는 한글발음으로 처리하나 KBS처럼 고유명사는 그대로 표기해서 영문소문자는 대문자로 변경
   rline = rline.replace("a","A")
   rline = rline.replace("b","B")
   rline = rline.replace("c","C")
   rline = rline.replace("d","D")
   rline = rline.replace("e","E")
   rline = rline.replace("f","F")
   rline = rline.replace("g","G") 
   rline = rline.replace("h","H")
   rline = rline.replace("i","I")    
   rline = rline.replace("j","J")
   rline = rline.replace("k","K")
   rline = rline.replace("l","L")
   rline = rline.replace("m","M")
   rline = rline.replace("n","N")
   rline = rline.replace("o","O")
   rline = rline.replace("p","P") 
   rline = rline.replace("q","Q")
   rline = rline.replace("r","R")    
   rline = rline.replace("s","S")
   rline = rline.replace("t","T")
   rline = rline.replace("u","U")
   rline = rline.replace("v","V")
   rline = rline.replace("w","W")
   rline = rline.replace("x","X")
   rline = rline.replace("y","Y") 
   rline = rline.replace("z","Z")
   
   # 필요없을 것 같은것 처리
   rline = rline.replace(",","") 
   rline = rline.replace("\n"," ")
           
   # KS C 5601 2350자에 없는 것은 비슷한 것으로 대체 처리 
   rline = rline.replace("샾","샵")
   rline = rline.replace("앚","앗")
   rline = rline.replace("띡","딕")
   rline = rline.replace("딲","딱")
   rline = rline.replace("됬","됫")
   rline = rline.replace("릏","를")
   
   transcript = rline
   ftran.close()

   fname=fname.split('/')[-1]
   jsontxt = '  { "files": [ { "sample_rate": 16000.0, "encoding": "Signed Integer PCM", "fname": "train_data/' + fname + '", "silent": false, "channels": 1, "num_samples": ' + str(num_samples) + ', "bitrate": 16, "speed": 1, "duration": ' + str(duration) + ' } ], "original_duration": ' + str(duration) + ', "transcript": " ' + transcript + ' ", "original_num_samples": ' + str(num_samples) + ' },' +'\n'
   #print ('jsontxt:',jsontxt)
   jfw = open(jsonpath,"a")
   jfw.write(jsontxt)
   jfw.close()


#### json file last line write
jsontxt ="]"+"\n"
jfw = open(jsonpath,"a")
jfw.write(jsontxt)
jfw.close()

