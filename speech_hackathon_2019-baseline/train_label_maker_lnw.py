#-*- coding: utf-8 -*-

"""
make train_label
for speech_hackathon_2019-baseline datasets

Liu Nae win
2019.9.3

use files :  hackathon.labels , label_loader.py


"""

import label_loader
import os
import glob



char2index = dict()
index2char = dict()
SOS_token = 0
EOS_token = 0
PAD_token = 0


outlist = []
notlabel_list = {}

#### read labels
char2index, index2char = label_loader.load_label('./hackathon.labels')
SOS_token = char2index['<s>']
EOS_token = char2index['</s>']
PAD_token = char2index['_']

#print(char2index)    # {'웰': 234, '와': 294, ... }
#print(index2char)   # {0: '_', 1: '군', 2: '철', ... }


#### get files list 
path = "./sample_dataset/train/train_data/*.txt"
#file_list = os.listdir(path)
file_list = glob.glob(path)
#file_list_txt = [file for file in file_list if file.endswith(".txt")]
#print (file_list)

#### read file
for fi in file_list :
   rf = open(fi,'rt',encoding='cp949')
   rline = rf.read()
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
   rline = rline.replace("늫","늦")
   rline = rline.replace("똠","돔")
   rline = rline.replace("맻","맺")
   rline = rline.replace("몀","면")
   rline = rline.replace("띃","뜻")
   rline = rline.replace("켔","켓")
   rline = rline.replace("왤","욀")
   rline = rline.replace("뽂","뽁")
   rline = rline.replace("썻","썼")
   rline = rline.replace("줐","줏")
   rline = rline.replace("냬","내")
   rline = rline.replace("떄","때")
   rline = rline.replace("줜","전")
   rline = rline.replace("헀","했")
   rline = rline.replace("깄","깃")
   rline = rline.replace("밲","백")
   rline = rline.replace("핬","핫")
   rline = rline.replace("쩰","쨀")


      
   #print(rline)
   rlineout = ","
   for rli  in rline:
      try : 
         resc2i = char2index[rli]
      except KeyError : 
         #resc2i = " "
         resc2i = "662"   # 662 = " "
         notlabel_list[rli]=0
      #print("rli resc2i",rli,resc2i)
      rlineout = rlineout+str(resc2i)+" "
      #print('rlineout:',rlineout)
   outtxt = os.path.basename(fi[:-4])+rlineout
   outlist.append(outtxt) 
   rf.close()


#### redult print and write
#print("outlist:",outlist)
print("notlabel_list:",notlabel_list)

fw = open("./sample_dataset/train/train_label_out","w")
for ol in outlist:
    fw.write(ol+"\n")
fw.close()

fw = open("./sample_dataset/train/train_data/data_list_out.csv","w")
for fi2 in file_list:
    ol2 = os.path.basename(fi2[:-4])+".wav,"+os.path.basename(fi2[:-4])+".label"
    fw.write(ol2+"\n")
fw.close()



