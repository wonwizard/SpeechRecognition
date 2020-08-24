#-*- coding: utf-8 -*-

"""
scripts print write
for speech_hackathon_2019-baseline datasets

Liu Nae win
2019.9.11

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


#### read labels
char2index, index2char = label_loader.load_label('./hackathon.labels')
SOS_token = char2index['<s>']
EOS_token = char2index['</s>']
PAD_token = char2index['_']

#print(char2index)    # {'웰': 234, '와': 294, ... }
#print(index2char)   # {0: '_', 1: '군', 2: '철', ... }


#### get files list 
fname = "./sample_dataset/train/train_label0"
outdir ="./sample_dataset/train/train_data/"

#### read file
#f = open(fname,'rt',encoding='cp949')
f = open(fname)


#### read lines
while True:
   rline = f.readline()
   if not rline: break
   print("read line:",rline)
   fs = rline.split(',')
   flist = fs[0]
   fscripts = fs[1]
   fscriptslist = fscripts.split(' ')
   rlineout = ""
   for fsci in fscriptslist:
      fsci2c = ""
      try : 
         fsci2c = index2char[int(fsci)]
      except : 
         print ("error ", fsci)
      rlineout = rlineout+str(fsci2c)
   outtxt = flist+","+rlineout
   print("outtxt:",outtxt)

   #### file write 
   fw = open (outdir+flist+".txt","w")
   fw.write(rlineout)
   fw.close()

f.close()




