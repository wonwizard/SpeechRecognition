#-*- coding: utf-8 -*-

"""
pcm to wav
for speech_hackathon_2019-baseline datasets

Liu Nae win
2019.9.3

ffmpeg -f s16le -ar 16.0k -ac 1 -i file.pcm file.wav

"""

from glob import glob
import os

pcmfiles = glob('*.pcm')
for f in pcmfiles:
   cmd = 'ffmpeg -f s16le -ar 16.0k -ac 1 -i "'+f+'" "'+f.split('.')[-2]+'.wav"'
   print (cmd)
   os.system(cmd)


