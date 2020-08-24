# -*- encoding : utf-8 -*-
#
# test data check program
# Liu Nae Won  2019.2.15
#
# usage :
# python3 test_check.py test_result.csv test_correct.csv
#
# test_result.scv :
# filename,predict_value
# 
# test_correct.csv :
# filenaem,correct_value
#


import sys



if len(sys.argv) == 3 :
     test_file = sys.argv[1]
     correct_file = sys.argv[2]
     #print(test_file,correct_file)
else :
     print("usage: python3 test_check.py test_result.csv test_correct.csv")
     sys.exit()


testfile = open(test_file,'r')
correctfile = open(correct_file,'r')


testlines = testfile.readlines()
correctlines = correctfile.readlines()

total = 0
correct = 0

for tl in testlines :
     tl_label = tl[tl.find(',')+1:]
     #print(tl_label)
     for cl in correctlines :
         indexNo = cl.rfind(tl)  
         if indexNo != -1 :
             cl_label = cl[cl.find(',')+1:] 
             #print(cl_label) 
             if tl_label == cl_label : 
                    correct = correct + 1
     total = total + 1

print("total:"+str(total)+" correct:"+str(correct)+" rate:"+str(correct/total))


