import csv
import tkinter.filedialog
from tkinter import *
from edgeDetection import *
import os
import matplotlib.pyplot as plt
import numpy as np
from readTabTxtFile import *
import re
filePath = "/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/4914202252023/06072023/12_52_03"

miniscopeFile = filePath + os.sep + 'My_V4_Miniscope' + os.sep + 'timeStamps.csv'
caTimeNum = 0
caTimeFile = "stim"+str(caTimeNum)+".txt"
caTime = readTabTxtFile(filePath+os.sep+caTimeFile, 'float')


def returnCa2DataLengthFromFile(dataFile):
    caData = readTabTxtFile(dataFile, 'float')
    return len(caData)

totLen = 0
for files in os.listdir(filePath):
    if re.search('cellTraces*',files):
        print(files)
        temp_len = returnCa2DataLengthFromFile(filePath + os.sep+ files)
        print('Length of data in ' + files + ' is:'+str(temp_len))
        totLen = totLen + temp_len
with open(miniscopeFile, "r", newline="") as readFile:
    file_reader = csv.reader(readFile, delimiter=',')
    csvData = []
    for entry in file_reader:
        try:
            if entry[0].isnumeric():
                csvData.append(entry)
        except:
            print("Skip a line")
print('The length of the csv file is:'+str(len(csvData)))
print('The length of miniscope data is:' + str(totLen))
print('The length of Ca2+ time trigger is:' + str(len(caTime)))


