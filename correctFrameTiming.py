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
from scipy import optimize
import copy
def segments_fit(X, Y, count, xanchors=slice(None), yanchors=slice(None)):
    xmin = X.min()
    xmax = X.max()
    seg = np.full(count - 1, (xmax - xmin) / count)

    px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
    py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 0.01].mean() for x in px_init])

    def func(p):
        seg = p[:count - 1]
        py = p[count - 1:]
        px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
        py = py[yanchors]
        px = px[xanchors]
        return px, py

    def err(p):
        px, py = func(p)
        Y2 = np.interp(X, px, py)
        return np.mean((Y - Y2)**2)

    r = optimize.minimize(err, x0=np.r_[seg, py_init], method='Nelder-Mead',bounds=((xmin,xmax),(xmin,xmax),(None,None),(None,None),(None,None),(None,None)))
    return func(r.x)
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
# plt.figure()
intervals_csv = [int(csvData[i+1][1]) - int(csvData[i][1]) for i in range(len(csvData)-1)]
# plt.subplot(2,1,1)
# plt.plot(intervals_csv)
intervals_csv_copy = copy.deepcopy(intervals_csv)
abnormalInd = [i for i in range(len(intervals_csv_copy)) if intervals_csv_copy[i]>50 or intervals_csv_copy[i]<20]
for i in reversed(abnormalInd):
    intervals_csv_copy.pop(i)
# plt.subplot(2,1,2)
# plt.plot(intervals_csv_copy)
#
# plt.show(block=True)

intervals_caTime = [caTime[i+1] - caTime[i] for i in range(len(caTime)-1)]
timePerFrame = np.mean(intervals_caTime)

predictedClock = [timePerFrame*1000*(i-1) for i in range(len(csvData))]
sysClock = [int(csvData[i][1]) for i in range(len(csvData))]

bbb = np.array([sysClock[i] - predictedClock[i] for i in range(len(sysClock))])
aaa = np.array([i for i in range(len(sysClock)) ])

fx,fy=segments_fit(aaa,bbb,3,yanchors=[1,1,3,3])
plt.figure()
plt.plot(aaa,bbb)
plt.plot(fx,fy,'o')
plt.show(block=True)
input()