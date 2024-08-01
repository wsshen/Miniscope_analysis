import csv
import tkinter.filedialog
from tkinter import *
from edgeDetection import *
import os
import matplotlib.pyplot as plt
import numpy as np
from readTabTxtFile import *
from calTempResolution import *
from trimmedTriggerTime import *
import seaborn as sn
import scipy
frameInput = 0
lickTrigNum = 1
pokeTrigNum = 4
caTimeNum = 0
dir = tkinter.filedialog.askdirectory(initialdir="/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/4914002252023/05192023/")
validIdFile = "validIdx_latterhalf.txt"
cellTracesFile = "cellTraces_latterhalf.txt"
pokeTriggerFile = "stim"+str(pokeTrigNum)+".txt"
lickTriggerFile = "stim"+str(lickTrigNum)+".txt"
caTimeFile = "stim"+str(caTimeNum)+".txt"
videoFrameNum = 1000
def findClosestTimeIndex(t,tSeries):
    tDiff = [abs(t-i) for i in tSeries]
    nd_tDiff = np.array(tDiff)
    pos = np.argmin(nd_tDiff)
    return pos
def findClosestLickIndex(t1Series,refT,tSeries):
    tDiff = [[i-refT,i] for i in t1Series if i-refT>=0]
    nd_tDiff = np.array(tDiff)
    lick_pos = np.argmin(nd_tDiff[:,0])
    pos = findClosestTimeIndex(nd_tDiff[lick_pos,1],tSeries)
    return pos

with open(dir+os.sep+cellTracesFile, "r", newline="\n") as readFile:
    data_reader = readFile.readlines()
# with open(dir+os.sep+validIdFile, "r", newline="") as readFile:
#     id_reader = readFile.readlines()
# with open(dir+os.sep+triggerFile, "r", newline="") as readFile:
#     trig_reader = readFile.readlines()
# with open(dir+os.sep+caTimeFile, "r", newline="") as readFile:
#     caTime_reader = readFile.readlines()

caData = readTabTxtFile(dir+os.sep+cellTracesFile, 'float')
valid_idx = readTabTxtFile(dir+os.sep+validIdFile, 'int')
pokeTrigTime = readTabTxtFile(dir+os.sep+pokeTriggerFile, 'float')
lickTrigTime = readTabTxtFile(dir+os.sep+lickTriggerFile, 'float')
caTime = readTabTxtFile(dir+os.sep+caTimeFile, 'float')
caFrameDur = calTempResolution(caTime)

# valid_idx = [int(i) for i in id_reader[0].strip().split("\t")]
# caTime = [float(i) for i in caTime_reader[0].strip().split("\t")]
# trigTime = [float(i) for i in trig_reader[0].strip().split("\t")]
numVideoShift=35
preTrigWindow = -100
postTrigWindow = 100
startTime = 5*60*30
stopTime = (5*60+30)*30
pokeTimeAdj = 0.03

trimmed_pokeTrigTime = trimmedTriggerTime(pokeTrigTime,caTime,startTime+videoFrameNum*numVideoShift,stopTime+videoFrameNum*numVideoShift)
trimmed_lickTrigTime = trimmedTriggerTime(lickTrigTime,caTime,startTime+videoFrameNum*numVideoShift,stopTime+videoFrameNum*numVideoShift)

numPokeTrig = len(trimmed_pokeTrigTime)
numLickTrig = len(trimmed_lickTrigTime)
numData = len(caData)
plotWindow = range(startTime,stopTime,1)


for i in valid_idx:
    print(i)
    ca_trace = [float(k) for k in data_reader[i].strip().split("\t")]
    ca_raw_traces = []
    pokeLine = np.zeros(len(plotWindow))
    lickLine= np.zeros(len(plotWindow))
    plot_ca = ca_trace[startTime:stopTime]

    for trig_id in range(numPokeTrig):
        try:
            poke_pos_temp = findClosestTimeIndex(trimmed_pokeTrigTime[trig_id]-pokeTimeAdj,caTime) - startTime -videoFrameNum * numVideoShift
            pokeLine[poke_pos_temp] =1 * 0.7* np.max(plot_ca)
        except:
            print("Plot window is outside ca2+ data size")
    for trig_id in range(numLickTrig):
        try:
            lick_pos_temp = findClosestTimeIndex(trimmed_lickTrigTime[trig_id], caTime) - startTime - videoFrameNum * numVideoShift
            lickLine[lick_pos_temp] = 1 * 0.7 * np.max(plot_ca)
        except:
            print("Plot window is outside ca2+ data size")


    fig= plt.figure()
    ax = fig.add_subplot(1,1,1)
    tVector = np.linspace(0, stopTime-startTime, len(plotWindow)) * caFrameDur

    ax.plot(tVector,plot_ca,color='b',label='ca2+')
    ax.plot(tVector,lickLine,color='r',label='lick')
    ax.plot(tVector, pokeLine,color='k',label='poke')
    ax.legend()
    print("plotting "+str(i))
    plt.xlabel("Time(s)")
    plt.ylabel("deltaF/F")
    plt.savefig(str(i)+'.eps',format='eps')
    # plt.figure()

#plt.show()

print("done")
input()





