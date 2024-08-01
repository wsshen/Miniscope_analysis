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
import pickle
frameInput = 0
lickTrigNum = 1
pokeTrigNum = 4
caTimeNum = 0
dir = tkinter.filedialog.askdirectory(initialdir="/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/4914202252023/05192023/")
validIdFile = "validIdx_all.txt"
cellTracesFile = "all_traces_union.pickle"
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

with open(cellTracesFile, "rb") as f:
    caData = pickle.load(f)
# with open(dir+os.sep+validIdFile, "r", newline="") as readFile:
#     id_reader = readFile.readlines()
# with open(dir+os.sep+triggerFile, "r", newline="") as readFile:
#     trig_reader = readFile.readlines()
# with open(dir+os.sep+caTimeFile, "r", newline="") as readFile:
#     caTime_reader = readFile.readlines()

# caData = readTabTxtFile(dir+os.sep+cellTracesFile, 'float')
pokeTrigTime = readTabTxtFile(dir+os.sep+pokeTriggerFile, 'float')
lickTrigTime = readTabTxtFile(dir+os.sep+lickTriggerFile, 'float')
caTime = readTabTxtFile(dir+os.sep+caTimeFile, 'float')
caFrameDur = calTempResolution(caTime)

# valid_idx = [int(i) for i in id_reader[0].strip().split("\t")]
# caTime = [float(i) for i in caTime_reader[0].strip().split("\t")]
# trigTime = [float(i) for i in trig_reader[0].strip().split("\t")]
numVideoShift = 0
preTrigWindow = -100
postTrigWindow = 100
trimmed_pokeTrigTime = trimmedTriggerTime(pokeTrigTime,caTime,videoFrameNum*numVideoShift,-1)
trimmed_lickTrigTime = trimmedTriggerTime(lickTrigTime,caTime,videoFrameNum*numVideoShift,-1)

numTrig = len(trimmed_pokeTrigTime)
numData = len(caData)
numAnalysisWindow = -1*preTrigWindow+postTrigWindow
pokeTimeAdj = 0.03


for i in range(caData.shape[0]):
    print(i)
    ca_trace = caData[i,:]
    ca_raw_traces = []
    # ca_raw_overlay_lines = []
    plt.figure()

    for trig_id in range(numTrig):
        try:
            ca_pos_temp = findClosestTimeIndex(trimmed_pokeTrigTime[trig_id]-pokeTimeAdj,caTime)
            ca_pos_temp_lick = findClosestLickIndex(trimmed_lickTrigTime,trimmed_pokeTrigTime[trig_id]-pokeTimeAdj,caTime)
            ca_pos = ca_pos_temp - videoFrameNum * numVideoShift
            ca_pos_lick = ca_pos_temp_lick - videoFrameNum * numVideoShift
            # overlay_temp =np.zeros(-1*preTrigWindow+postTrigWindow)
            # overlay_temp[100]=0
            # overlay_temp[100+ca_pos_lick-ca_pos]=0
            plt.plot((100,100),(trig_id,trig_id+1),scaley=False,color='k')
            plt.plot((100+ca_pos-ca_pos_lick,100+ca_pos-ca_pos_lick),(trig_id,trig_id+1),scaley=False,color='k')
            plot_ca = ca_trace[ca_pos_lick + preTrigWindow : ca_pos_lick + postTrigWindow]
            ca_raw_traces.append(plot_ca)
            # ca_raw_overlay_lines.append(overlay_temp)

        except:
            print(trig_id)
            print("Analysis window is outside ca2+ data size")
    tVector = np.linspace(preTrigWindow, postTrigWindow, numAnalysisWindow) * caFrameDur
    xLabel = np.linspace(preTrigWindow, postTrigWindow, numAnalysisWindow)
    xLabel[:] = np.nan
    xSegments = np.linspace(preTrigWindow,postTrigWindow,9)
    xSegments_integer = [int(i+(-1)*preTrigWindow) for i in xSegments]
    xSegments_integer[-1]=-1
    xLabel[xSegments_integer]=xSegments*caFrameDur
    ax = sn.heatmap(ca_raw_traces, cmap="coolwarm",xticklabels=xLabel)
    ax.set_title("component_"+str(i))
    plt.savefig("heatmap"+str(i)+'.eps',format='eps')

    fig= plt.figure()
    ax = fig.add_subplot(1,1,1)
    tVector = np.linspace(preTrigWindow,postTrigWindow,numAnalysisWindow)*caFrameDur
    ca_mean_traces = np.mean(ca_raw_traces,axis=0)
    sem = scipy.stats.sem(ca_raw_traces,axis=0)
    ccc = [[tVector[-1 - i], ca_mean_traces[-1 - i] - sem[-1 - i]] for i in range(len(tVector))]
    ccc = ccc + [[tVector[i], ca_mean_traces[i] + sem[i]] for i in range(len(tVector))]
    pp = plt.Polygon(ccc)

    ax.plot(tVector,ca_mean_traces,color='r')
    ax.plot(np.zeros(10),np.linspace(np.min(ca_mean_traces),np.max(ca_mean_traces),10),color='k')
    ax.add_patch(pp)
    print("plotting "+str(i))
    plt.xlabel("Time(s)")
    plt.ylabel("deltaF/F")
    ax.set_title("component_"+str(i))
    plt.savefig(str(i)+'.eps',format='eps')
    # plt.figure()

#plt.show()

print("done")
input()





