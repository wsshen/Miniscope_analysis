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
from caiman.source_extraction.cnmf import load_CNMF
dir = tkinter.filedialog.askdirectory(initialdir="/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/4914202252023/06072023/13_04_23")
cnmf_save = load_CNMF(dir + os.sep+'cnmf_save_0_39.hdf5')
cellTraces = cnmf_save.estimates.C
valid_idx = cnmf_save.estimates.idx_components
import csv

traces_sz = cellTraces.shape
with open(dir + os.sep + "cellTraces.txt", 'w', newline="\n") as f:
    f_writer = csv.writer(f, delimiter='\t')
    for i in range(traces_sz[0]):
        f_writer.writerow(cellTraces[i, :])
with open(dir+os.sep+"validIdx.txt", 'w', newline="\n") as f:
    f_writer = csv.writer(f, delimiter='\t')
    f_writer.writerow(valid_idx)


frameInput = 0
lickTrigNum = 1
pokeTrigNum = 2
caTimeNum = 0

validIdFile = "validIdx.txt"
cellTracesFile = "cellTraces.txt"
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

def findAllActIndices(t1Series,refT,tSeries,window):
    pos=[]
    tDiff = [[i - refT, i] for i in t1Series if i-refT>0 and i-refT-window<0]
    nd_tDiff = np.array(tDiff)
    for i in nd_tDiff:
        pos_temp = findClosestTimeIndex(i[1],tSeries)
        pos.append(pos_temp)
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
numVideoShift = 0
startTime = 255
timeWindow = 20
trimmed_pokeTrigTime = trimmedTriggerTime(pokeTrigTime,caTime,videoFrameNum*numVideoShift,-1)
trimmed_lickTrigTime = trimmedTriggerTime(lickTrigTime,caTime,videoFrameNum*numVideoShift,-1)

numTrig = len(trimmed_pokeTrigTime)
numData = len(caData)
pokeTimeAdj = 0.03

ca_pos_temp_allLicks = findAllActIndices(trimmed_lickTrigTime, startTime, caTime, timeWindow)
ca_pos_temp_allPokes = findAllActIndices(trimmed_pokeTrigTime, startTime, caTime, timeWindow)

startInd = findClosestTimeIndex(startTime, caTime)
endInd = findClosestTimeIndex(startTime+timeWindow, caTime)
plt.figure()
tVector = np.linspace(0, endInd-startInd, endInd-startInd) * caFrameDur
lickVector = np.zeros(len(tVector))
lickVector[ca_pos_temp_allLicks-np.tile(startInd,len(ca_pos_temp_allLicks))]=1*50
pokeVector = np.zeros(len(tVector))
pokeVector[ca_pos_temp_allPokes-np.tile(startInd,len(ca_pos_temp_allPokes))]=1*50
ca_raw_traces = []
k=1
for i in valid_idx:
    print(i)
    plt.subplot(len(valid_idx),1,k)
    ca_trace = [float(k) for k in data_reader[i].strip().split("\t")]
    plot_ca = ca_trace[startInd:endInd]
    k=k+1
    plt.plot(tVector,plot_ca)
    plt.plot(tVector,lickVector,color='r',linewidth=1)
    plt.plot(tVector, pokeVector, color='b', linewidth=1)

    ca_raw_traces.append(plot_ca)
plt.savefig('rasterPlot_2.eps',format='eps')
print("done")

fig=plt.figure()
ax = fig.add_subplot(1, 1, 1)
ca_mean_traces = np.mean(ca_raw_traces, axis=0)
# sem = scipy.stats.sem(ca_raw_traces, axis=0)
# ccc = [[tVector[-1 - i], ca_mean_traces[-1 - i] - sem[-1 - i]] for i in range(len(tVector))]
# ccc = ccc + [[tVector[i], ca_mean_traces[i] + sem[i]] for i in range(len(tVector))]
# pp = plt.Polygon(ccc)

ax.plot(tVector, ca_mean_traces, color='r')
# ax.add_patch(pp)
ax.plot(tVector, lickVector, color='r', linewidth=1)
ax.plot(tVector, pokeVector, color='b', linewidth=1)


print("plotting " + str(i))
plt.xlabel("Time(s)")
plt.ylabel("deltaF/F")
ax.set_title("component_" + str(i))
plt.savefig('averagePlot_2.eps', format='eps')




