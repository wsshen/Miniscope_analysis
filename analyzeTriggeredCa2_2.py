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
from caiman.source_extraction.cnmf.cnmf import load_CNMF
dir = tkinter.filedialog.askdirectory(initialdir="/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/4914202252023/06092023/11_37_38")
with open(dir+os.sep+'traces_registered_2.pickle', 'rb') as f:
    cellTraces =pickle.load(f)
with open(dir+os.sep+'idx_registered_2.pickle', 'rb') as f:
    valid_idx =pickle.load(f)

import csv

traces_sz = cellTraces.shape
validIdFile = "validIdx_2.txt"
cellTracesFile = "cellTraces_2.txt"
with open(dir + os.sep + cellTracesFile, 'w', newline="\n") as f:
    f_writer = csv.writer(f, delimiter='\t')
    for i in range(traces_sz[0]):
        f_writer.writerow(cellTraces[i, :])
with open(dir+os.sep+validIdFile, 'w', newline="\n") as f:
    f_writer = csv.writer(f, delimiter='\t')
    f_writer.writerow(valid_idx)


frameInput = 0
lickTrigNum = 1
pokeTrigNum = 2
caTimeNum = 0
pumpTrigNum = 4
labviewFile = 'LabVIEWStims_06_08_2023_13_20_10.txt'

pokeTriggerFile = "stim"+str(pokeTrigNum)+".txt"
lickTriggerFile = "stim"+str(lickTrigNum)+".txt"
pumpTriggerFile = "stim"+str(pumpTrigNum)+".txt"
caTimeFile = "stim"+str(caTimeNum)+".txt"
plotSavingFolder = 'Plots'
videoFrameNum = 1000
def findClosestTimeIndex(t,tSeries):
    tDiff = [abs(t-i) for i in tSeries]
    nd_tDiff = np.array(tDiff)
    pos = np.argmin(nd_tDiff)
    return pos
def findAfterClosestIndex(t1Series,refT,tSeries):
    tDiff = [[i-refT,i] for i in t1Series if i-refT>=0]
    nd_tDiff = np.array(tDiff)
    lick_pos = np.argmin(nd_tDiff[:,0])
    pos = findClosestTimeIndex(nd_tDiff[lick_pos,1],tSeries)
    return pos

def findBeforeClosestIndex(t1Series,refT,tSeries):
    tDiff = [[i-refT,i] for i in t1Series if i-refT<=0]
    nd_tDiff = np.array(tDiff)
    lick_pos = np.argmax(nd_tDiff[:,0])
    pos = findClosestTimeIndex(nd_tDiff[lick_pos,1],tSeries)
    return pos

def findAllLickIndices(t1Series,refT,tSeries,window):
    pos=[]
    tDiff = [[i - tSeries[refT], i] for i in t1Series if i-tSeries[refT]>0 and i-tSeries[refT]-window*caFrameDur<0]
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
valid_idx = readTabTxtFile(dir+os.sep+validIdFile, 'int_float')
pokeTrigTime = readTabTxtFile(dir+os.sep+pokeTriggerFile, 'float')
lickTrigTime = readTabTxtFile(dir+os.sep+lickTriggerFile, 'float')
caTime = readTabTxtFile(dir+os.sep+caTimeFile, 'float')
caFrameDur = calTempResolution(caTime)
pumpTrigTime = readTabTxtFile(dir+os.sep+pumpTriggerFile, 'float')
pumpDelaysTime = readTabTxtFile(dir+os.sep+labviewFile, 'int')

# valid_idx = [int(i) for i in id_reader[0].strip().split("\t")]
# caTime = [float(i) for i in caTime_reader[0].strip().split("\t")]
# trigTime = [float(i) for i in trig_reader[0].strip().split("\t")]
numVideoShift = 0
preTrigWindow = -100
postTrigWindow = 200
trimmed_pokeTrigTime = trimmedTriggerTime(pokeTrigTime,caTime,videoFrameNum*numVideoShift,-1)
trimmed_lickTrigTime = trimmedTriggerTime(lickTrigTime,caTime,videoFrameNum*numVideoShift,-1)
trimmed_pumpTrigTime = trimmedTriggerTime(pumpTrigTime,caTime,videoFrameNum*numVideoShift,-1)

# numTrig = len(trimmed_pokeTrigTime)
numTrig = len(trimmed_pumpTrigTime)
numData = len(caData)
numAnalysisWindow = -1*preTrigWindow+postTrigWindow
pokeTimeAdj = 0.03

for i in range(len(valid_idx)):
    print(i)
    ca_trace = [float(k) for k in data_reader[i].strip().split("\t")]
    ca_raw_traces = []
    pumpDelays = []
    # ca_raw_overlay_lines = []
    plt.figure()
    k=0
    pump_trigger_history = []
    for trig_id in range(numTrig):
        try:

            # ca_pos_temp = findClosestTimeIndex(trimmed_pokeTrigTime[trig_id]-pokeTimeAdj,caTime)
            # ca_pos_temp_lick = findClosestLickIndex(trimmed_lickTrigTime,trimmed_pokeTrigTime[trig_id]-pokeTimeAdj,caTime)
            # ca_pos_temp_pump = findClosestLickIndex(trimmed_pumpTrigTime,trimmed_pokeTrigTime[trig_id]-pokeTimeAdj,caTime)
            #
            # ca_pos = ca_pos_temp - videoFrameNum * numVideoShift
            # ca_pos_lick = ca_pos_temp_lick - videoFrameNum * numVideoShift
            # ca_pos_pump = ca_pos_temp_pump - videoFrameNum * numVideoShift

            # ca_pos_temp_allLicks = findAllLickIndices(trimmed_lickTrigTime,ca_pos_temp_lick,caTime,postTrigWindow)
            # ca_pos_allLicks = ca_pos_temp_allLicks - np.tile(videoFrameNum * numVideoShift,(1,len(ca_pos_temp_allLicks)))
            # # overlay_temp =np.zeros(-1*preTrigWindow+postTrigWindow)
            # # overlay_temp[100]=0
            # # overlay_temp[100+ca_pos_lick-ca_pos]=0
            # plt.plot((100,100),(trig_id,trig_id+1),scaley=False,color='k')
            # plt.plot((100+ca_pos-ca_pos_lick,100+ca_pos-ca_pos_lick),(trig_id,trig_id+1),scaley=False,color='k')
            # plt.plot((100+ca_pos_pump-ca_pos,100+ca_pos_pump-ca_pos),(trig_id,trig_id+1),scaley=False,color='r')
            #
            # for lick_ind in ca_pos_allLicks:
            #     plt.plot((100 + lick_ind - ca_pos_lick, 100 + lick_ind - ca_pos_lick), (trig_id, trig_id + 1), scaley=False, color='g',linewidth=1)
            #
            # plot_ca = ca_trace[ca_pos_lick + preTrigWindow : ca_pos_lick + postTrigWindow]
            ca_pos_temp_pump = findClosestTimeIndex(trimmed_pumpTrigTime[trig_id],caTime)
            ca_pos_temp = findBeforeClosestIndex(trimmed_pokeTrigTime,trimmed_pumpTrigTime[trig_id],caTime)
            ca_pos_temp_lick = findAfterClosestIndex(trimmed_lickTrigTime,caTime[ca_pos_temp],caTime)


            ca_pos = ca_pos_temp - videoFrameNum * numVideoShift
            ca_pos_lick = ca_pos_temp_lick - videoFrameNum * numVideoShift
            ca_pos_pump = ca_pos_temp_pump - videoFrameNum * numVideoShift

            ca_pos_temp_allLicks = findAllLickIndices(trimmed_lickTrigTime,ca_pos_temp_lick,caTime,postTrigWindow)
            ca_pos_allLicks = ca_pos_temp_allLicks - np.tile(videoFrameNum * numVideoShift,(1,len(ca_pos_temp_allLicks)))
            # overlay_temp =np.zeros(-1*preTrigWindow+postTrigWindow)
            # overlay_temp[100]=0
            # overlay_temp[100+ca_pos_lick-ca_pos]=0
            plt.plot((100,100),(trig_id,trig_id+1),scaley=False,color='k')
            plt.plot((100+ca_pos-ca_pos_lick,100+ca_pos-ca_pos_lick),(trig_id,trig_id+1),scaley=False,color='k')
            plt.plot((100+ca_pos_pump-ca_pos_lick,100+ca_pos_pump-ca_pos_lick),(trig_id,trig_id+1),scaley=False,color='r')

            for lick_ind in ca_pos_allLicks:
                plt.plot((100 + lick_ind - ca_pos_lick, 100 + lick_ind - ca_pos_lick), (trig_id, trig_id + 1), scaley=False, color='g',linewidth=1)

            plot_ca = ca_trace[ca_pos_lick + preTrigWindow : ca_pos_lick + postTrigWindow]
            # print(ca_pos_temp)
            # print(len(plot_ca))
            if len(plot_ca)==abs(preTrigWindow)+abs(postTrigWindow):
                ca_raw_traces.append(plot_ca)
                pumpDelays.append(pumpDelaysTime[k])
                pump_trigger_history.append(ca_pos_temp_pump)
                k=k+1
            # ca_raw_overlay_lines.append(overlay_temp)

        except:
            print("Analysis window is outside ca2+ data size")
            k=k+1


    tVector = np.linspace(preTrigWindow, postTrigWindow, numAnalysisWindow) * caFrameDur
    xLabel = np.linspace(preTrigWindow, postTrigWindow, numAnalysisWindow)
    xLabel[:] = np.nan
    xSegments = np.linspace(preTrigWindow,postTrigWindow,9)
    xSegments_integer = [int(i+(-1)*preTrigWindow) for i in xSegments]
    xSegments_integer[-1]=-1
    xLabel[xSegments_integer]=xSegments*caFrameDur
    ax = sn.heatmap(scipy.stats.zscore(ca_raw_traces,axis=1), cmap="coolwarm",xticklabels=xLabel)
    ax.set_title("component_"+str(valid_idx[i]))
    plt.savefig(dir+os.sep+plotSavingFolder+os.sep+"heatmap"+str(valid_idx[i])+'.eps',format='eps')

    plt.figure()
    ca_traces_sorted = []
    pump_trigger_history_sorted = []
    unique_pumpDelays = np.sort(np.unique(pumpDelays))
    for iii in range(len(unique_pumpDelays)):
        for jjj in range(len(pumpDelays)):
            if pumpDelays[jjj] == unique_pumpDelays[iii]:
                pump_trigger_history_sorted.append(pump_trigger_history[jjj])
    for iii in range(len(pump_trigger_history_sorted)):
        ca_pos_temp_pump = pump_trigger_history_sorted[iii]
        ca_pos_temp = findBeforeClosestIndex(trimmed_pokeTrigTime, caTime[ca_pos_temp_pump], caTime)
        ca_pos_temp_lick = findAfterClosestIndex(trimmed_pumpTrigTime, caTime[ca_pos_temp], caTime)

        ca_pos = ca_pos_temp - videoFrameNum * numVideoShift
        ca_pos_lick = ca_pos_temp_lick - videoFrameNum * numVideoShift
        ca_pos_pump = ca_pos_temp_pump - videoFrameNum * numVideoShift

        ca_pos_temp_allLicks = findAllLickIndices(trimmed_lickTrigTime, ca_pos_temp_lick, caTime, postTrigWindow)
        ca_pos_allLicks = ca_pos_temp_allLicks - np.tile(videoFrameNum * numVideoShift, (1, len(ca_pos_temp_allLicks)))
        # overlay_temp =np.zeros(-1*preTrigWindow+postTrigWindow)
        # overlay_temp[100]=0
        # overlay_temp[100+ca_pos_lick-ca_pos]=0
        plt.plot((100, 100), (iii, iii + 1), scaley=False, color='k')
        plt.plot((100 + ca_pos - ca_pos_lick, 100 + ca_pos - ca_pos_lick), (iii, iii + 1), scaley=False,
                 color='k')
        plt.plot((100 + ca_pos_pump - ca_pos_lick, 100 + ca_pos_pump - ca_pos_lick), (iii, iii + 1),
                 scaley=False, color='r')

        for lick_ind in ca_pos_allLicks:
            plt.plot((100 + lick_ind - ca_pos_lick, 100 + lick_ind - ca_pos_lick), (iii, iii + 1), scaley=False,
                     color='g', linewidth=1)
        plot_ca = ca_trace[ca_pos_pump + preTrigWindow: ca_pos_pump + postTrigWindow]
        ca_traces_sorted.append(plot_ca)

    ax = sn.heatmap(scipy.stats.zscore(ca_traces_sorted,axis=1), cmap="coolwarm",xticklabels=xLabel)
    ax.set_title("component_" + str(valid_idx[i]))
    plt.savefig(dir+os.sep+plotSavingFolder+os.sep+"sorted"+str(valid_idx[i])+'.eps',format='eps')


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
    ax.set_title("component_"+str(valid_idx[i]))
    plt.savefig(dir+os.sep+plotSavingFolder+os.sep+str(valid_idx[i])+'.eps',format='eps')
    # plt.figure()

#plt.show()

print("done")
input()





