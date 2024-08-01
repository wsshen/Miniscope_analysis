import csv
import tkinter.filedialog
from tkinter import *
from edgeDetection import *
import os
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import numpy as np
from readTabTxtFile import *
from calTempResolution import *
from trimmedTriggerTime import *
import seaborn as sn
import scipy
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import csv
import pickle
directory = tkinter.filedialog.askdirectory(initialdir="/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/newcohort_03242024/sound_discrimination/7277102232024/training_with_miniscope_recordings/day6_training_no_time/11_10_51")

if not os.path.exists(directory+os.sep+'cellTraces_norm.txt'):
    cnmf_save = load_CNMF(directory + os.sep+'output_rescaled_cnmf_save.hdf5')
    cellTraces = cnmf_save.estimates.C
    valid_idx = cnmf_save.estimates.idx_components
    traces_sz = cellTraces.shape

    with open(directory + os.sep + "cellTraces.txt", 'w', newline="\n") as f:
        f_writer = csv.writer(f, delimiter='\t')
        for i in range(traces_sz[0]):
            f_writer.writerow(cellTraces[i, :])
    with open(directory+os.sep+"validIdx.txt", 'w', newline="\n") as f:
        f_writer = csv.writer(f, delimiter='\t')
        f_writer.writerow(valid_idx)

# for 49140 and 49142
# frameInput = 0
# lickTrigNum = 1
# pokeTrigNum = 2
# caTimeNum = 0
# pumpTrigNum = 4

# for 51331, 51332 and 51333
frameInput = 3
lickTrigNum = 2
pokeTrigNum = 0
caTimeNum = 3
pumpTrigNum = 1

# for 72771
frameInput = 7
leftLickTrigNum = 1
RightLickTrigNum = 2
leftSoundTrigNum = 3
rightSoundTrigNum = 4
leftPumpTrigNum = 5
pokeTrigNum = 0
caTimeNum = 7
rightPumpTrigNum = 6

validIdFile = "validIdx.txt"
cellTracesFile = "cellTraces_norm.txt"
pokeTriggerFile = "stim"+str(pokeTrigNum)+".txt"
lickTriggerFile = "stim"+str(lickTrigNum)+".txt"
pumpTriggerFile = "stim"+str(pumpTrigNum)+".txt"

caTimeFile = "stim"+str(caTimeNum)+".txt"
plotSavingFolder = 'Plots'
videoFrameNum = 1000

with open(directory + os.sep + 'frame_correction_pos.pickle','rb') as f:
    frame_correction = pickle.load(f)

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

def interpolateNaN(x):
    loc = np.argwhere(np.isnan(x))
    for i in loc:
        x[i[0]] = (x[i[0]-1] + x[i[0]+1])/2
    return x
with open(directory+os.sep+cellTracesFile, "r", newline="\n") as readFile:
    data_reader = readFile.readlines()
# with open(directory+os.sep+validIdFile, "r", newline="") as readFile:
#     id_reader = readFile.readlines()
# with open(directory+os.sep+triggerFile, "r", newline="") as readFile:
#     trig_reader = readFile.readlines()
# with open(directory+os.sep+caTimeFile, "r", newline="") as readFile:
#     caTime_reader = readFile.readlines()

caData = readTabTxtFile(directory+os.sep+cellTracesFile, 'float')
valid_idx = readTabTxtFile(directory+os.sep+validIdFile, 'int')
pokeTrigTime = readTabTxtFile(directory+os.sep+pokeTriggerFile, 'float')
lickTrigTime = readTabTxtFile(directory+os.sep+lickTriggerFile, 'float')
caTime = readTabTxtFile(directory+os.sep+caTimeFile, 'float')
caFrameDur = calTempResolution(caTime)
pumpTrigTime = readTabTxtFile(directory+os.sep+pumpTriggerFile, 'float')

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


for i in valid_idx:
    print(i)
    ca_trace = [float(k) for k in data_reader[i].strip().split("\t")]
    if frame_correction:
        ca_trace_corrected = ca_trace[0:frame_correction[0]+1]+ [float('NaN')]*frame_correction[1] + ca_trace[frame_correction[0]+1:-1]
    else:
        ca_trace_corrected = ca_trace
    ca_raw_traces = []
    # ca_raw_overlay_lines = []
    plt.figure()

    for trig_id in range(numTrig):
        try:
            # ca_pos_temp = findClosestTimeIndex(trimmed_pokeTrigTime[trig_id]-pokeTimeAdj,caTime)
            # ca_pos_temp_lick = findClosestLickIndex(trimmed_lickTrigTime,trimmed_pokeTrigTime[trig_id]-pokeTimeAdj,caTime)
            #
            # ca_pos = ca_pos_temp - videoFrameNum * numVideoShift
            # ca_pos_lick = ca_pos_temp_lick - videoFrameNum * numVideoShift
            # ca_pos_temp_allLicks = findAllLickIndices(trimmed_lickTrigTime,ca_pos_lick,caTime,postTrigWindow)
            # # overlay_temp =np.zeros(-1*preTrigWindow+postTrigWindow)
            # # overlay_temp[100]=0
            # # overlay_temp[100+ca_pos_lick-ca_pos]=0
            # plt.plot((100,100),(trig_id,trig_id+1),scaley=False,color='k')
            # plt.plot((100+ca_pos-ca_pos_lick,100+ca_pos-ca_pos_lick),(trig_id,trig_id+1),scaley=False,color='k')
            # for lick_ind in ca_pos_temp_allLicks:
            #     plt.plot((100 + lick_ind - ca_pos_lick, 100 + lick_ind - ca_pos_lick), (trig_id, trig_id + 1), scaley=False, color='g',linewidth=1)
            #
            # plot_ca = ca_trace[ca_pos_lick + preTrigWindow : ca_pos_lick + postTrigWindow]

            ca_pos_temp_pump = findClosestTimeIndex(trimmed_pumpTrigTime[trig_id], caTime)
            ca_pos_temp = findBeforeClosestIndex(trimmed_pokeTrigTime, trimmed_pumpTrigTime[trig_id], caTime)
            ca_pos_temp_lick = findAfterClosestIndex(trimmed_lickTrigTime, caTime[ca_pos_temp], caTime)

            ca_pos = ca_pos_temp - videoFrameNum * numVideoShift
            ca_pos_lick = ca_pos_temp_lick - videoFrameNum * numVideoShift
            ca_pos_pump = ca_pos_temp_pump - videoFrameNum * numVideoShift

            ca_pos_temp_allLicks = findAllLickIndices(trimmed_lickTrigTime, ca_pos_temp_lick, caTime, postTrigWindow)
            ca_pos_allLicks = ca_pos_temp_allLicks - np.tile(videoFrameNum * numVideoShift,
                                                             (1, len(ca_pos_temp_allLicks)))
            # overlay_temp =np.zeros(-1*preTrigWindow+postTrigWindow)
            # overlay_temp[100]=0
            # overlay_temp[100+ca_pos_lick-ca_pos]=0
            plt.plot((100, 100), (trig_id, trig_id + 1), scaley=False, color='w')
            # plt.text(trig_id, trig_id + 1,str(trig_id))
            plt.plot((100 + ca_pos - ca_pos_lick, 100 + ca_pos - ca_pos_lick), (trig_id, trig_id + 1), scaley=False,
                     color='k')
            plt.plot((100 + ca_pos_pump - ca_pos_lick, 100 + ca_pos_pump - ca_pos_lick), (trig_id, trig_id + 1),
                     scaley=False, color='r')

            for lick_ind in ca_pos_allLicks:
                plt.plot((100 + lick_ind - ca_pos_lick, 100 + lick_ind - ca_pos_lick), (trig_id, trig_id + 1),
                         scaley=False, color='g', linewidth=1)

            plot_ca = ca_trace_corrected[ca_pos_lick + preTrigWindow: ca_pos_lick + postTrigWindow]
            # print(ca_pos_temp)
            # print(len(plot_ca))
            if len(plot_ca)==abs(preTrigWindow)+abs(postTrigWindow):
                ca_raw_traces.append(plot_ca)
            # ca_raw_overlay_lines.append(overlay_temp)

        except:
            print("Analysis window is outside ca2+ data size")
    tVector = np.linspace(preTrigWindow, postTrigWindow, numAnalysisWindow) * caFrameDur
    xLabel = np.linspace(preTrigWindow, postTrigWindow, numAnalysisWindow)
    xLabel[:] = np.nan
    xSegments = np.linspace(preTrigWindow,postTrigWindow,9)
    xSegments_integer = [int(i+(-1)*preTrigWindow) for i in xSegments]
    xSegments_integer[-1]=-1
    xLabel[xSegments_integer]=xSegments*caFrameDur
    ax = sn.heatmap(ca_raw_traces, cmap="coolwarm",xticklabels=xLabel) #scipy.stats.zscore(ca_raw_traces,axis=1,nan_policy='omit')
    ax.set_title("component_"+str(i))
    plt.savefig(directory+os.sep+plotSavingFolder+os.sep+"heatmap"+str(i)+'.eps',format='eps')
    plt.close()


    fig= plt.figure()
    ax = fig.add_subplot(1,1,1)
    tVector = np.linspace(preTrigWindow,postTrigWindow,numAnalysisWindow)*caFrameDur
    ca_mean_traces = np.mean(ca_raw_traces,axis=0)
    sem = scipy.stats.sem(ca_raw_traces,axis=0)

    ca_mean_traces = interpolateNaN(ca_mean_traces)
    sem = interpolateNaN(sem)
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
    plt.savefig(directory+os.sep+plotSavingFolder+os.sep+str(i)+'.eps',format='eps')
    plt.close()

    fig= plt.figure()
    ca_raw_traces_np = np.array(ca_raw_traces)

    for row_i in range(ca_raw_traces_np.shape[0]):
        plt.plot(tVector,row_i+ca_raw_traces_np[row_i,:]/np.max(ca_raw_traces_np[row_i,:]))
    plt.savefig(directory+os.sep+plotSavingFolder+os.sep+str(i)+'_rasterPlot.eps',format='eps')
    plt.close()

#plt.show()

print("done")
input()





