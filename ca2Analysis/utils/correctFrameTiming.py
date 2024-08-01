import csv
import tkinter.filedialog
from tkinter import *
from edgeDetection import *
import os
import matplotlib.pyplot as plt
import numpy as np
from readTabTxtFile import *
import re
import pickle
from scipy import optimize
import copy
from caiman.source_extraction.cnmf.cnmf import load_CNMF


filePath = "/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/newcohort_03242024/reward_seeking/7277702162024/poke_lick_sham_day20/15_37_36"
miniscopeFile = filePath + os.sep + 'My_V4_Miniscope' + os.sep + 'timeStamps.csv'
caTimeNum = 4
caTimeFile = "stim"+str(caTimeNum)+".txt"
caTime = readTabTxtFile(filePath+os.sep+caTimeFile, 'float')
plotSavingFolder = 'Plots'
if not os.path.exists(filePath + os.sep + plotSavingFolder):
    os.mkdir(filePath + os.sep + plotSavingFolder)
if not os.path.exists(filePath+os.sep+'cellTraces.txt'):
    cnmf_save = load_CNMF(filePath + os.sep + 'output_rescaled_cnmf_save.hdf5')
    cellTraces = cnmf_save.estimates.C
    valid_idx = cnmf_save.estimates.idx_components
    import csv

    traces_sz = cellTraces.shape
    with open(filePath + os.sep + "cellTraces.txt", 'w', newline="\n") as f:
        f_writer = csv.writer(f, delimiter='\t')
        for i in range(traces_sz[0]):
            f_writer.writerow(cellTraces[i, :])
    with open(filePath + os.sep + "validIdx.txt", 'w', newline="\n") as f:
        f_writer = csv.writer(f, delimiter='\t')
        f_writer.writerow(valid_idx)
def segments_fit(X, Y, count, xanchors=slice(None), yanchors=slice(None)):
    xmin = X.min()
    xmax = X.max()
    ymin = Y.min()
    ymax = Y.max()

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

    r = optimize.minimize(err, x0=np.r_[seg, py_init], method='Nelder-Mead',bounds=((xmin,xmax),(xmin,xmax),(ymin,ymax),(ymin,ymax),(ymin,ymax),(ymin,ymax)))
    while not r.success:
        r = optimize.minimize(err, x0=r.x, method='Nelder-Mead', bounds=(
        (xmin, xmax), (xmin, xmax), (ymin, ymax), (ymin, ymax), (ymin, ymax), (ymin, ymax)))
    return func(r.x)
def returnCa2DataLengthFromFile(dataFile):
    caData = readTabTxtFile(dataFile, 'float')
    return len(caData)

totLen = 0
for files in os.listdir(filePath):
    if re.search('cellTraces_norm*',files):
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
# plt.show(block=True)

intervals_caTime = [caTime[i+1] - caTime[i] for i in range(len(caTime)-1)]
# timePerFrame = np.mean(intervals_caTime)
timePerFrame = np.mean(intervals_csv)/1000
plt.figure()
plt.plot(intervals_caTime)
plt.xlabel("Frame number")
plt.ylabel("inter-frame interval (s)")
plt.title('From recorded Ca2+ frame triggers')
plt.savefig(filePath + os.sep + plotSavingFolder + os.sep + 'time_intervals_recorded_frameTime.eps', format='eps')

predictedClock = [timePerFrame*1000*(i-1) for i in range(len(csvData))]
sysClock = [int(csvData[i][1]) for i in range(len(csvData))]

interval_diff = np.array([sysClock[i] - predictedClock[i] for i in range(len(sysClock))])
frame_number = np.array([i for i in range(len(sysClock)) ])

fx,fy=segments_fit(frame_number,interval_diff,3,xanchors=[0,1,1,3])
plt.figure()
plt.plot(frame_number,interval_diff)
plt.plot(fx,fy,'o')
plt.xlabel("Frame number")
plt.ylabel("Diff between predicted clock and csv timestamps (ms)")
plt.savefig(filePath + os.sep + plotSavingFolder + os.sep + 'segment_fit_frame_correction.eps', format='eps')

# frame_correction_pos = [int(fx[2]), round((fy[2]-fy[1])/1000/timePerFrame)]
frame_correction_pos = []
with open(filePath + os.sep + 'frame_correction_pos'+'.pickle','wb') as f:
    pickle.dump(frame_correction_pos,f)
print(fx)
print(fy)
print(frame_correction_pos)

input()