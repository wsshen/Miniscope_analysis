import csv
import tkinter.filedialog
from tkinter import *
from edgeDetection import *
import os
import matplotlib.pyplot as plt
import numpy as np
from readTabTxtFile import *
import sys

dir = tkinter.filedialog.askdirectory(initialdir="/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/5133207132023/reward_seeking/days_with_miniscope_recording/day1_poke_lick/session1/14_29_23/My_V4_miniscope")

def findClosestTimeIndex(t,tSeries):
    tDiff = abs(t-tSeries)
    minVal = min(tDiff)
    pos = np.where(tDiff ==minVal)
    return pos
daq_type = 'type2'

if daq_type == 'type1':
    fileName = "DAQStims_05_18_2023_10_46_36.txt"
    stimData = readTabTxtFile(dir+os.sep+fileName,'string')

    edgeDetectList = [7,2]
    risingEdgeDetect = [6,5,4,3,1,0]
    fallingEdgeDetect = []
    expStartStop =5

    trigData = []
    trigTime = []

    if len(stimData) % 2 != 0:
        sys.exit("Odd number in the DAQ file. Please double check the number of triggers collected.")
    else:
        endTimeStr = stimData[-2]
        endTime = float(format(float(endTimeStr), '.3f'))

    timePoints = np.arange(0,endTime,0.001)
    for index in range(1, len(stimData), 2):
        trigTime.append(format(float(stimData[index-1]),'.3f'))
        trigData.append(format(int(stimData[index]),'08b'))
    numTrig = len(trigData[0])

    for index in range(numTrig):
        trig = [port[index] for port in trigData]
        if index in edgeDetectList:
            trig_time = edgeDetection(trig, trigTime)

        if index in risingEdgeDetect:
            trig_time = risingEdgeDetection(trig, trigTime)
        if index in fallingEdgeDetect:
            trig_time = fallingEdgeDetection(trig, trigTime)

        trig_time_str = ""
        for ii in trig_time:
            trig_time_str += str(ii) + '\t'
        with open(dir+os.sep+"raw_stim_"+str(numTrig-1-index)+".txt", "w") as writeFile:
            writeFile.write(trig_time_str)

    startStopData = readTabTxtFile(dir+os.sep+"raw_stim_"+str(expStartStop)+".txt",'float')

    if len(startStopData)>2:
        expStart = startStopData[1] #if there is an additional trigger collected at the beginning
        expStop = startStopData[2]
    else:
        expStart = startStopData[0]
        expStop = startStopData[1]

    for index in range(numTrig):
        stimData = readTabTxtFile(dir+os.sep+"raw_stim_"+str(index)+".txt",'float')
        trig_time_str=""
        for data in stimData:
            if data>=expStart and data<=expStop:
                trig_time_str += str(data) + '\t'
        with open(dir+os.sep+"stim"+str(index)+".txt", "w") as writeFile:
            writeFile.write(trig_time_str)

if daq_type == 'type2':

    fileName = "trigger_all_data.txt"
    stimData =[]
    with open(dir+os.sep+fileName, "r", newline="") as readFile:
        file_reader = csv.reader(readFile, delimiter='\t')
        for entry in file_reader:
            stimData.append(entry)
    edgeDetectList = []
    risingEdgeDetect = [7, 2, 6, 5, 4, 3, 1, 0]
    fallingEdgeDetect = []

    trigTime = []

    samplingRate = 1e4
    endTime = float(format(float(len(stimData)/samplingRate), '.3f'))

    timePoints = np.arange(0, endTime, 0.001)
    for index in range(len(stimData)):
        trigTime.append(format(float(index/samplingRate), '.3f'))
    numTrig = len(stimData[0])

    for index in range(numTrig):
        trig = [port[index] for port in stimData]
        if index in edgeDetectList:
            trig_time = edgeDetection(trig, trigTime)

        if index in risingEdgeDetect:
            trig_time = risingEdgeDetection(trig, trigTime)
        if index in fallingEdgeDetect:
            trig_time = fallingEdgeDetection(trig, trigTime)

        trig_time_str = ""
        for ii in trig_time:
            trig_time_str += str(ii) + '\t'
        with open(dir + os.sep + "raw_stim_" + str(index) + ".txt", "w") as writeFile:
            writeFile.write(trig_time_str)

