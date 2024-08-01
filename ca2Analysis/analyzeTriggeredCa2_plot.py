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
import sys
import time

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

def findAllLickIndices(t1Series,refT,tSeries,window,caFrameDur):
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

def plot(args):

    directory = args.directory
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
    videoFrameNum = 1000
    numVideoShift = 0
    # for 72771
    exp_dict = {}
    args_dict = vars(args)
    for arg in args_dict:
        if arg[-3:] == 'Num':
            exp_dict[arg] = args_dict[arg]
            exp_dict[arg[0:-3]+'Time'] = readTabTxtFile(directory+os.sep+"stim"+str(args_dict[arg])+".txt", 'float')
            
    for el in exp_dict:
        if el[-7:] == 'TrigTime':
            exp_dict[el] = trimmedTriggerTime(exp_dict[el],exp_dict['caTime'],videoFrameNum*numVideoShift,-1)

    # leftLickTrigNum = args.leftLickTrigNum
    # rightLickTrigNum = args.rightLickTrigNum
    # leftSoundTrigNum = args.leftSoundTrigNum
    # rightSoundTrigNum = args.rightSoundTrigNum
    # leftPumpTrigNum = args.leftPumpTrigNum
    # pokeTrigNum = args.pokeTrigNum
    # caTimeNum = args.caTimeNum
    # rightPumpTrigNum = args.rightPumpTrigNum

    validIdFile = "validIdx.txt"
    cellTracesFile = "cellTraces_norm.txt"
    
    # pokeTriggerFile = "stim"+str(pokeTrigNum)+".txt"
    # leftLickTriggerFile = "stim"+str(leftLickTrigNum)+".txt"
    # rightLickTriggerFile = "stim"+str(rightLickTrigNum)+".txt"
    # leftSoundTriggerFile = "stim"+str(leftSoundTrigNum)+".txt"
    # rightSoundTriggerFile = "stim"+str(rightSoundTrigNum)+".txt"    
    # rightPumpTriggerFile = "stim"+str(rightPumpTrigNum)+".txt"
    # leftPumpTriggerFile = "stim"+str(leftPumpTrigNum)+".txt"
    # caTimeFile = "stim"+str(caTimeNum)+".txt"

    plotSavingFolder = args.plot_dir


    with open(directory + os.sep + 'frame_correction_pos.pickle','rb') as f:
        frame_correction = pickle.load(f)


    with open(directory+os.sep+cellTracesFile, "r", newline="\n") as readFile:
        data_reader = readFile.readlines()
    # with open(directory+os.sep+validIdFile, "r", newline="") as readFile:
    #     id_reader = readFile.readlines()
    # with open(directory+os.sep+triggerFile, "r", newline="") as readFile:
    #     trig_reader = readFile.readlines()
    # with open(directory+os.sep+caTimeFile, "r", newline="") as readFile:
    #     caTime_reader = readFile.readlines()

    valid_idx = readTabTxtFile(directory+os.sep+validIdFile, 'int')
    pokeTrigTime = exp_dict['pokeTrigTime']
    leftLickTrigTime = exp_dict['leftLickTrigTime']
    rightLickTrigTime = exp_dict['rightLickTrigTime']
    leftSoundTrigTime = exp_dict['leftSoundTrigTime']
    rightSoundTrigTime = exp_dict['rightSoundTrigTime']
    leftPumpTrigTime = exp_dict['leftPumpTrigTime']
    rightPumpTrigTime = exp_dict['rightPumpTrigTime']
    caTime = exp_dict['caTime']
    
    caFrameDur = calTempResolution(exp_dict['caTime'])

    if args.plot_zero == 'rightLick':
        print(args.plot_zero)
        refTrigTime = rightPumpTrigTime # indicates whether a trial starts
        pre_refTrigTime = pokeTrigTime # indicates the action that started before the ref
        post_refTrigTime = rightLickTrigTime # indicates the action ensuing the ref
    if args.plot_zero == 'leftLick':
        print(args.plot_zero)
        refTrigTime = leftPumpTrigTime # indicates whether a trial starts
        pre_refTrigTime = pokeTrigTime # indicates the action that started before the ref
        post_refTrigTime = leftLickTrigTime # indicates the action ensuing the ref
    if args.plot_zero == 'leftSound':
        refTrigTime = leftSoundTrigTime # indicates whether a trial starts
        pre_refTrigTime = pokeTrigTime # indicates the action that started before the ref
        post_refTrigTime = leftLickTrigTime # indicates the action ensuing the ref

    # valid_idx = [int(i) for i in id_reader[0].strip().split("\t")]
    # caTime = [float(i) for i in caTime_reader[0].strip().split("\t")]
    # trigTime = [float(i) for i in trig_reader[0].strip().split("\t")]

    preTrigWindow = args.pre_window
    plot_prezero = -1 * preTrigWindow
    postTrigWindow = args.post_window

    numTrig = len(exp_dict['leftPumpTrigTime']) + len(exp_dict['rightPumpTrigTime'])

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

                ca_pos_temp_ref = findClosestTimeIndex(refTrigTime[trig_id], caTime)
                ca_pos_temp_pre_ref = findBeforeClosestIndex(pre_refTrigTime, refTrigTime[trig_id], caTime)
                ca_pos_temp_post_ref = findAfterClosestIndex(post_refTrigTime, caTime[ca_pos_temp_pre_ref], caTime) #caTime[ca_pos_temp_pre_ref]

                ca_pos_pre_ref = ca_pos_temp_pre_ref - videoFrameNum * numVideoShift
                ca_pos_post_ref = ca_pos_temp_post_ref - videoFrameNum * numVideoShift
                ca_pos_ref = ca_pos_temp_ref - videoFrameNum * numVideoShift

                ca_pos_temp_post_ref_all = findAllLickIndices(post_refTrigTime, ca_pos_temp_pre_ref, caTime, postTrigWindow,caFrameDur)
                ca_pos_post_ref_all = ca_pos_temp_post_ref_all - np.tile(videoFrameNum * numVideoShift,
                                                                (1, len(ca_pos_temp_post_ref_all)))
                
                ca_pos_temp_pre_ref_all = findAllLickIndices(pre_refTrigTime, ca_pos_temp_pre_ref, caTime, postTrigWindow,caFrameDur)
                ca_pos_pre_ref_all = ca_pos_temp_pre_ref_all - np.tile(videoFrameNum * numVideoShift,
                                                                (1, len(ca_pos_temp_pre_ref_all)))
                
                ca_pos_temp_ref_all = findAllLickIndices(refTrigTime, ca_pos_temp_pre_ref, caTime, postTrigWindow,caFrameDur)
                ca_pos_ref_all = ca_pos_temp_ref_all - np.tile(videoFrameNum * numVideoShift,
                                                                (1, len(ca_pos_temp_ref_all)))
                # overlay_temp =np.zeros(-1*preTrigWindow+postTrigWindow)
                # overlay_temp[100]=0
                # overlay_temp[100+ca_pos_lick-ca_pos]=0
                plt.plot((plot_prezero, plot_prezero), (trig_id, trig_id + 1), scaley=False, color='w')
                # plt.text(trig_id, trig_id + 1,str(trig_id))
                plt.plot((plot_prezero + ca_pos_pre_ref - ca_pos_post_ref, plot_prezero + ca_pos_pre_ref - ca_pos_post_ref), (trig_id, trig_id + 1), scaley=False,
                        color='k')
                plt.plot((plot_prezero + ca_pos_ref - ca_pos_post_ref, plot_prezero + ca_pos_ref - ca_pos_post_ref), (trig_id, trig_id + 1),
                        scaley=False, color='r')

                for ind in ca_pos_post_ref_all:
                    plt.plot((plot_prezero + ind - ca_pos_post_ref, plot_prezero + ind - ca_pos_post_ref), (trig_id, trig_id + 1),
                            scaley=False, color='g', linewidth=1)
                    
                for ind in ca_pos_pre_ref_all:
                    plt.plot((plot_prezero + ind - ca_pos_post_ref, plot_prezero + ind - ca_pos_post_ref), (trig_id, trig_id + 1),
                            scaley=False, color='k', linewidth=1)
                    
                for ind in ca_pos_ref_all:
                    plt.plot((plot_prezero + ind - ca_pos_post_ref, plot_prezero + ind - ca_pos_post_ref), (trig_id, trig_id + 1),
                            scaley=False, color='r', linewidth=1)
                plot_ca = ca_trace_corrected[ca_pos_post_ref + preTrigWindow: ca_pos_post_ref + postTrigWindow]
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
        plt.savefig(plotSavingFolder+os.sep+"heatmap"+str(i)+'.eps',format='eps')
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
        plt.savefig(plotSavingFolder+os.sep+str(i)+'.eps',format='eps')
        plt.close()

        # fig= plt.figure()
        # ca_raw_traces_np = np.array(ca_raw_traces)

        # for row_i in range(ca_raw_traces_np.shape[0]):
        #     plt.plot(tVector,row_i+ca_raw_traces_np[row_i,:]/np.max(ca_raw_traces_np[row_i,:]))
        # plt.savefig(os.sep+plotSavingFolder+os.sep+str(i)+'_rasterPlot.eps',format='eps')
        # plt.close()

    #plt.show()

    print("done")
    input()


def main():
    import argparse
    # class LoadFromFile (argparse.Action):
    #     def __call__ (self, parser, namespace, values, option_string = None):
    #         print("inside loadfromfile")
    #         with open(os.path.join(data_path,plot_folder,'inputs.txt')) as f:
    #             # parse arguments in the file and store them in the target namespace
    #             parser.parse_args(f.read().split(), namespace)

    parser = argparse.ArgumentParser() # fromfile_prefix_chars='@'

    parser.add_argument("--plot_zero",type=str)
    parser.add_argument("--file",'-f',type=str)
    parser.add_argument("--directory", type=str)
    parser.add_argument("--pre_window",type=int,default=-100)
    parser.add_argument("--post_window",type=int,default=200)
    parser.add_argument("--leftLickTrigNum",type=int)
    parser.add_argument("--rightLickTrigNum",type=int)
    parser.add_argument("--leftSoundTrigNum",type=int)
    parser.add_argument("--rightSoundTrigNum",type=int)
    parser.add_argument("--leftPumpTrigNum",type=int)
    parser.add_argument("--rightPumpTrigNum",type=int)
    parser.add_argument("--pokeTrigNum",type=int)
    parser.add_argument("--caNum",type=int)
    # parser.add_argument('--file',type=open,action=LoadFromFile)
    # parser.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)


    data_path = "/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/newcohort_03242024/sound_discrimination/7277102232024/training_with_miniscope_recordings/day6_training_no_timeout/11_10_51"
    plot_folder = 'Plots'
    plot_path = os.path.join(data_path,plot_folder)
    os.chdir(plot_path)
    # args = parser.parse_args(['@inputs.txt'])
    # args = parser.parse_args(['--file',os.path.join(plot_path,'inputs.txt')])
    args = parser.parse_known_args()[0]

    if args.file:
        with open(args.file,'r') as f:
            print('inside')
            parser.parse_known_args(f.read().split(),namespace=args)[0]
    
    if not (os.path.exists(plot_path)):
        os.makedirs(plot_path)
 
    plotdir = (
        args.plot_zero
        + "_"
        + time.strftime("%d-%m-%Y_%H-%M-%S")
    )
    plotdir = os.path.join(plot_path, plotdir)
    args.plot_dir = plotdir
    if not (os.path.exists(plotdir)):
        os.makedirs(plotdir)

    plot(args)


if __name__ == "__main__":
    main()