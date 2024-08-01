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


def makePlot(exp):
    for i in exp.valid_idx:
        ca_trace = [float(k) for k in exp.data_reader[i].strip().split("\t")]
        if exp.frame_correction:
            exp.ca_trace_corrected = ca_trace[0:exp.frame_correction[0]+1]+ [float('NaN')]* exp.frame_correction[1] + ca_trace[exp.frame_correction[0]+1:-1]
        else:
            exp.ca_trace_corrected = ca_trace

        
        if len(exp.numTrig)>1: # if this is a comparison plot
            ca_traces_left = []
            ca_traces_right = []
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

            for side_key,side_value in exp.goodTrialTime.items():

                if side_key == 'left':
                    triggers = []
                    for trig_id in range(exp.numTrig[0]):
                        try:
                            trigger = exp.findTriggerPositionsForPlot(side_value[trig_id],side_key)
                            # if len(plot_ca)==abs(exp.preTrigWindow)+abs(exp.postTrigWindow):
                            #     ca_traces_left.append(plot_ca)
                            if trigger:
                                triggers.append(trigger)
                        except:
                            print("Analysis window is outside ca2+ data size")
                    ca_traces_left = exp.findCaResponsesFromTriggers(triggers)
                    exp.plotResponsesWithTriggers(triggers,0,ax)

                        # ca_pos_temp_trial = exp.findClosestTimeIndex(side_value[trig_id], exp.caTime)
                        # if hasattr(exp,'pre'):
                        #     ca_pos_temp_pre = exp.findBeforeClosestIndex(exp.pre['1'], side_value[trig_id], exp.caTime)
                        #     ca_pos_temp_ref = exp.findAfterClosestIndex(exp.ref[side_key], exp.caTime[ca_pos_temp_pre], exp.caTime) #caTime[ca_pos_temp_pre_ref]
                        #     ca_pos_ref = ca_pos_temp_ref - exp.videoFrameNum * exp.numVideoShift
                        #     ca_pos_temp_ref_all = exp.findAllIndicesPostRef(exp.ref[side_key], ca_pos_temp_pre, exp.caTime, exp.postTrigWindow,exp.caFrameDur)
                        #     ca_pos_ref_all = ca_pos_temp_ref_all - np.tile(exp.videoFrameNum * exp.numVideoShift,
                        #                                         (1, len(ca_pos_temp_ref_all)))
                        #     for pre_ind in range(len(exp.pre)):
                        #         ca_pos_temp_pre_ref = exp.findBeforeClosestIndex(exp.pre[str(pre_ind+1)], side_value[trig_id], exp.caTime)
                        #         ca_pos_pre_ref = ca_pos_temp_pre_ref - exp.videoFrameNum * exp.numVideoShift
                        #         plt.plot((exp.plot_prezero + ca_pos_pre_ref - ca_pos_ref, exp.plot_prezero + ca_pos_pre_ref - ca_pos_ref), (trig_id, trig_id + 1), scaley=False,
                        #             color=colors[pre_ind,0])
                        #         ca_pos_temp_pre_ref_all = exp.findAllIndicesPostRef(exp.pre[str(pre_ind+1)], ca_pos_temp_pre, exp.caTime, exp.postTrigWindow,exp.caFrameDur)
                        #         ca_pos_pre_ref_all = ca_pos_temp_pre_ref_all - np.tile(exp.videoFrameNum * exp.numVideoShift,
                        #                                                 (1, len(ca_pos_temp_pre_ref_all)))
                        #         for ind in ca_pos_pre_ref_all:
                        #             plt.plot((exp.plot_prezero + ind - ca_pos_ref, exp.plot_prezero + ind - ca_pos_ref), (trig_id, trig_id + 1),
                        #                     scaley=False, color=colors[pre_ind,0], linewidth=1)
                        # for ind in ca_pos_ref_all:
                        #     plt.plot((exp.plot_prezero + ind - ca_pos_ref, exp.plot_prezero + ind - ca_pos_ref), (trig_id, trig_id + 1), 
                        #                             scaley=False, color=colors[0,2], linewidth=1)       

                        # if hasattr(exp,'post'):
                        #     for post_ind in range(len(exp.post)):

                        #         ca_pos_temp_post_ref = exp.findAfterClosestIndex(exp.post[str(post_ind+1)][side_key], exp.caTime[ca_pos_temp_pre], exp.caTime) #caTime[ca_pos_temp_pre_ref]
                        #         ca_pos_post_ref = ca_pos_temp_post_ref - exp.videoFrameNum * exp.numVideoShift
                        #         ca_pos_temp_post_ref_all = exp.findAllIndicesPostRef(exp.post[str(post_ind+1)][side_key], ca_pos_temp_pre, exp.caTime, exp.postTrigWindow,exp.caFrameDur)
                        #         ca_pos_post_ref_all = ca_pos_temp_post_ref_all - np.tile(exp.videoFrameNum * exp.numVideoShift,
                        #                                                         (1, len(ca_pos_temp_post_ref_all)))
                                
                        #         for ind in ca_pos_post_ref_all:
                        #             plt.plot((exp.plot_prezero + ind - ca_pos_ref, exp.plot_prezero + ind - ca_pos_ref), (trig_id, trig_id + 1),
                        #                     scaley=False, color=colors[post_ind,1], linewidth=1)
                        # plot_ca = exp.ca_trace_corrected[ca_pos_ref + exp.preTrigWindow: ca_pos_ref + exp.postTrigWindow]
                        # if len(plot_ca)==abs(exp.preTrigWindow)+abs(exp.postTrigWindow):
                        #     ca_traces_left.append(plot_ca)
                            
                if side_key == 'right':
                    ax.plot(np.linspace(0,exp.numAnalysisWindow,10),np.ones(10)*exp.numTrig[0],scaley=False, color='yellow', linewidth=1)
                    triggers = []
                    for trig_id in range(exp.numTrig[1]):
                        try:
                            trigger = exp.findTriggerPositionsForPlot(side_value[trig_id],side_key)
                            if trigger:
                                triggers.append(trigger)
                        except:
                            print("Analysis window is outside ca2+ data size")
                    ca_traces_right = exp.findCaResponsesFromTriggers(triggers)
                    exp.plotResponsesWithTriggers(triggers,exp.numTrig[0],ax)
            
            ca_traces = np.concatenate((ca_traces_left , ca_traces_right))
            ca_traces = exp.plotHeatmapResponses(ca_traces,ax)
            ax.set_title("component_"+str(i))
            plt.savefig(exp.plotSavingFolder+os.sep+"heatmap"+str(i)+'.eps',format='eps')
            plt.close()

            fig= plt.figure()
            ax = fig.add_subplot(1,1,1)
            if exp.zscore:
                ca_traces_left = ca_traces[0:len(ca_traces_left),:]
                ca_traces_right = ca_traces[len(ca_traces_left):,:]
            exp.plotAverageResponses(ca_traces_left,ax,1)
            exp.plotAverageResponses(ca_traces_right,ax,2)
            print("plotting "+str(i))
            plt.xlabel("Time(s)")
            plt.ylabel("deltaF/F")
            ax.set_title("component_"+str(i))
            plt.legend(["left","", "","right","",""])
            plt.savefig(exp.plotSavingFolder+os.sep+str(i)+'.eps',format='eps')
            plt.close()

            continue
            #             ca_pos_temp_trial = exp.findClosestTimeIndex(side_value[trig_id], exp.caTime)
            #             if hasattr(exp,'pre'):
            #                 ca_pos_temp_pre = exp.findBeforeClosestIndex(exp.pre['1'], side_value[trig_id], exp.caTime)
            #                 ca_pos_temp_ref = exp.findAfterClosestIndex(exp.ref[side_key], exp.caTime[ca_pos_temp_pre], exp.caTime) #caTime[ca_pos_temp_pre_ref]
            #                 ca_pos_ref = ca_pos_temp_ref - exp.videoFrameNum * exp.numVideoShift
            #                 ca_pos_temp_ref_all = exp.findAllIndicesPostRef(exp.ref[side_key], ca_pos_temp_pre, exp.caTime, exp.postTrigWindow,exp.caFrameDur)
            #                 ca_pos_ref_all = ca_pos_temp_ref_all - np.tile(exp.videoFrameNum * exp.numVideoShift,
            #                                                     (1, len(ca_pos_temp_ref_all)))
            #                 for pre_ind in range(len(exp.pre)):
            #                     ca_pos_temp_pre_ref = exp.findBeforeClosestIndex(exp.pre[str(pre_ind+1)], side_value[trig_id], exp.caTime)
            #                     ca_pos_pre_ref = ca_pos_temp_pre_ref - exp.videoFrameNum * exp.numVideoShift
            #                     plt.plot((exp.plot_prezero + ca_pos_pre_ref - ca_pos_ref, exp.plot_prezero + ca_pos_pre_ref - ca_pos_ref), (trig_id + exp.numTrig[0], trig_id + exp.numTrig[0]+ 1), scaley=False,
            #                         color=colors[pre_ind,0])
            #                     ca_pos_temp_pre_ref_all = exp.findAllIndicesPostRef(exp.pre[str(pre_ind+1)], ca_pos_temp_pre, exp.caTime, exp.postTrigWindow,exp.caFrameDur)
            #                     ca_pos_pre_ref_all = ca_pos_temp_pre_ref_all - np.tile(exp.videoFrameNum * exp.numVideoShift,
            #                                                             (1, len(ca_pos_temp_pre_ref_all)))
            #                     for ind in ca_pos_pre_ref_all:
            #                         plt.plot((exp.plot_prezero + ind - ca_pos_ref, exp.plot_prezero + ind - ca_pos_ref), (trig_id+ exp.numTrig[0], trig_id + exp.numTrig[0]+ 1),
            #                                 scaley=False, color=colors[pre_ind,0], linewidth=1)
            #             plt.plot(np.linspace(0,exp.numAnalysisWindow,10),np.ones(10)*exp.numTrig[0],
            #                                 scaley=False, color='yellow', linewidth=1)

            #             for ind in ca_pos_ref_all:
            #                 plt.plot((exp.plot_prezero + ind - ca_pos_ref, exp.plot_prezero + ind - ca_pos_ref), (trig_id+ exp.numTrig[0], trig_id + exp.numTrig[0]+ 1), 
            #                                     scaley=False, color=colors[0,2], linewidth=1)
            #             if hasattr(exp,'post'):
            #                 for post_ind in range(len(exp.post)):

            #                     ca_pos_temp_post_ref = exp.findAfterClosestIndex(exp.post[str(post_ind+1)][side_key], exp.caTime[ca_pos_temp_ref], exp.caTime) #caTime[ca_pos_temp_pre_ref]
            #                     ca_pos_post_ref = ca_pos_temp_post_ref - exp.videoFrameNum * exp.numVideoShift
            #                     ca_pos_temp_post_ref_all = exp.findAllIndicesPostRef(exp.post[str(post_ind+1)][side_key], caQKpwRDslXm6_pos_temp_ref, exp.caTime, exp.postTrigWindow,exp.caFrameDur)
            #                     ca_pos_post_ref_all = ca_pos_temp_post_ref_all - np.tile(exp.videoFrameNum * exp.numVideoShift,
            #                                                                     (1, len(ca_pos_temp_post_ref_all)))
                                
            #                     for ind in ca_pos_post_ref_all:
            #                         plt.plot((exp.plot_prezero + ind - ca_pos_ref, exp.plot_prezero + ind - ca_pos_ref), (trig_id+ exp.numTrig[0], trig_id + exp.numTrig[0]+ 1),
            #                                 scaley=False, color=colors[post_ind,1], linewidth=1)   
                            
            #             plot_ca = exp.ca_trace_corrected[ca_pos_ref + exp.preTrigWindow: ca_pos_ref + exp.postTrigWindow]
            #             if len(plot_ca)==abs(exp.preTrigWindow)+abs(exp.postTrigWindow):
            #                 ca_traces_right.append(plot_ca)

            # xLabel = np.linspace(exp.preTrigWindow, exp.postTrigWindow, exp.numAnalysisWindow)
            # xLabel[:] = np.nan
            # xSegments = np.linspace(exp.preTrigWindow,exp.postTrigWindow,9)
            # xSegments_integer = [int(i+(-1)*exp.preTrigWindow) for i in xSegments]
            # xSegments_integer[-1]=-1
            # xLabel[xSegments_integer]=xSegments * exp.caFrameDur

            # if exp.max_outlier:
            #     ax = sn.heatmap(ca_traces, cmap="coolwarm",xticklabels=xLabel,vmax=np.percentile(ca_traces_left, 99))

            # elif exp.zscore:
            #     ca_traces = scipy.stats.zscore(ca_traces,axis=1,nan_policy='omit')
            #     ax = sn.heatmap(ca_traces, cmap="coolwarm",xticklabels=xLabel)
            # else:
            #     ax = sn.heatmap(ca_traces, cmap="coolwarm",xticklabels=xLabel) #scipy.stats.zscore(ca_raw_traces,axis=1,nan_policy='omit')

            # ax.set_title("component_"+str(i))
            # plt.savefig(exp.plotSavingFolder+os.sep+"heatmap"+str(i)+'.eps',format='eps')
            # plt.close()


            # fig = plt.figure()
            
            # ax = fig.add_subplot(1,1,1)
            # tVector = np.linspace(exp.preTrigWindow,exp.postTrigWindow,exp.numAnalysisWindow) * exp.caFrameDur

            # ca_mean_traces_left = np.mean(ca_traces_left,axis=0)
            # sem_left = scipy.stats.sem(ca_traces_left,axis=0)
            # ca_mean_traces_left = exp.interpolateNaN(ca_mean_traces_left)
            # sem_left = exp.interpolateNaN(sem_left)

            # ca_mean_traces_right = np.mean(ca_traces_right,axis=0)
            # sem_right = scipy.stats.sem(ca_traces_right,axis=0)
            # ca_mean_traces_right = exp.interpolateNaN(ca_mean_traces_right)
            # sem_right = exp.interpolateNaN(sem_right)

            # ccc_left = [[tVector[-1 - i], ca_mean_traces_left[-1 - i] - sem_left[-1 - i]] for i in range(len(tVector))]
            # ccc_left = ccc_left + [[tVector[i], ca_mean_traces_left[i] + sem_left[i]] for i in range(len(tVector))]
            # ccc_right = [[tVector[-1 - i], ca_mean_traces_right[-1 - i] - sem_right[-1 - i]] for i in range(len(tVector))]
            # ccc_right = ccc_right + [[tVector[i], ca_mean_traces_right[i] + sem_right[i]] for i in range(len(tVector))]
            # pp = plt.Polygon(ccc_left,color='orange')
            # ax.plot(tVector,ca_mean_traces_left,color='r')
            # ax.plot(np.zeros(10),np.linspace(np.min(ca_mean_traces_left),np.max(ca_mean_traces_left),10),color='k')
            # ax.add_patch(pp)

            # pp = plt.Polygon(ccc_right,color='springgreen')
            # ax.plot(tVector,ca_mean_traces_right,color='g')
            # ax.plot(np.zeros(10),np.linspace(np.min(ca_mean_traces_right),np.max(ca_mean_traces_right),10),color='k')
            # ax.add_patch(pp)

            # print("plotting "+str(i))
            # plt.xlabel("Time(s)")
            # plt.ylabel("deltaF/F")
            # ax.set_title("component_"+str(i))
            # plt.legend(["left", "right"])
            # plt.savefig(exp.plotSavingFolder+os.sep+str(i)+'.eps',format='eps')
            # plt.close()



        ca_raw_traces = []
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        if exp.sorted:
            triggers_normal = []
            for trig_id in range(exp.numTrig[0]):
                try:
                    trigger= exp.findTriggerPosition_single(exp.goodTrialTime[trig_id],2)
                    if trigger:
                        triggers_normal.append(trigger)
                except:
                    print("Analysis window is outside ca2+ data size")
            triggers_normal  = exp.sortTriggers(triggers_normal)
            triggers = triggers_normal
            
            if exp.sham:
                trig_id = 0
                triggers_sham = []
                while trig_id < exp.numTrig[0]:
                    trigger, poke_time = exp.findTriggerPosition_sham(exp.pre['1'][trig_id],1)
                    
                    if not trigger:
                        trig_id += 1
                    else:
                        triggers_sham.append(trigger)
                        while exp.pre['1'][trig_id] <= poke_time:
                            trig_id += 1
                        else:
                            trig_id += 1
                triggers = triggers + triggers_sham
            
            if exp.double_rewards:
                triggers_double = []    
                for trig_id in range(exp.numTrig[0]):
                    trigger= exp.findTriggerPosition_double(exp.goodTrialTime[trig_id],2)
                    if trigger:
                        triggers_double.append(trigger)
                triggers_double = exp.sortTriggers(triggers_double)
                triggers = triggers + triggers_double

            ca_raw_traces = exp.findCaResponsesFromTriggers(triggers)
            exp.plotResponsesWithTriggers(triggers,0,ax)


            
        elif exp.sham:
            trig_id = 0

            triggers = []
            while trig_id < exp.numTrig[0]:
                trigger, poke_time = exp.findTriggerPosition_sham(exp.pre['1'][trig_id],1)
                
                if not trigger:
                    trig_id += 1
                else:
                    triggers.append(trigger)
                    while exp.pre['1'][trig_id] <= poke_time:
                        trig_id += 1
                    else:
                        trig_id += 1
                
            ca_raw_traces = exp.findCaResponsesFromTriggers(triggers)
            exp.plotResponsesWithTriggers(triggers,0,ax)
        elif exp.double_rewards:
            triggers= []
            for trig_id in range(exp.numTrig[0]):
                trigger= exp.findTriggerPosition_double(exp.goodTrialTime[trig_id],2)
                if trigger:
                    triggers.append(trigger)
            ca_raw_traces = exp.findCaResponsesFromTriggers(triggers)
            exp.plotResponsesWithTriggers(triggers,0,ax)
           
        else:
            triggers = []
            for trig_id in range(exp.numTrig[0]):
                try:
                    trigger= exp.findTriggerPositionsForPlot(exp.goodTrialTime[trig_id])
                    if trigger:
                        triggers.append(trigger)
                except:
                    print("Analysis window is outside ca2+ data size")
            
            ca_raw_traces = exp.findCaResponsesFromTriggers(triggers)
            exp.plotResponsesWithTriggers(triggers,0,ax)
        ca_traces = exp.plotHeatmapResponses(ca_raw_traces,ax)
        ax.set_title("component_"+str(i))
        plt.savefig(exp.plotSavingFolder+os.sep+"heatmap"+str(i)+'.eps',format='eps')
        plt.close()

        fig= plt.figure()
        ax = fig.add_subplot(1,1,1)

        exp.plotAverageResponses(ca_traces,ax,1)
        is_modulated = exp.testModulated(ca_traces,triggers)
        print("plotting "+str(i))
        plt.xlabel("Time(s)")
        plt.ylabel("deltaF/F")
        ax.set_title("component_"+str(i)+is_modulated)
        plt.savefig(exp.plotSavingFolder+os.sep+str(i)+'.eps',format='eps')
        plt.close()

        if is_modulated == 'True':
            with open(exp.plotSavingFolder+os.sep+'ca_trace_'+'component_'+str(i)+'.pickle','wb') as f:
                pickle.dump(ca_traces,f)
    print("done")


