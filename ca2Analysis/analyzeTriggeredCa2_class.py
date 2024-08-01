import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import seaborn as sn
import scipy
import os
from random import randrange
import copy
import math
from scipy.stats import t

class Exp:
    def __init__(self) -> None:
        
        self.videoFrameNum = 1000
        self.numVideoShift = 0
        self.validIdFile = "validIdx.txt"
        self.cellTracesFile = "cellTraces_norm.txt"
        self.frameCorrectionFile = "frame_correction_pos.pickle"
        self.colors = np.array([['k','r','g'], # pre, post, ref
                                ['dimgray','orange','lime'],
                                ['silver','darkred','springgreen']])
        #for basic analysis
        # self.analysis_pre = 15 # pre_window frames
        # self.analysis_post = 80 # post_window frames
        self.confidence_level = 0.95

    def findClosestTimeIndex(self,t,tSeries):
        tDiff = [abs(t-i) for i in tSeries]
        nd_tDiff = np.array(tDiff)
        pos = np.argmin(nd_tDiff)
        return pos
    # def findClosestLickIndex(self,t1Series,refT,tSeries):
    #     tDiff = [[i-refT,i] for i in t1Series if i-refT>=0]
    #     nd_tDiff = np.array(tDiff)
    #     lick_pos = np.argmin(nd_tDiff[:,0])
    #     pos = self.findClosestTimeIndex(nd_tDiff[lick_pos,1],tSeries)
    #     return pos

    def findAfterClosestIndex(self,t1Series,refT,tSeries):
        tDiff = [[i-refT,i] for i in t1Series if i-refT>=0]
        if tDiff:
            nd_tDiff = np.array(tDiff)
            lick_pos = np.argmin(nd_tDiff[:,0])
            pos = self.findClosestTimeIndex(nd_tDiff[lick_pos,1],tSeries)
        else:
            print('tDiff is empty!')
            pos = []
        return pos

    def findBeforeClosestIndex(self,t1Series,refT,tSeries):
        tDiff = [[i-refT,i] for i in t1Series if i-refT<=0]
        nd_tDiff = np.array(tDiff)
        lick_pos = np.argmax(nd_tDiff[:,0])
        pos = self.findClosestTimeIndex(nd_tDiff[lick_pos,1],tSeries)
        return pos

    def findAllIndicesPostRef(self,t1Series,refT,tSeries,window,caFrameDur):
        pos=[]
        tDiff = [[i - tSeries[refT], i] for i in t1Series if i-tSeries[refT]>0 and i-tSeries[refT]-window*caFrameDur<0]
        nd_tDiff = np.array(tDiff)
        for i in nd_tDiff:
            pos_temp = self.findClosestTimeIndex(i[1],tSeries)
            pos.append(pos_temp)
        return pos
    
    def findAllIndicesBeforeRef(self,t1Series,refT,tSeries,window,caFrameDur):
        pos=[]
        tDiff = [[i - tSeries[refT], i] for i in t1Series if i-tSeries[refT]<0 and tSeries[refT]- i < window*caFrameDur]
        nd_tDiff = np.array(tDiff)
        for i in nd_tDiff:
            pos_temp = self.findClosestTimeIndex(i[1],tSeries)
            pos.append(pos_temp)
        return pos

    def interpolateNaN(self,x):
        loc = np.argwhere(np.isnan(x))
        for i in loc:
            if i[0]+1 <=len(x)-1:
                x[i[0]] = (x[i[0]-1] + x[i[0]+1])/2
            else:
                x[i[0]] = x[i[0]-1]
        return x

    def findTriggerPositionsForPlot(self,trialTime=[],side_key=''):

        trigger={}

        if hasattr(self,'pre'):
            if side_key:
                ca_pos_temp_pre = self.findBeforeClosestIndex(self.pre['1'][side_key], trialTime, self.caTime)
                ca_pos_temp_ref = self.findAfterClosestIndex(self.ref[side_key], self.caTime[ca_pos_temp_pre], self.caTime) #caTime[ca_pos_temp_pre_ref]
                ca_pos_temp_ref_all = self.findAllIndicesPostRef(self.ref[side_key], ca_pos_temp_pre, self.caTime, self.postTrigWindow,self.caFrameDur)

            else:
                ca_pos_temp_pre = self.findBeforeClosestIndex(self.pre['1'], trialTime, self.caTime)
                ca_pos_temp_ref = self.findAfterClosestIndex(self.ref, self.caTime[ca_pos_temp_pre], self.caTime) #caTime[ca_pos_temp_pre_ref]
                ca_pos_temp_ref_all = self.findAllIndicesPostRef(self.ref, ca_pos_temp_pre, self.caTime, self.postTrigWindow,self.caFrameDur)
            
            ca_pos_ref = ca_pos_temp_ref - self.videoFrameNum * self.numVideoShift
            ca_pos_ref_all = ca_pos_temp_ref_all - np.tile(self.videoFrameNum * self.numVideoShift,
                                                            (1, len(ca_pos_temp_ref_all)))
            trigger['ref'] = ca_pos_ref
            trigger['ref_all'] = ca_pos_ref_all 
            trigger['pre'] = {}
            trigger['pre_all'] = {}
            for pre_ind in range(len(self.pre)):
                if side_key:
                    ca_pos_temp_pre_ref = self.findBeforeClosestIndex(self.pre[str(pre_ind+1)][side_key], trialTime, self.caTime)
                    ca_pos_temp_pre_ref_all = self.findAllIndicesPostRef(self.pre[str(pre_ind+1)][side_key], ca_pos_temp_pre, self.caTime, self.postTrigWindow,self.caFrameDur)
                else:
                    ca_pos_temp_pre_ref = self.findBeforeClosestIndex(self.pre[str(pre_ind+1)], trialTime, self.caTime)
                    ca_pos_temp_pre_ref_all = self.findAllIndicesPostRef(self.pre[str(pre_ind+1)], ca_pos_temp_pre, self.caTime, self.postTrigWindow,self.caFrameDur)

                ca_pos_pre_ref = ca_pos_temp_pre_ref - self.videoFrameNum * self.numVideoShift
                ca_pos_pre_ref_all = ca_pos_temp_pre_ref_all - np.tile(self.videoFrameNum * self.numVideoShift,
                                                        (1, len(ca_pos_temp_pre_ref_all)))
                trigger['pre'][pre_ind] = ca_pos_pre_ref
                trigger['pre_all'][pre_ind] = ca_pos_pre_ref_all

        else:
            ca_pos_temp_ref = self.findBeforeClosestIndex(self.ref, trialTime, self.caTime)
            ca_pos_ref = ca_pos_temp_ref - self.videoFrameNum * self.numVideoShift
            ca_pos_temp_ref_all = self.findAllIndicesPostRef(self.ref, ca_pos_temp_ref, self.caTime, self.postTrigWindow,self.caFrameDur)
            ca_pos_ref_all = ca_pos_temp_ref_all - np.tile(self.videoFrameNum * self.numVideoShift,
                                                            (1, len(ca_pos_temp_ref_all)))
            trigger['ref'] = ca_pos_ref
            trigger['ref_all'] = ca_pos_ref_all 

        if hasattr(self,'post'):
            trigger['post'] = {}
            trigger['post_all'] = {}
            for post_ind in range(len(self.post)):
                if side_key:
                    ca_pos_temp_post_ref = self.findAfterClosestIndex(self.post[str(post_ind+1)][side_key], self.caTime[ca_pos_temp_ref], self.caTime) #caTime[ca_pos_temp_pre_ref]
                    ca_pos_temp_post_ref_all = self.findAllIndicesPostRef(self.post[str(post_ind+1)][side_key], ca_pos_temp_ref, self.caTime, self.postTrigWindow,self.caFrameDur)
                else:
                    ca_pos_temp_post_ref = self.findAfterClosestIndex(self.post[str(post_ind+1)], self.caTime[ca_pos_temp_ref], self.caTime) #caTime[ca_pos_temp_pre_ref]
                    ca_pos_temp_post_ref_all = self.findAllIndicesPostRef(self.post[str(post_ind+1)], ca_pos_temp_ref, self.caTime, self.postTrigWindow,self.caFrameDur)

                ca_pos_post_ref = ca_pos_temp_post_ref - self.videoFrameNum * self.numVideoShift
                ca_pos_post_ref_all = ca_pos_temp_post_ref_all - np.tile(self.videoFrameNum * self.numVideoShift,
                                                                (1, len(ca_pos_temp_post_ref_all)))

                trigger['post'][post_ind] = ca_pos_post_ref
                trigger['post_all'][post_ind] = ca_pos_post_ref_all
        # plot_ca = self.ca_trace_corrected[ca_pos_ref + self.preTrigWindow: ca_pos_ref + self.postTrigWindow]

        return trigger
    
    def plotResponsesWithTriggers(self,triggers,id_base,ax):
        
        for trig_id,trigger in enumerate(triggers):
            if 'pre' in trigger:
                for pre_ind in range(len(trigger['pre'])):
                    ax.plot((self.plot_prezero + trigger['pre'][pre_ind] - trigger['ref'], self.plot_prezero + trigger['pre'][pre_ind] - trigger['ref']),
                            (trig_id + id_base, trig_id + id_base + 1), scaley=False, color=self.colors[pre_ind,0])
                    for ind in trigger['pre_all'][pre_ind]:
                        ax.plot((self.plot_prezero + ind - trigger['ref'], self.plot_prezero + ind - trigger['ref']), (trig_id+id_base, trig_id+id_base + 1),
                                        scaley=False, color=self.colors[pre_ind,0], linewidth=1)
            
            for ind in trigger['ref_all']:
                ax.plot((self.plot_prezero + ind - trigger['ref'], self.plot_prezero + ind - trigger['ref']), (trig_id+id_base, trig_id+id_base + 1), 
                                            scaley=False, color=self.colors[0,2], linewidth=1)
                
            if 'post' in trigger:
                for post_ind in range(len(trigger['post'])):
                    # ax.plot((self.plot_prezero + trigger['post'][post_ind] - trigger['ref'], self.plot_prezero + trigger['post'][post_ind] - trigger['ref']),
                    #          (trig_id, trig_id + 1), scaley=False, color=self.colors[post_ind,0])
                    for ind in trigger['post_all'][post_ind]:
                        ax.plot((self.plot_prezero + ind - trigger['ref'], self.plot_prezero + ind - trigger['ref']), (trig_id+id_base, trig_id+id_base + 1),
                                        scaley=False, color=self.colors[post_ind,1], linewidth=1)
        

    def plotHeatmapResponses(self,ca_traces,ax):
        xLabel = np.linspace(self.preTrigWindow, self.postTrigWindow, self.numAnalysisWindow)
        xLabel[:] = np.nan
        xSegments = np.linspace(self.preTrigWindow,self.postTrigWindow,9)
        xSegments_integer = [int(ii+(-1)*self.preTrigWindow) for ii in xSegments]
        xSegments_integer[-1]=-1
        xLabel[xSegments_integer]=xSegments * self.caFrameDur
        

        if self.max_outlier:
            ax = sn.heatmap(ca_traces, cmap="coolwarm",xticklabels=xLabel,vmax=np.percentile(ca_traces, 95),ax=ax)
        elif self.zscore:
            ca_traces = scipy.stats.zscore(ca_traces,axis=1,nan_policy='omit')
            ax = sn.heatmap(ca_traces, cmap="coolwarm",xticklabels=xLabel)
        else:
            ax = sn.heatmap(ca_traces, cmap="coolwarm",xticklabels=xLabel,ax=ax) #scipy.stats.zscore(ca_raw_traces,axis=1,nan_policy='omit')
        return ca_traces
    
    def plotAverageResponses(self,ca_raw_traces,ax,color_set):

        tVector = np.linspace(self.preTrigWindow,self.postTrigWindow,self.numAnalysisWindow) * self.caFrameDur
        ca_mean_traces = np.mean(ca_raw_traces,axis=0)
        sem = scipy.stats.sem(ca_raw_traces,axis=0)

        ca_mean_traces = self.interpolateNaN(ca_mean_traces)
        sem = self.interpolateNaN(sem)
        ccc = [[tVector[-1 - ii], ca_mean_traces[-1 - ii] - sem[-1 - ii]] for ii in range(len(tVector))]
        ccc = ccc + [[tVector[ii], ca_mean_traces[ii] + sem[ii]] for ii in range(len(tVector))]
        pp = plt.Polygon(ccc,color=self.colors[1][color_set])
        ax.plot(tVector,ca_mean_traces,color=self.colors[0][color_set])
        ax.plot(np.zeros(10),np.linspace(np.min(ca_mean_traces),np.max(ca_mean_traces),10),color='k')
        ax.plot(np.ones(10)*(self.analysis_pre*self.caFrameDur),np.linspace(np.min(ca_mean_traces),np.max(ca_mean_traces),10),color='g')
        ax.plot(np.ones(10)*(self.analysis_post*self.caFrameDur),np.linspace(np.min(ca_mean_traces),np.max(ca_mean_traces),10),color='g')

        ax.add_patch(pp)

    def findTriggerPosition_sham(self,trialTime=[],sham_gap=1):
        trigger={}

        ca_pos_temp_ref = self.findAfterClosestIndex(self.ref, trialTime, self.caTime) # find the closest lick time after the poke 

        ca_pos_temp_pre_ref = self.findBeforeClosestIndex(self.pre['1'], self.caTime[ca_pos_temp_ref], self.caTime) # find the closest poke time before the lick
        ca_pos_temp_post_pre = self.findAfterClosestIndex(self.post['1'], self.caTime[ca_pos_temp_pre_ref], self.caTime) # find the closest pump time after the poke
        
        if ca_pos_temp_post_pre - ca_pos_temp_ref > round(sham_gap/self.caFrameDur): #and ca_pos_temp_ref - ca_pos_temp_pre_ref > round(1/self.caFrameDur):
            
            ca_pos_temp_ref_all = self.findAllIndicesPostRef(self.ref, ca_pos_temp_pre_ref, self.caTime, self.postTrigWindow,self.caFrameDur)
            ca_pos_ref_all = ca_pos_temp_ref_all - np.tile(self.videoFrameNum * self.numVideoShift,
                                                                (1, len(ca_pos_temp_ref_all)))
            ca_pos_ref = ca_pos_temp_ref - self.videoFrameNum * self.numVideoShift
            ca_test = self.ca_trace_corrected[ca_pos_ref + self.preTrigWindow: ca_pos_ref + self.postTrigWindow]
            if not len(ca_test)==abs(self.preTrigWindow)+abs(self.postTrigWindow):
                return trigger,self.caTime[ca_pos_temp_pre_ref]
            trigger['ref'] = ca_pos_ref
            trigger['ref_all'] = ca_pos_ref_all
            
            trigger['pre'] = {}
            trigger['pre_all'] = {}
            for pre_ind in range(len(self.pre)):
                ca_pos_temp_pre_ref_all = self.findAllIndicesPostRef(self.pre[str(pre_ind+1)],ca_pos_temp_pre_ref,self.caTime,self.postTrigWindow,self.caFrameDur)
                ca_pos_pre_ref_all = ca_pos_temp_pre_ref_all - np.tile(self.videoFrameNum * self.numVideoShift,
                                                                (1, len(ca_pos_temp_pre_ref_all)))
                ca_pos_pre_ref = ca_pos_temp_pre_ref - self.videoFrameNum * self.numVideoShift

                trigger['pre'][pre_ind] = ca_pos_pre_ref
                trigger['pre_all'][pre_ind] = ca_pos_pre_ref_all
            
            trigger['post'] = {}
            trigger['post_all'] = {}
            for post_ind in range(len(self.post)):

                ca_pos_temp_post_pre_all = self.findAllIndicesPostRef(self.post[str(post_ind+1)], ca_pos_temp_ref, self.caTime, self.postTrigWindow,self.caFrameDur)

                ca_pos_post_pre = ca_pos_temp_post_pre - self.videoFrameNum * self.numVideoShift
                ca_pos_post_pre_all = ca_pos_temp_post_pre_all - np.tile(self.videoFrameNum * self.numVideoShift,
                                                                (1, len(ca_pos_temp_post_pre_all)))

                trigger['post'][post_ind] = ca_pos_post_pre
                trigger['post_all'][post_ind] = ca_pos_post_pre_all

        else:
            ca_pos_temp_pre_ref = 0

        return trigger,self.caTime[ca_pos_temp_pre_ref]
    
    def findTriggerPosition_double(self,trialTime=[],double_gap=2):
        trigger={}
        ca_pos_temp_pre = self.findBeforeClosestIndex(self.pre['1'], trialTime, self.caTime) 
        ca_pos_temp_ref = self.findAfterClosestIndex(self.ref, self.caTime[ca_pos_temp_pre], self.caTime) 
        ca_pos_temp_post_ref = self.findAfterClosestIndex(self.post['1'], self.caTime[ca_pos_temp_ref], self.caTime)      
        ca_pos_temp_post_ref_all = self.findAllIndicesPostRef(self.post['1'], ca_pos_temp_ref, self.caTime, self.postTrigWindow,self.caFrameDur)
        
        if len(ca_pos_temp_post_ref_all)>1 and (ca_pos_temp_post_ref_all[1] - ca_pos_temp_post_ref_all[0] )< round(double_gap/self.caFrameDur):
            
            ca_pos_temp_ref_all = self.findAllIndicesPostRef(self.ref, ca_pos_temp_pre, self.caTime, self.postTrigWindow,self.caFrameDur)
            ca_pos_ref_all = ca_pos_temp_ref_all - np.tile(self.videoFrameNum * self.numVideoShift,
                                                                (1, len(ca_pos_temp_ref_all)))
            ca_pos_ref = ca_pos_temp_ref - self.videoFrameNum * self.numVideoShift

            trigger['ref'] = ca_pos_ref
            trigger['ref_all'] = ca_pos_ref_all
            
            trigger['pre'] = {}
            trigger['pre_all'] = {}
            for pre_ind in range(len(self.pre)):
                ca_pos_temp_pre_ref_all = self.findAllIndicesPostRef(self.pre[str(pre_ind+1)],ca_pos_temp_pre,self.caTime,self.postTrigWindow,self.caFrameDur)
                ca_pos_pre_ref_all = ca_pos_temp_pre_ref_all - np.tile(self.videoFrameNum * self.numVideoShift,
                                                                (1, len(ca_pos_temp_pre_ref_all)))
                ca_pos_pre_ref = ca_pos_temp_pre - self.videoFrameNum * self.numVideoShift

                trigger['pre'][pre_ind] = ca_pos_pre_ref
                trigger['pre_all'][pre_ind] = ca_pos_pre_ref_all
            
            trigger['post'] = {}
            trigger['post_all'] = {}
            for post_ind in range(len(self.post)):

                ca_pos_temp_post_pre_all = self.findAllIndicesPostRef(self.post[str(post_ind+1)], ca_pos_temp_ref, self.caTime, self.postTrigWindow,self.caFrameDur)

                ca_pos_post_pre = ca_pos_temp_post_ref - self.videoFrameNum * self.numVideoShift
                ca_pos_post_pre_all = ca_pos_temp_post_pre_all - np.tile(self.videoFrameNum * self.numVideoShift,
                                                                (1, len(ca_pos_temp_post_pre_all)))

                trigger['post'][post_ind] = ca_pos_post_pre
                trigger['post_all'][post_ind] = ca_pos_post_pre_all


        return trigger
    def findTriggerPosition_single(self,trialTime=[],single_gap=2):
        trigger={}
        ca_pos_temp_pre = self.findBeforeClosestIndex(self.pre['1'], trialTime, self.caTime) 
        ca_pos_temp_ref = self.findAfterClosestIndex(self.ref, self.caTime[ca_pos_temp_pre], self.caTime) 
        ca_pos_temp_post_ref = self.findAfterClosestIndex(self.post['1'], self.caTime[ca_pos_temp_ref], self.caTime)      
        ca_pos_temp_post_ref_all = self.findAllIndicesPostRef(self.post['1'], ca_pos_temp_ref, self.caTime, self.postTrigWindow,self.caFrameDur)
        
        if len(ca_pos_temp_post_ref_all) == 1 or (len(ca_pos_temp_post_ref_all)>1 and (ca_pos_temp_post_ref_all[1] - ca_pos_temp_post_ref_all[0] )> round(single_gap/self.caFrameDur)):
            
            ca_pos_temp_ref_all = self.findAllIndicesPostRef(self.ref, ca_pos_temp_pre, self.caTime, self.postTrigWindow,self.caFrameDur)
            ca_pos_ref_all = ca_pos_temp_ref_all - np.tile(self.videoFrameNum * self.numVideoShift,
                                                                (1, len(ca_pos_temp_ref_all)))
            ca_pos_ref = ca_pos_temp_ref - self.videoFrameNum * self.numVideoShift

            trigger['ref'] = ca_pos_ref
            trigger['ref_all'] = ca_pos_ref_all
            
            trigger['pre'] = {}
            trigger['pre_all'] = {}
            for pre_ind in range(len(self.pre)):
                ca_pos_temp_pre_ref_all = self.findAllIndicesPostRef(self.pre[str(pre_ind+1)],ca_pos_temp_pre,self.caTime,self.postTrigWindow,self.caFrameDur)
                ca_pos_pre_ref_all = ca_pos_temp_pre_ref_all - np.tile(self.videoFrameNum * self.numVideoShift,
                                                                (1, len(ca_pos_temp_pre_ref_all)))
                ca_pos_pre_ref = ca_pos_temp_pre - self.videoFrameNum * self.numVideoShift

                trigger['pre'][pre_ind] = ca_pos_pre_ref
                trigger['pre_all'][pre_ind] = ca_pos_pre_ref_all
            
            trigger['post'] = {}
            trigger['post_all'] = {}
            for post_ind in range(len(self.post)):

                ca_pos_temp_post_pre_all = self.findAllIndicesPostRef(self.post[str(post_ind+1)], ca_pos_temp_ref, self.caTime, self.postTrigWindow,self.caFrameDur)

                ca_pos_post_pre = ca_pos_temp_post_ref - self.videoFrameNum * self.numVideoShift
                ca_pos_post_pre_all = ca_pos_temp_post_pre_all - np.tile(self.videoFrameNum * self.numVideoShift,
                                                                (1, len(ca_pos_temp_post_pre_all)))

                trigger['post'][post_ind] = ca_pos_post_pre
                trigger['post_all'][post_ind] = ca_pos_post_pre_all


        return trigger
    def findCaResponsesFromTriggers(self,triggers):
        # if self.zscore:
        #     ca_traces_for_plot = scipy.stats.zscore(self.ca_trace_corrected)
        # else:
        ca_traces_for_plot = np.array(self.ca_trace_corrected)
        ca_raw_traces = []
        for trig in triggers:

            plot_ca = ca_traces_for_plot[trig['ref'] + self.preTrigWindow: trig['ref'] + self.postTrigWindow]
            if len(plot_ca)==abs(self.preTrigWindow)+abs(self.postTrigWindow):
                ca_raw_traces.append(plot_ca)
        return np.array(ca_raw_traces)
    
    def sortTriggers(self,triggers):

        triggers_sorted = sorted(triggers,key=lambda trigger:trigger['post'][0] - trigger['ref'])
        return triggers_sorted


    def modulation_bootstrap(self,trace,n,iteration=1000,threshold=99):
        # each bootstrapped response is selected from the un-zscored trace and then zscored
        maxima = []
        for i in range(iteration):
            trials_per_iteration = []
            for j in range(n):
                ind = randrange(-1*self.preTrigWindow,len(trace)-self.postTrigWindow,1)
                single_trial = trace[ind+self.preTrigWindow:ind+self.postTrigWindow]
                single_trial = scipy.stats.zscore(single_trial)
                trials_per_iteration.append(single_trial)
                # trials_per_iteration = np.array(trials_per_iteration)
            trials_per_iteration_mean = np.mean(trials_per_iteration,axis=0)

            single_trial_maximum = np.max(trials_per_iteration_mean[-1*self.preTrigWindow+self.analysis_pre:-1*self.preTrigWindow+self.analysis_post])
            maxima.append(single_trial_maximum)
        maxima = np.array(maxima)
        maxima_threshold = np.percentile(maxima,threshold)

        return maxima_threshold
    
    def testModulated(self,ca_traces,triggers):
        # 99 percentile of boot strapped maxima
        ca_mean_traces = np.mean(ca_traces,axis=0)
        maximum = np.max(ca_mean_traces[-1*self.preTrigWindow+self.analysis_pre:-1*self.preTrigWindow+self.analysis_post])
        #zscore_whole_trace = scipy.stats.zscore(self.ca_trace_corrected)
        threshold = self.modulation_bootstrap(self.ca_trace_corrected,ca_traces.shape[0])
        print(maximum,threshold)

        # the average response outside the restricted time window
        # the average response is calculated from the zscored ca trace and then exluding the responses in the restricted time windows
        zscore_whole_trace = scipy.stats.zscore(self.ca_trace_corrected)
        ca_traces_copy = np.array(copy.copy(zscore_whole_trace))
        ca_traces_rest = []
        for trig in triggers:
            ca_traces_copy[trig['ref'] + self.preTrigWindow: trig['ref'] + self.postTrigWindow] = math.nan
                
        ca_traces_rest = ca_traces_copy[~np.isnan(ca_traces_copy)]
        #ca_traces_rest_zscore = scipy.stats.zscore(ca_traces_rest)
        ca_traces_rest_mean = np.mean(ca_traces_rest,axis=0)
        print("ca_traces_rest_mean is",ca_traces_rest_mean)
        # the upper limit of 95% confidence interval
        maxima = np.max(ca_traces,axis=0)
        confidence_interval = t.interval(self.confidence_level,maxima.shape[0]-1,0, np.std(maxima, ddof=1)/np.sqrt(maxima.shape[0]))
        
        print("the upper limit of the confidence interval is:", confidence_interval[1])
        #
        if maximum>threshold and maximum>ca_traces_rest_mean and maximum>confidence_interval[1]:
            is_modulated = 'True'
        else:
            is_modulated = 'False'

        return is_modulated