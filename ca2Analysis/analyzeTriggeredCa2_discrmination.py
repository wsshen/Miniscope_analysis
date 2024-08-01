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
from analyzeTriggeredCa2_class import Exp
from analyzeTriggeredCa2_plot2 import makePlot

def plot(args):

    exp = Exp()
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

    exp_dict = {}
    args_dict = vars(args)
    for arg in args_dict:
        if args_dict[arg] is not None and arg[-3:] == 'Num':
            exp_dict[arg] = args_dict[arg]
            exp_dict[arg[0:-3]+'Time'] = readTabTxtFile(directory+os.sep+"stim"+str(args_dict[arg])+".txt", 'float')

            if 'Lick' in arg and 'BoutEnd' in args.plot_zero:
                exp_dict[arg[0:-3]+'Time_alt'] = readTabTxtFile(directory+os.sep+"stim"+str(args_dict[arg])+"_alt.txt", 'float')
            if 'lick' in arg and args.add_bout_end:
                exp_dict[arg[0:-3]+'Time_alt'] = readTabTxtFile(directory+os.sep+"stim"+str(args_dict[arg])+"_alt.txt", 'float')

    for el in exp_dict:
        if el[-7:] == 'TrigTime':
            exp_dict[el] = trimmedTriggerTime(exp_dict[el],exp_dict['caTime'],exp.videoFrameNum*exp.numVideoShift,-1)

    exp.plotSavingFolder = args.plot_dir

    with open(directory + os.sep + exp.frameCorrectionFile,'rb') as f:
        exp.frame_correction = pickle.load(f)

    with open(directory + os.sep + exp.cellTracesFile, "r", newline="\n") as readFile:
        exp.data_reader = readFile.readlines()

    exp.valid_idx = readTabTxtFile(directory + os.sep + exp.validIdFile, 'int')
    exp.caTime = exp_dict['caTime']
    exp.caFrameDur = calTempResolution(exp_dict['caTime'])

    if args.plot_zero == 'rightLick':
        exp.goodTrialTime = exp_dict['rightPumpTrigTime'] # indicates when a trial starts
        exp.pre ={
            '1':exp_dict['pokeTrigTime'], # '1' should store the earlist trig time
            '2':exp_dict['rightSoundTrigTime']
        }
        exp.ref = exp_dict['rightLickTrigTime'] # indicates the action when time zero aligns
        exp.post = {
            '1': exp_dict['rightPumpTrigTime'], # indicates the action ensuing the ref
        } 
        exp.numTrig = [len(exp_dict['rightPumpTrigTime'])]
    
    if args.plot_zero == 'rightLickBoutEnd':
        exp.goodTrialTime = exp_dict['rightPumpTrigTime'] # indicates when a trial starts
        exp.pre ={
            '1':exp_dict['pokeTrigTime'], # '1' should store the earlist trig time
            '2':exp_dict['rightSoundTrigTime'],
            '3':exp_dict['rightLickTrigTime']
        }
        exp.ref = exp_dict['rightLickTrigTime_alt'] # indicates the action when time zero aligns
        exp.post = {
            '1': exp_dict['rightPumpTrigTime'], # indicates the action ensuing the ref
        } 
        exp.numTrig = [len(exp_dict['rightPumpTrigTime'])]

    if args.plot_zero == 'leftLick':
        exp.goodTrialTime = exp_dict['leftPumpTrigTime'] # indicates when a trial starts
        exp.pre ={
            '1':exp_dict['pokeTrigTime'],
            '2':exp_dict['leftSoundTrigTime']
        }
        exp.ref = exp_dict['leftLickTrigTime'] # indicates the action when time zero aligns
        exp.post = {
            '1': exp_dict['leftPumpTrigTime'], # indicates the action ensuing the ref
        } 
        exp.numTrig = [len(exp_dict['leftPumpTrigTime'])] 
    
    if args.plot_zero == 'lick':
        if args.sham:
            exp.pre ={
                '1':exp_dict['pokeTrigTime']
            }
            exp.ref = exp_dict['lickTrigTime'] # indicates the action when time zero aligns
            exp.post = {
                '1': exp_dict['pumpTrigTime'], # indicates the action ensuing the ref
            } 
            exp.numTrig = [len(exp_dict['pokeTrigTime'])] 
        elif args.double_rewards:
            exp.goodTrialTime = exp_dict['pumpTrigTime'] 
            exp.pre ={
                '1':exp_dict['pokeTrigTime']
            }
            exp.ref = exp_dict['lickTrigTime'] # indicates the action when time zero aligns
            exp.post = {
                '1': exp_dict['pumpTrigTime'], # indicates the action ensuing the ref
            } 
            exp.numTrig = [len(exp_dict['pumpTrigTime'])] 
        
        else:
            exp.goodTrialTime = exp_dict['pumpTrigTime'] # indicates when a trial starts
            exp.pre ={
                '1':exp_dict['pokeTrigTime']
            }
            exp.ref = exp_dict['lickTrigTime'] # indicates the action when time zero aligns
            exp.post = {
                '1': exp_dict['pumpTrigTime'], # indicates the action ensuing the ref
            } 
            exp.numTrig = [len(exp_dict['pumpTrigTime'])] 
    
    if args.plot_zero == 'leftLickBoutEnd':
        exp.goodTrialTime = exp_dict['leftPumpTrigTime'] # indicates when a trial starts
        exp.pre ={
            '1':exp_dict['pokeTrigTime'], # '1' should store the earlist trig time
            '2':exp_dict['leftSoundTrigTime'],
            '3':exp_dict['leftLickTrigTime']
        }
        exp.ref = exp_dict['leftLickTrigTime_alt']  # indicates the action when time zero aligns
        exp.post = {
            '1': exp_dict['leftPumpTrigTime'], # indicates the action ensuing the ref
        } 
        exp.numTrig = [len(exp_dict['leftPumpTrigTime'])]

    if args.plot_zero == 'leftSound':
        exp.goodTrialTime = exp_dict['leftPumpTrigTime'] # indicates when a trial starts
        exp.pre ={
            '1':exp_dict['pokeTrigTime']
        }
        exp.ref = exp_dict['leftSoundTrigTime'] # indicates the action when time zero aligns

        exp.post = {
            '1': exp_dict['leftPumpTrigTime'], # indicates the action ensuing the ref
            '2': exp_dict['leftLickTrigTime']
        } 
        exp.numTrig = [len(exp_dict['leftPumpTrigTime'])]
    
    if args.plot_zero == 'rightSound':
        exp.goodTrialTime = exp_dict['rightPumpTrigTime'] # indicates when a trial starts
        exp.pre ={
            '1':exp_dict['pokeTrigTime']
    }
        exp.ref = exp_dict['rightSoundTrigTime'] # indicates the action when time zero aligns

        exp.post = {
            '1': exp_dict['rightPumpTrigTime'], # indicates the action ensuing the ref
            '2': exp_dict['rightLickTrigTime']
        } 
        exp.numTrig = [len(exp_dict['rightPumpTrigTime'])]

    if args.plot_zero == 'leftPump':
        exp.goodTrialTime = exp_dict['leftPumpTrigTime']# indicates when a trial starts
        exp.pre ={
            '1':exp_dict['pokeTrigTime'],
            '2':exp_dict['leftSoundTrigTime']
        }
        exp.ref = exp_dict['leftPumpTrigTime'] # indicates the action when time zero aligns

        exp.post = {
            '1': exp_dict['leftLickTrigTime']
        } 
        exp.numTrig = [len(exp_dict['leftPumpTrigTime'])]
    
    if args.plot_zero == 'rightPump':
        exp.goodTrialTime = exp_dict['rightPumpTrigTime'] # indicates when a trial starts
        exp.pre ={
            '1':exp_dict['pokeTrigTime'],
            '2':exp_dict['rightSoundTrigTime']
        }
        exp.ref = exp_dict['rightPumpTrigTime'] # indicates the action when time zero aligns

        exp.post = {
            '1': exp_dict['rightLickTrigTime']
        } 
        exp.numTrig = [len(exp_dict['leftPumpTrigTime'])]

    if args.plot_zero == 'poke':
        if 'rightPumpTrigTime' in exp_dict:
            exp.goodTrialTime = exp_dict['rightPumpTrigTime'] + exp_dict['leftPumpTrigTime']  # indicates when a trial starts
        else:
            exp.goodTrialTime = exp_dict['pumpTrigTime']
        if args.add_bout_end:
            exp.pre = {
                '1':exp_dict['lickTrigTime_alt']
            }
        exp.ref = exp_dict['pokeTrigTime'] # indicates the action when time zero aligns

        if 'rightSoundTrigTime' in exp_dict:
            exp.post = {
                '1': exp_dict['rightLickTrigTime'] + exp_dict['leftLickTrigTime'],
                '2': exp_dict['rightSoundTrigTime'] + exp_dict['leftSoundTrigTime'],
                '3': exp_dict['rightPumpTrigTime'] + exp_dict['leftPumpTrigTime'] 
            } 
            exp.numTrig = [len(exp_dict['leftPumpTrigTime'])  + len(exp_dict['rightPumpTrigTime'])]
        else:
            exp.post = {
                '1':exp_dict['lickTrigTime'],
                '2':exp_dict['pumpTrigTime']
            }
            exp.numTrig = [len(exp_dict['pumpTrigTime']) ]

    if args.plot_zero == 'lickComparison':
        exp.goodTrialTime = {
            'left':exp_dict['leftPumpTrigTime'],
            'right':exp_dict['rightPumpTrigTime']
        }
        exp.pre = {
            '1':{'left':exp_dict['pokeTrigTime'],
                 'right':exp_dict['pokeTrigTime']

            },
            
            '2':{'left':exp_dict['leftSoundTrigTime'],
                 'right':exp_dict['rightSoundTrigTime']
            }
        }
        exp.ref = {
            'left':exp_dict['leftLickTrigTime'],
            'right':exp_dict['rightLickTrigTime']
        }
        exp.post = {
            '1': {'left': exp_dict['leftPumpTrigTime'],
                  'right': exp_dict['rightPumpTrigTime']
            }
        } 

        exp.numTrig = [len(exp_dict['leftPumpTrigTime']),len(exp_dict['rightPumpTrigTime'])]

    if args.plot_zero == 'soundComparison':
        exp.goodTrialTime = {
            'left':exp_dict['leftPumpTrigTime'],
            'right':exp_dict['rightPumpTrigTime']
        }
        exp.pre = {
            '1':exp_dict['pokeTrigTime']
        }
        exp.ref = {
            'left':exp_dict['leftSoundTrigTime'],
            'right':exp_dict['rightSoundTrigTime']
        }
        exp.post = {
            '1': {'left':exp_dict['leftLickTrigTime'],
                     'right':exp_dict['rightLickTrigTime']
            },
            '2': {'left':exp_dict['leftPumpTrigTime'],
                      'right':exp_dict['rightPumpTrigTime']
            }
        } 
        exp.numTrig = [len(exp_dict['leftPumpTrigTime']),len(exp_dict['rightPumpTrigTime'])]

    if args.plot_zero == 'pumpComparison':
        exp.goodTrialTime = {
            'left':exp_dict['leftPumpTrigTime'],
            'right':exp_dict['rightPumpTrigTime']
        }
        exp.pre = {
            '1':exp_dict['pokeTrigTime']
        }
        exp.ref = {
            'left':exp_dict['leftPumpTrigTime'],
            'right':exp_dict['rightPumpTrigTime']
        }
        exp.numTrig = [len(exp_dict['leftPumpTrigTime']),len(exp_dict['rightPumpTrigTime'])]

    exp.preTrigWindow = args.pre_window
    exp.plot_prezero = -1 * exp.preTrigWindow
    exp.postTrigWindow = args.post_window
    exp.analysis_pre = args.analysis_pre
    exp.analysis_post = args.analysis_post

    exp.numAnalysisWindow = -1*exp.preTrigWindow + exp.postTrigWindow
    exp.pokeTimeAdj = 0.03
    exp.sham = args.sham
    exp.double_rewards = args.double_rewards
    exp.sorted = args.sorted
    exp.zscore = args.zscore
    exp.max_outlier = args.max_outlier
    makePlot(exp)


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
    parser.add_argument("--analysis_pre",type=int,default=-15)
    parser.add_argument("--analysis_post",type=int,default=80)
    parser.add_argument("--leftLickTrigNum",type=int)
    parser.add_argument("--rightLickTrigNum",type=int)
    parser.add_argument("--leftSoundTrigNum",type=int)
    parser.add_argument("--rightSoundTrigNum",type=int)
    parser.add_argument("--leftPumpTrigNum",type=int)
    parser.add_argument("--rightPumpTrigNum",type=int)
    parser.add_argument("--pokeTrigNum",type=int)
    parser.add_argument("--lickTrigNum",type=int)
    parser.add_argument("--pumpTrigNum",type=int)
    parser.add_argument("--caNum",type=int)

    parser.add_argument("--zscore",type=bool,default=False)
    parser.add_argument("--max_outlier",type=bool,default=False)
    parser.add_argument("--add_bout_end",type=bool,default=False)
    parser.add_argument("--sham",type=bool,default=False)
    parser.add_argument("--double_rewards",type=bool,default=False)
    parser.add_argument("--sorted",type=bool,default=False)
    # parser.add_argument('--file',type=open,action=LoadFromFile)
    # parser.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)


    data_path = "/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/newcohort_03242024/reward_seeking/7277102232024/poke_lick_sham_day21/16_24_44"
    plot_folder = 'Plots'
    plot_path = os.path.join(data_path,plot_folder)
    os.chdir(plot_path)
    # args = parser.parse_args(['@inputs.txt'])
    # args = parser.parse_args(['--file',os.path.join(plot_path,'inputs.txt')])
    args = parser.parse_known_args()[0]

    # example command line
    # python analyzeTriggeredCa2_discrimination.py -f inputs.txt --plot_zero lickComparison --pre_window -100 --post_window 200 --zscore 
    if args.file:
        with open(args.file,'r') as f:
            parser.parse_known_args(f.read().split(),namespace=args)[0]
    
    if not (os.path.exists(plot_path)):
        os.makedirs(plot_path)
    plot_flags = ''
    if args.zscore:
        plot_flags += 'zscore_'
    if args.sham:
        plot_flags += 'sham_'
    if args.double_rewards:
        plot_flags += 'double_rewards_'
    if args.max_outlier:
        plot_flags += 'max_outlier_'
    if args.add_bout_end:
        plot_flags += 'add_bout_end_'
    if args.sorted:
        plot_flags += 'sorted_'
    plotdir = (
        args.plot_zero
        + "_"
        + plot_flags
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