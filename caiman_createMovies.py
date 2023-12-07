#! /usr/bin python
try:
    get_ipython().magic(u'load_ext autoreload')
    get_ipython().magic(u'autoreload 2')
    get_ipython().magic(u'matplotlib qt')
except:
    pass

import logging
import matplotlib.pyplot as plt
import numpy as np
logging.basicConfig(format="%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",level=logging.DEBUG)
import pickle
import caiman as cm

from caiman.source_extraction import cnmf
from caiman.utils.utils import download_demo
from caiman.utils.visualization import inspect_correlation_pnr,nb_inspect_correlation_pnr
from caiman.source_extraction.cnmf import load_CNMF
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
import cv2
try:
    cv2.setNumThreads(0)
except:
    pass


import os
# dataset dependent parameters
fname_new = '/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/4914002252023/06082023/14_40_57/My_V4_Miniscope/memmap_d1_600_d2_600_d3_1_order_C_frames_1000.mmap'
bord_px=0
Yr,dims,T=cm.load_memmap(fname_new)
images = Yr.T.reshape((T,)+dims,order='F')

print("after memmap loaded")
aaa = load_CNMF('/home/watson/Documents/cnmf_save.hdf5')
print("Done!")
step=14890
path = '/home/watson/Documents/'
baseName = 'results_movie'
suffix = '.avi'
for x in range(0,T,step):
    if x+step>T:
        aaa.estimates.play_movie(images,q_max=99.5,frame_range = slice(x,T),magnification=2,include_bck=False,gain_res=10,bpx=bord_px,save_movie=True,thr=0.5,movie_name=path+os.sep+baseName+'_'+str(x)+'_'+str(T)+suffix)
    else:
        aaa.estimates.play_movie(images,q_max=99.5,frame_range = slice(x,x+step),magnification=2,include_bck=False,gain_res=10,bpx=bord_px,save_movie=True,thr=0.5,movie_name=path+os.sep+baseName+'_'+str(x)+'_'+str(x+step)+suffix)


