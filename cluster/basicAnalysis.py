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
import os

from caiman.source_extraction import cnmf
from caiman.utils.utils import download_demo
from caiman.utils.visualization import inspect_correlation_pnr,nb_inspect_correlation_pnr
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
import cv2
try:
    cv2.setNumThreads(0)
except:
    pass
#import bokeh.plotting as bpl
#import holoviews as hv
import csv
import glob

filePath = '/om2/user/shenwang/miniscope/72769/reward_seeking/day10'
fileName = 'output_rescaled'
fileSuffix = ''
fileFormat = '.avi'
# filename to be processed
fnames = [filePath + os.sep + fileName + fileSuffix + fileFormat]
do_cnmfe = False

print("entering dview")
if 'dview' in locals():
    cm.stop_server(dview=dview)
    print("inside dview")
c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)

print("n_processes is:")
print(n_processes)

# dataset dependent parameters
frate = 30                       # movie frame rate
decay_time = 0.8                 # length of a typical transient in seconds
downscale = 2
# motion correction parameters
motion_correct = True   # flag for performing motion correction
pw_rigid = True         # flag for performing piecewise-rigid motion correction (otherwise just rigid)
gSig_filt = tuple(i//downscale for i in (5, 5))       # size of high pass spatial filtering, used in 1p data
max_shifts = tuple(i//downscale for i in (30,30))      # maximum allowed rigid shift
strides = tuple(i//downscale for i in (64, 64))       # start a new patch for pw-rigid motion correction every x pixels
overlaps = tuple(i//downscale for i in (32, 32))      # overlap between pathes (size of patch strides+overlaps)
max_deviation_rigid = 15//downscale  # maximum deviation allowed for patch with respect to rigid shifts
border_nan = 'copy'      # replicate values along the boundaries

mc_dict = {
    'fnames': fnames,
    'fr': frate,
    'decay_time': decay_time,
    'pw_rigid': pw_rigid,
    'max_shifts': max_shifts,
    'gSig_filt': gSig_filt,
    'strides': strides,
    'overlaps': overlaps,
    'max_deviation_rigid': max_deviation_rigid,
    'border_nan': border_nan,
}

opts = params.CNMFParams(params_dict=mc_dict)
baseName = 'memmap_'
if motion_correct:
    mc = MotionCorrect(fnames,dview=dview,**opts.get_group('motion'))
    mc.motion_correct(save_movie=True)
    fname_mc = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig
    if pw_rigid:
        bored_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                      np.max(np.abs(mc.y_shifts_els)))).astype(int)
    else:
        bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(int)
        print('PW_rigid is false')
        # plt.subplot(1,2,1);
        # plt.imshow(mc.total_template_rig)
        # plt.subplot(1,2,2);
        # plt.plot(mc.shifts_rig)
        # plt.legend(['x shifts','y shifts'])
        # plt.xlabel('frames')
        # plt.ylabel('pixels')
    bord_px=0 if border_nan == 'copy' else bord_px
    print(bord_px)
    fname_new = cm.save_memmap(fname_mc,base_name=baseName,order='C',border_to_0=bord_px)
else:
    #fname_new = cm.save_memmap(fnames,base_name=baseName,order='C',border_to_0=0,dview=dview)
    print("Motion correction done!")
    fname_new_list = glob.glob(filePath + os.sep + 'memmap_d1*_.mmap')
    fname_new = fname_new_list[0]    
    bord_px = 0 

Yr,dims,T=cm.load_memmap(fname_new)
images = Yr.T.reshape((T,)+dims,order='F')
del Yr
R = cm.load(fname_new)
mR = R.mean(0)
Ain = cm.base.rois.extract_binary_masks_from_structural_channel(mR,gSig=7,expand_method='dilation')[0]
print("after memmap loaded")
# parameters for source extraction and deconvolution
p = 1               # order of the autoregressive system
K = None            # upper bound on number of components per patch, in general None
gSig = tuple(i//downscale for i in (5, 5))       # gaussian width of a 2D gaussian kernel, which approximates a neuron
gSiz = tuple(i//downscale for i in (21, 21))     # average diameter of a neuron, in general 4*gSig+1
#Ain = None          # possibility to seed with predetermined binary masks
merge_thr = .7      # merging threshold, max correlation allowed
rf = 40//downscale             # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
stride_cnmf = 20//downscale    # amount of overlap between the patches in pixels
#                     (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
tsub = 2            # downsampling factor in time for initialization,
#                     increase if you have memory problems
ssub = 1            # downsampling factor in space for initialization,
#                     increase if you have memory problems
#                     you can pass them here as boolean vectors
low_rank_background = None  # None leaves background of each patch intact,
#                     True performs global low-rank approximation if gnb>0
gnb = 0             # number of background components (rank) if positive,
#                     else exact ring model with following settings
#                         gnb= 0: Return background as b and W
#                         gnb=-1: Return full rank background B
#                         gnb<-1: Don't return background
nb_patch = 0        # number of background components (rank) per patch if gnb>0,
#                     else it is set automatically
min_corr = .5 # min peak value from correlation image
min_pnr = 10        # min peak to noise ration from PNR image
ssub_B = 2          # additional downsampling factor in space for background
ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor
opts.change_params(params_dict={'method_init':'corr_pnr',
                               'K':K,
                               'gSig':gSig,
                               'gSiz':gSiz,
                               'merge_thr':merge_thr,
                               'rf':rf,
                               'stride':stride_cnmf,
                               'tsub':tsub,
                               'p':p,
                               'ssub':ssub,
                               'only_init':True,
                               'nb':gnb,
                               'nb_patch':nb_patch,
                               'method_deconvolution':'oasis',
                               'low_rank_background':low_rank_background,
                               'update_background_components':True,
                               'min_corr':min_corr,
                               'min_pnr':min_pnr,
                               'normalize_init':False,
                               'center_psf':True,
                               'ssub_B':ssub_B,
                               'ring_size_factor':ring_size_factor,
                               'del_duplicates':True,
			       'memory_fact':10,
                               'border_pix':bord_px})

cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1],gSig=gSig[0],swap_dim=False)
# nb_inspect_correlation_pnr(cn_filter,pnr)
import pickle
with open(filePath + os.sep + fileName + fileSuffix + '.pickle','wb') as f:
    pickle.dump([cn_filter,pnr],f)
print(min_corr)
print(min_pnr)
print(n_processes)
if do_cnmfe:
    cnm=cnmf.CNMF(n_processes=n_processes,dview=dview,Ain=Ain,params=opts)
    cnm.fit(images)
    min_SNR = 10            # adaptive way to set threshold on the transient size
    r_values_min = 0.85    # threshold on space consistency (if you lower more components
    #                        will be accepted, potentially with worst quality)
    cnm.params.set('quality', {'min_SNR': min_SNR,
			       'rval_thr': r_values_min,
			       'use_cnn': False})
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

    print(' ***** ')
    print('Number of total components: ', len(cnm.estimates.C))
    print('Number of accepted components: ', len(cnm.estimates.idx_components))
    valid_idx = cnm.estimates.idx_components
    cellTraces = cnm.estimates.C

    #traces_sz = cellTraces.shape
    #with open("/om2/user/shenwang/miniscope/49140/cellTraces_0_36.txt", 'w', newline="\n") as f:
    #    f_writer = csv.writer(f, delimiter='\t')
    #    for i in range(traces_sz[0]):
    #        f_writer.writerow(cellTraces[i, :])
    #with open("/om2/user/shenwang/miniscope/49140/validIdx_0_36.txt", 'w', newline="") as f:
    #    f_writer = csv.writer(f, delimiter='\t')
    #    f_writer.writerow(valid_idx)
    #
    cnm.save(filePath + os.sep+ fileName + '_cnmf_save' + fileSuffix + '.hdf5')
    print("Saving hdf5 done.")
    #savedMovie = cnm.estimates.play_movie(images,q_max=99.5,magnification=2,include_bck=False,gain_res=10,bpx=bord_px,save_movie=True,thr=0.5,movie_name='/om2/user/shenwang/miniscope/results_movie_41_80.avi')
    cm.stop_server(dview=dview)

print("Done!")
