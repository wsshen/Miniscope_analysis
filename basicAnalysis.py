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

import caiman as cm

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
import bokeh.plotting as bpl
import holoviews as hv
# bpl.output_notebook()
# hv.notebook_extension('bokeh')

fnames = ['/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/4914202252023/05192023/10_48_49/My_V4_Miniscope/90.avi']
# fnames = [download_demo(fnames[0])]
print("entering dview")
if 'dview' in locals():
    cm.stop_server(dview=None)
    print("exiting dview")
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
print(n_processes)
#dataset dependent parameters
frate = 30
decay_time = 0.8

# motion correction parameters
motion_correct = True
pw_rigid = False
gSig_filt = (3,3)
max_shifts = (5,5)
strides = (48,48)
overlaps = (24,24)
max_deviation_rigid = 3
border_nan = 'copy'

mc_dict = {
    'fnames': fnames,
    'fr': frate,
    'decay_time': decay_time,
    'pw_rigid': pw_rigid,
    'gSig_filt':gSig_filt,
    'max_shifts':max_shifts,
    'strides': strides,
    'overlaps': overlaps,
    'max_deviation_rigid': max_deviation_rigid,
    'border_nan': border_nan
}

opts=params.CNMFParams(params_dict=mc_dict)
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
    fname_new = cm.save_memmap(fname_mc,base_name='memmap_',order='C',border_to_0=bord_px)
else:
    fname_new = cm.save_memmap(fnames,base_name='memmap_',order='C',border_to_0=0,dview=None)

# bord_px =0
# fname_new = '/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/4914202252023/05192023/10_48_49/My_V4_Miniscope/memmap_d1_600_d2_600_d3_1_order_C_frames_1000.mmap'
Yr,dims,T=cm.load_memmap(fname_new)
images = Yr.T.reshape((T,)+dims,order='F')
# parameters for source extraction and deconvolution
p=1
K=None
gSig = (3,3)
gSiz = (13,13)
Ain = None
merge_thr = .7
rf=40
stride_cnmf = 20
tsub = 2
ssub=1
low_rank_background = None
gnb=0
nb_patch =0
min_corr = .5
min_pnr = 10
ssub_B=2
ring_size_factor = 1.4
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

print(min_corr)
print(min_pnr)
cnm=cnmf.CNMF(n_processes=1,dview=dview,Ain=Ain,params=opts)
cnm.fit(images)

min_SNR = 3            # adaptive way to set threshold on the transient size
r_values_min = 0.85    # threshold on space consistency (if you lower more components
#                        will be accepted, potentially with worst quality)
cnm.params.set('quality', {'min_SNR': min_SNR,
                           'rval_thr': r_values_min,
                           'use_cnn': False})
cnm.estimates.evaluate_components(images, cnm.params, dview=None)

print(' ***** ')
print('Number of total components: ', len(cnm.estimates.C))
print('Number of accepted components: ', len(cnm.estimates.idx_components))

# cm.stop_server(dview=dview)
cellTraces = cnm.estimates.C
valid_idx = cnm.estimates.idx_components
# import csv
#
# traces_sz = cellTraces.shape
# with open("cellTraces.txt", 'w', newline="\n") as f:
#     f_writer = csv.writer(f, delimiter='\t')
#     for i in range(traces_sz[0]):
#         f_writer.writerow(cellTraces[i, :])
# with open("validIdx.txt", 'w', newline="") as f:
#     f_writer = csv.writer(f, delimiter='\t')
#     f_writer.writerow(valid_idx)
cnm.estimates.view_components(img=cn_filter,idx=cnm.estimates.idx_components)
import pickle
with open('test.pickle', 'wb') as f:
    pickle.dump(cn_filter, f)
savedMovie = cnm.estimates.play_movie(images,frame_range = slice(0,1000), q_max=99.5,magnification=2,include_bck=True,gain_res=10,
                                      bpx=0,save_movie=True,thr=0.5,movie_name='/home/watson/Documents/results_movie.avi')
cnm.save('cnmf_save.hdf5')
cm.stop_server(dview=dview)

print('Done')
