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

# fnames = ['/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/4914202252023/05192023/10_48_49/My_V4_Miniscope/90.avi']
# # fnames = [download_demo(fnames[0])]
# print("entering dview")
# if 'dview' in locals():
#     cm.stop_server(dview=None)
#     print("exiting dview")
# c, dview, n_processes = cm.cluster.setup_cluster(
#     backend='local', n_processes=None, single_thread=False)
# print(n_processes)
# #dataset dependent parameters
# frate = 30
# decay_time = 0.8
#
# # motion correction parameters
# motion_correct = True
# pw_rigid = True
# gSig_filt = (7,7)
# max_shifts = (20,20)
# strides = (64,64)
# overlaps = (32,32)
# max_deviation_rigid = 20
# border_nan = 'copy'
#
# mc_dict = {
#     'fnames': fnames,
#     'fr': frate,
#     'decay_time': decay_time,
#     'pw_rigid': pw_rigid,
#     'gSig_filt':gSig_filt,
#     'max_shifts':max_shifts,
#     'strides': strides,
#     'overlaps': overlaps,
#     'max_deviation_rigid': max_deviation_rigid,
#     'border_nan': border_nan
# }
#
# opts=params.CNMFParams(params_dict=mc_dict)
# if motion_correct:
#     mc = MotionCorrect(fnames,dview=dview,**opts.get_group('motion'))
#     mc.motion_correct(save_movie=True)
#     fname_mc = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig
#     if pw_rigid:
#         bored_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
#                                      np.max(np.abs(mc.y_shifts_els)))).astype(int)
#     else:
#         bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(int)
#         print('PW_rigid is false')
#        # plt.subplot(1,2,1);
#        # plt.imshow(mc.total_template_rig)
#        # plt.subplot(1,2,2);
#        # plt.plot(mc.shifts_rig)
#        # plt.legend(['x shifts','y shifts'])
#        # plt.xlabel('frames')
#        # plt.ylabel('pixels')
#     bord_px=0 if border_nan == 'copy' else bord_px
#     fname_new = cm.save_memmap(fname_mc,base_name='memmap_',order='C',border_to_0=bord_px)
# else:
#     fname_new = cm.save_memmap(fnames,base_name='memmap_',order='C',border_to_0=0,dview=None)

bord_px =0
fname_new = '/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/4914002252023/06082023/14_40_57/My_V4_Miniscope/memmap_d1_600_d2_600_d3_1_order_C_frames_1000.mmap'
Yr,dims,T=cm.load_memmap(fname_new)
images = Yr.T.reshape((T,)+dims,order='F')
# parameters for source extraction and deconvolution
# m = cm.base.movies.movie(images,start_time=0,fr=30)
R = cm.load(fname_new)
mR = R.mean(0)
# m.save('test.avi')
Ain = cm.base.rois.extract_binary_masks_from_structural_channel(mR, gSig=7, expand_method='dilation')[0]
gSig = (5, 5)
cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=gSig[0], swap_dim=False) # change swap dim if output looks weird, it is a problem with tiffile
# inspect the summary images and set the parameters
inspect_correlation_pnr(cn_filter, pnr)

# cm.stop_server(dview=dview)
print('Done')
input()