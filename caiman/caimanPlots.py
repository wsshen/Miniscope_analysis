from caiman.source_extraction.cnmf import load_CNMF
import os
dir = '/home/watson/Documents/caiman_fromCluster/49142/06072023/13_04_23'
hdf5Path = 'cnmf_save_40_79.hdf5'
picklePath = 'output_40_79.pickle'
aaa = load_CNMF(dir+os.sep+hdf5Path,n_processes=1,dview=None)

import caiman as cm
# bord_px =0
# fname_new = '/home/watson/Documents/caiman_fromCluster/49142/06072023/13_04_23/memmap_d1_600_d2_600_d3_1_order_C_frames_40000.mmap'
# Yr,dims,T=cm.load_memmap(fname_new)
# images = Yr.T.reshape((T,)+dims,order='F')
# from caiman.source_extraction import cnmf
# min_SNR = 5            # adaptive way to set threshold on the transient size
# r_values_min = 0.85
# cnm=cnmf.CNMF(n_processes=1,dview=None,Ain=None,params=None)
# cnm.params.set('quality', {'min_SNR': min_SNR,
#                            'rval_thr': r_values_min,
#                            'use_cnn': False})
#
# aaa.estimates.evaluate_components(images, cnm.params, dview=None)
import pickle
with open(dir+os.sep+picklePath,'rb') as f:
    template01 = pickle.load(f)
aaa.estimates.view_components(img=template01[0],idx=None)
input()