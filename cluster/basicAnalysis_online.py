try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import logging
import numpy as np

logging.basicConfig(format=
                          "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    # filename="/tmp/caiman.log",
                    level=logging.INFO)

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.utils import download_demo
import matplotlib.pyplot as plt

fnames = ['/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/4914202252023/05192023/10_48_49/My_V4_Miniscope/0.avi']

print(fnames)

fr = 30                                                             # frame rate (Hz)
decay_time = 0.8                                                    # approximate length of transient event in seconds
gSig = (3,3)
gSiz = (13,13)# expected half size of neurons
p = 1                                                               # order of AR indicator dynamics
min_SNR = 10                                                         # minimum SNR for accepting new components
rval_thr = 0.90                                                     # correlation threshold for new component inclusion
ds_factor = 1                                                       # spatial downsampling factor (increases speed but may lose some fine structure)
gnb = 2                                                             # number of background components
gSig = tuple(np.ceil(np.array(gSig)/ds_factor).astype('int'))       # recompute gSig if downsampling is involved
mot_corr = True                                                     # flag for online motion correction
pw_rigid = False                                                   # flag for pw-rigid motion correction (slower but potentially more accurate)
max_shifts_online = np.ceil(10./ds_factor).astype('int')            # maximum allowed shift during motion correction
sniper_mode = True                                                  # flag using a CNN to detect new neurons (o/w space correlation is used)
init_batch = 100                                                    # number of frames for initialization (presumably from the first file)
expected_comps = 50                                                # maximum number of expected components used for memory pre-allocation (exaggerate here)
dist_shape_update = True                                            # flag for updating shapes in a distributed way
min_num_trial = 10                                                  # number of candidate components per frame
K = 5                                                               # initial number of components
epochs = 1                                                          # number of passes over the data
show_movie = False                                                 # show the movie with the results as the data gets processed
rf = 40
stride = 20
params_dict = {'fnames': fnames,
               'fr': fr,
               'decay_time': decay_time,
               'gSig': gSig,
               'p': p,
               'min_SNR': min_SNR,
               'rval_thr': rval_thr,
               'ds_factor': ds_factor,
               'nb': gnb,
               'motion_correct': mot_corr,
               'init_batch': init_batch,
               'init_method': 'bare',
               'normalize': True,
               'expected_comps': expected_comps,
               'sniper_mode': sniper_mode,
               'dist_shape_update' : dist_shape_update,
               'min_num_trial': min_num_trial,
               'K': K,
               'epochs': epochs,
               'max_shifts_online': max_shifts_online,
               'pw_rigid': pw_rigid,
               'stride' : stride,
               'rf' : rf,
               'show_movie': show_movie}

if 'dview' in locals():
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

opts = cnmf.params.CNMFParams(params_dict=params_dict)

cnm = cnmf.online_cnmf.OnACID(params=opts,dview=dview)
cnm.fit_online()
cnm.save('/home/watson/Documents/cnmf_save.hdf5')
images = cm.load(fnames[0],subindices=slice(0,1000))
cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=gSig[0], swap_dim=False) # change swap dim if output looks weird, it is a problem with tiffile
import pickle
with open('test.pickle', 'wb') as f:
    pickle.dump(cn_filter, f)

savedMovie = cnm.estimates.play_movie(images,frame_range = slice(0,1000), q_max=99.5,magnification=2,include_bck=False,gain_res=10,
                                      bpx=0,save_movie=True,thr=0.5,movie_name='/home/watson/Documents/results_movie.avi')
cm.stop_server(dview=dview)
print('done')