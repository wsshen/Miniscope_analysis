import caiman as cm
import numpy as np
import pickle
fname = '/home/watson/Documents/caiman_fromCluster/memmap__d1_524_d2_530_d3_1_order_C_frames_10000.mmap'
# fname = '/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/4914202252023/05212023/10_03_47/My_V4_Miniscope/memmap_d1_600_d2_600_d3_1_order_C_frames_1000.mmap'
Yr, dims, T = cm.load_memmap(fname)
images = Yr.T.reshape((T,) + dims, order='F')
gSig=[8,8]
cn_filter, pnr = cm.summary_images.correlation_pnr(images[::5],gSig=gSig[0],swap_dim=False)
with open('pnr.pickle','wb') as f:
    pickle.dump([cn_filter, pnr],f)
print('done')

