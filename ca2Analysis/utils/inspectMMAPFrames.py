import caiman as cm
import numpy as np
import pickle
fname = '/home/watson/Documents/caiman_fromCluster/memmap__d1_524_d2_530_d3_1_order_C_frames_1000.mmap'
# fname = '/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/4914202252023/05212023/10_03_47/My_V4_Miniscope/memmap_d1_600_d2_600_d3_1_order_C_frames_1000.mmap'
Yr, dims, T = cm.load_memmap(fname)
images = Yr.T.reshape((T,) + dims, order='F')

image = images[1,:,:]
q_max=99.75
q_min=1
maxmov = np.nanpercentile(image[::max(1, len(image) // 100)], q_max)
minmov = np.nanpercentile(image[::max(1, len(image) // 100)], q_min)
data = 255 * (image - minmov) / (maxmov - minmov)
np.clip(data, 0, 255, data)
data = data.astype(np.uint8)
from PIL import Image as IM
data_tif = IM.fromarray(data)
data_tif.save('test.tif')
