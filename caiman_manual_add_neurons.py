import pickle
from caiman.base.rois import register_multisession
from caiman.utils import visualization
from caiman.utils.utils import download_demo
from matplotlib import pyplot as plt
import numpy as np
from scipy.sparse import csc_array

dir = '/home/watson/Documents/caiman_fromCluster/49142/06072023/13_04_23/'
data01 = 'cnmf_save_manual.hdf5'
data02 = 'cnmf_save_40_79.hdf5'

temp01 = 'output_0_39.pickle'
temp02 = 'output_40_79.pickle'

import os
tempFile01 = dir + os.sep + temp01
tempFile02 = dir + os.sep + temp02

dataFile01 = dir + os.sep + data01
dataFile02 = dir + os.sep + data02

with open(tempFile01, 'rb') as f:
    template01 = pickle.load(f)
with open(tempFile02, 'rb') as f:
    template02 = pickle.load(f)

from caiman.source_extraction.cnmf import load_CNMF

data1 = load_CNMF(dataFile01)
data2 = load_CNMF(dataFile02)

