from caiman.source_extraction.cnmf.initialization import downscale
import pickle
from caiman.base.rois import register_multisession
from caiman.utils import visualization
from caiman.utils.utils import download_demo
from matplotlib import pyplot as plt
import numpy as np
from scipy.sparse import csc_array
import caiman as cm
import os
dir = '/om2/user/shenwang/miniscope/49142/06072023/13_04_23/'
dataFileName = 'cnmf_save_manual.hdf5'
dataPath = dir + os.sep + dataFileName
from caiman.source_extraction.cnmf import load_CNMF

data = load_CNMF(dataPath)
# def findEstimatesB(savedEst,imgs):
#     AC = savedEst.A.dot(savedEst.C)
#     if savedEst.W is not None:
#         ssub_B = int(round(np.sqrt(np.prod(dims) / savedEst.W.shape[0])))
#         B = imgs.reshape((-1, np.prod(dims)), order='F').T - AC
#         if ssub_B == 1:
#             B = savedEst.b0[:, None] + savedEst.W.dot(B - savedEst.b0[:, None])
#         else:
#             WB = savedEst.W.dot(downscale(B.reshape(dims + (B.shape[-1],), order='F'),
#                           (ssub_B, ssub_B, 1)).reshape((-1, B.shape[-1]), order='F'))
#             Wb0 = savedEst.W.dot(downscale(savedEst.b0.reshape(dims, order='F'),
#                           (ssub_B, ssub_B)).reshape((-1, 1), order='F'))
#             B = savedEst.b0.flatten('F')[:, None] + (np.repeat(np.repeat((WB - Wb0).reshape(((dims[0] - 1) // ssub_B + 1, (dims[1] - 1) // ssub_B + 1, -1), order='F'),
#                                  ssub_B, 0), ssub_B, 1)[:dims[0], :dims[1]].reshape((-1, B.shape[-1]), order='F'))
#         B = B.reshape(dims + (-1,), order='F').transpose([2, 0, 1])
#     elif savedEst.b is not None and savedEst.f is not None:
#         B = savedEst.b.dot(savedEst.f)
#         if 'matrix' in str(type(B)):
#             B = B.toarray()
#         B = B.reshape(dims + (-1,), order='F').transpose([2, 0, 1])
#     else:
#         B = np.zeros_like(Y_rec)
def HALS4activity(Yr, A, b, iters=100):
    if b is not None:
        Ab =  np.c_[A, b]
    else:
        Ab = A
    U = Ab.T.dot(Yr)
    V = Ab.T.dot(Ab) + np.finfo(Ab.dtype).eps
    Cf = U/V.diagonal()[:,None]
    for _ in range(iters):
        for m in range(len(U)):  # neurons and background
            Cf[m] = np.clip(Cf[m] + (U[m] - V[m].dot(Cf)) / V[m, m], 0, np.inf)
    return Cf
fname_new =  '/om2/user/shenwang/miniscope/49142/06072023/13_04_23/memmap_d1_600_d2_600_d3_1_order_C_frames_40000.mmap'
Yr,dims,T=cm.load_memmap(fname_new)
images = Yr.T.reshape((T,)+dims,order='F')

savedEst = data.estimates
A = data.estimates.A
A_np = A.toarray()
# B = findEstimatesB(savedEst,images)
b = data.estimates.b
Cf = HALS4activity(Yr,A_np,b,iters=100)
with (dir+os.sep+'cf.pickle','wb') as f:
    pickle.dump(Cf,f)
print('Done')