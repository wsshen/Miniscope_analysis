import pickle
from caiman.base.rois import register_multisession
from caiman.utils import visualization
from caiman.utils.utils import download_demo
from matplotlib import pyplot as plt
import numpy as np
from scipy.sparse import csc_array
import os
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from scipy.signal import welch
from caiman.utils.visualization import get_contours
import time
from caiman.source_extraction.cnmf.deconvolution import GetSn
from caiman.source_extraction.cnmf.utilities import detrend_df_f
import csv

directory = '/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/5133207132023/reward_seeking/days_with_miniscope_recording/day24/10_49_21/'
data_file = 'output_rescaled_cnmf_save.hdf5'
temp_file = 'output_rescaled.pickle'
tempFile = directory + os.sep + temp_file
dataFile = directory + os.sep + data_file
with open(tempFile, 'rb') as f:
    template = pickle.load(f)
data = load_CNMF(dataFile)

spatial = data.estimates.A
spatial_np = spatial.toarray()
idx = data.estimates.idx_components
spatial_csc = csc_array(spatial_np[:,idx],dtype=np.float64)


def zscore_trace(denoised_trace,
                 raw_trace,
                 offset_method='floor',
                 sn_method='logmexp',
                 range_ff=[0.25, 0.5]):
    """
    "Z-score" calcium traces based on a calculation of noise from
    power spectral density high frequency.

    Inputs:
        denoised_trace: from estimates.C
        raw_trace: from estimates.C + estimates.YrA
        offset_method: offset to use when shifting the trace (default: 'floor')
            'floor': minimum of the trace so new min will be zero
            'mean': mean of the raw/denoised trace as the zeroing point
            'median': median of raw/denoised trace
            'none': no offset just normalize
        sn_method: how are psd central values caluclated
            mean
            median' or 'logmexp')
        range_ff: 2-elt array-like range of frequencies (input for GetSn) (default [0.25, 0.5])

    Returns
        z_denoised: same shape as denoised_trace
        z_raw: same shape as raw trace
        trace_noise: noise level from z_raw

    Adapted from code by Zach Barry.
    """
    noise = GetSn(raw_trace, range_ff=range_ff, method=sn_method)  # import this from caiman

    if offset_method == 'floor':
        raw_offset = np.min(raw_trace)
        denoised_offset = np.min(denoised_trace)
    elif offset_method == 'mean':
        raw_offset = np.mean(raw_trace)
        denoised_offset = np.mean(denoised_trace)
    elif offset_method == 'median':
        raw_offset = np.median(raw_trace)
        denoised_offset = np.median(denoised_trace)
    elif offset_method == 'none':
        raw_offset = 0
        denoised_offset = 0
    else:
        raise ValueError("offset_method should be floor, mean, median, or none.")
    # print(noise)
    z_raw = (raw_trace - raw_offset) / noise
    z_denoised = (denoised_trace - denoised_offset) / noise

    return z_denoised, z_raw, noise


def zscore_traces(cnm_c,
                  raw_traces,
                  offset_method='floor',
                  sn_method='logmexp',
                  range_ff=[0.25, 0.5]):
    """
    apply zscore_trace to all traces in estimates

    inputs:
        cnm_c: C array of denoised traces from cnm.estimates
        cnm_yra: YrA array of residuals from cnm.estimate
        offset_method: floor/mean/median (see zscore_trace)
        sn_method: mean/median/logmexp (see zscore_trace)
        range_ff: frequency range for GetSn

    outputs:
        denoised_z_traces
        raw_z_traces
        noise_all
    """
    # raw_traces = cnm_c + cnm_yra  # raw_trace[i] = c[i] + yra[i]
    raw_z_traces = []
    denoised_z_traces = []
    noise_all = []
    for ind, raw_trace in enumerate(raw_traces):
        denoised_trace = cnm_c[ind, :]
        z_denoised, z_raw, noise = zscore_trace(denoised_trace,
                                                raw_trace,
                                                offset_method=offset_method,
                                                sn_method=sn_method,
                                                range_ff=range_ff)

        denoised_z_traces.append(z_denoised)
        raw_z_traces.append(z_raw)
        noise_all.append(noise)

    denoised_z_traces = np.array(denoised_z_traces)
    raw_z_traces = np.array(raw_z_traces)
    noise_all = np.array(noise_all)

    return denoised_z_traces, raw_z_traces, noise_all

A = data.estimates.A
b = data.estimates.b
C = data.estimates.C
f = data.estimates.f
YrA = data.estimates.YrA
raw_traces = C + YrA

C_detrended = detrend_df_f(A, b, C, f, YrA=None, frames_window=500, flag_auto=False, use_fast=False,detrend_only=True,quantileMin=10)

denoised_z_C_detrended, raw_z_C_detrended, noise_all_C_detrended = zscore_traces(C_detrended,
                                                           C_detrended + YrA,
                                                           offset_method='floor')

traces_sz = denoised_z_C_detrended.shape
with open(directory + os.sep + "cellTraces_norm.txt", 'w', newline="\n") as f:
    f_writer = csv.writer(f, delimiter='\t')
    for i in range(traces_sz[0]):
        f_writer.writerow(denoised_z_C_detrended[i, :])

