#! /usr/bin python
import numpy as np
import caiman as cm
import os
import pickle
from PIL import Image as IM
import cv2
import glob
# dataset dependent parameters
def add_noise(args):
    dir_name = args.directory
    # suffix='.mmap'
    fname_new_list = glob.glob(dir_name + os.sep + 'memmap_d1*.mmap')
    if not fname_new_list:
        print("There is no mmap file in",dir_name)
        return
    fname_new = fname_new_list[0]    
    # file_name = 'memmap_d1_300_d2_300_d3_1_order_C_frames_51881'
    # fname_new = dir_name+os.sep+file_name+suffix

    fname_split = fname_new.split(".")
    file_name = fname_split[0]
    suffix = fname_split[1]
    Yr,dims,T=cm.load_memmap(fname_new)
    images = Yr.T.reshape((T,)+dims,order='F')

    circle = np.array(IM.open(dir_name+os.sep+'mask.tif'))
    plot_summary = True
    add_noise = True

    if add_noise:
        newFrames=[]
        for i in range(T):
            image = np.array(images[i,:,:])
            masked = cv2.bitwise_and(image,image,mask=circle)
            masked += np.random.uniform(low=0.0,high=1,size=np.shape(image))*0.0001
            newFrames.append(masked)

        newFrames = np.transpose(newFrames,list(range(1,len(dims)+1))+[0])
        newFrames = np.reshape(newFrames,(np.prod(dims),T),order='F')
        newFrames = np.ascontiguousarray(newFrames,dtype=np.float32) + np.float32(0)

        new_fname = file_name+'_'+ '.'+suffix
        newFrames.tofile(new_fname)
    gSig=[3,3]
    if plot_summary:
        if add_noise:
            newFrames = np.reshape(newFrames.T, (T,)+dims, order='F')
            cn_filter,pnr = cm.summary_images.correlation_pnr(newFrames[::1],gSig=gSig[0],swap_dim=False)
        else:
            cn_filter,pnr = cm.summary_images.correlation_pnr(images[::1],gSig=gSig[0],swap_dim=False)
    with open(dir_name+os.sep+'addedNoise.pickle','wb') as f:
        pickle.dump([cn_filter,pnr],f)

    print("Done!")

def main():
    import argparse
    parser = argparse.ArgumentParser() # fromfile_prefix_chars='@'

    parser.add_argument("--directory",type=str)

    args = parser.parse_known_args()[0]
    add_noise(args)

if __name__ == "__main__":
    main()
