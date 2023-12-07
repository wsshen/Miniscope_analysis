import caiman as cm

from caiman.utils.visualization import inspect_correlation_pnr,nb_inspect_correlation_pnr

fname_new = '/home/watson/Documents/memmap_d1_600_d2_600_d3_1_order_C_frames_14890.mmap'
Yr,dims,T=cm.load_memmap(fname_new)
images = Yr.T.reshape((T,)+dims,order='F')

# m = cm.base.movies.movie(images,start_time=0,fr=30)
# m.save('test.avi')
gSig = (5, 5)
cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=gSig[0], swap_dim=False) # change swap dim if output looks weird, it is a problem with tiffile
# inspect the summary images and set the parameters
inspect_correlation_pnr(cn_filter, pnr)

print('Done')
input()