import cv2
from PIL import Image as IM
import glob, os
import numpy as np
import math
import caiman as cm
from caiman.mmapping import prepare_shape
# dir = '/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/tail_striatum/2023_03_14/11_43_53/My_V4_Miniscope/'
# dir = "/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/5133207132023/reward_seeking/days_with_miniscope_recording/day1_poke_lick/session1/14_29_23/My_V4_miniscope"
# # os.chdir(dir)
# path = dir+os.sep+'1.avi'
# newFrames = []
# for aviFile in glob.glob("*.avi"):
# v = cv2.VideoCapture(path)
drawing = False
ix,iy = -1,-1
params = [-1,-1,-1,-1]
circle=[]
# mouse callback function
def calc_radius(x1,y1,x2,y2):
    delta_x = abs(x1-x2)
    delta_y = abs(y1-y2)
    return int(delta_x/2)
def calc_center(x1,y1,x2,y2):
    return (int((x1+x2)/2),int((y1+y2)/2))
def draw_circle(event, x, y, flags, param):
    """ Draw rectangle on mouse click and drag """
    global ix,iy,drawing,params
    # if the left mouse button was clicked, record the starting and set the drawing flag to True
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    # mouse is being moved, draw rectangle
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            params = [ix, iy, x, y]
    # if the left mouse button was released, set the drawing flag to False
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        params = [ix,iy,x,y]
        cv2.circle(first_image, calc_center(ix, iy,x,y), calc_radius(ix, iy,x,y), (255, 0, 0), -1)

# numFrames = v.get(cv2.CAP_PROP_FRAME_COUNT)
# for i in range(1,int(numFrames),1):
# hasFrame, frame = v.read()
# if hasFrame:
#     print(type(frame))
#     gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     newFrames.append(IM.fromarray(gray_frame))
#     ker = cv2.getGaussianKernel(31, 10)
#     ker2D = ker.dot(ker.T)
#     nz = np.nonzero(ker2D >= ker2D[:, 0].max())
#     zz = np.nonzero(ker2D < ker2D[:, 0].max())
#     ker2D[nz] -= ker2D[nz].mean()
#     ker2D[zz] = 0
#     filtered_image = cv2.filter2D(np.array(gray_frame, dtype=np.float32),
#                             -1, ker2D, borderType=cv2.BORDER_REFLECT)
#     ret, thresh2 = cv2.threshold(filtered_image, 0.01, 255, cv2.THRESH_BINARY)
#
#     enhanced_image = cv2.convertScaleAbs(filtered_image,alpha=1000)
#     # cv2.startWindowThread()
#     cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
#     cv2.setMouseCallback('frame', draw_circle)
#
#     cv2.imshow('frame',enhanced_image)
#
#     while True:
#         cv2.imshow('frame', enhanced_image)
#         k = cv2.waitKey(1)
#         if k == ord("c"):  # c to terminate a program
#
#             break
#     print(params)
#     blank_img = np.zeros(np.shape(enhanced_image))
#     circle = cv2.circle(blank_img, calc_center(params[0], params[1], params[2], params[3]), calc_radius(params[0], params[1], params[2], params[3]), (255, 0, 0), -1)
#     circle=circle.astype(np.uint8)
#     cv2.namedWindow('test',cv2.WINDOW_NORMAL)
#     cv2.imshow('test',circle)
#     cv2.waitKey(0)
# v.release()
# newFrames[0].save("test.tif",save_all=True,append_images=newFrames[1:])
# mask_save = IM.fromarray(circle)
# mask_save.save('mask.tif')

# circle = np.array(IM.open('mask.tif'))
fname_new='/home/watson/Documents/caiman_fromCluster/memmap_d1_524_d2_530_d3_1_order_C_frames_52000.mmap'
Yr, dims, T = cm.load_memmap(fname_new)
images = Yr.T.reshape((T,) + dims, order='F')
a,b,c=np.shape(images)
# m=cm.base.movies.movie(images[0:100,:,:],start_time=0,fr=30)
# m.save('test.avi')
newFrames = []
a=10000
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('frame', draw_circle)
first_image = np.ascontiguousarray(np.array(images[1,:,:]))

cv2.imshow('frame',first_image)

while True:
    cv2.imshow('frame', first_image)
    k = cv2.waitKey(1)
    if k == ord("c"):  # c to terminate a program
        break

print(params)
blank_img = np.zeros(np.shape(first_image))
circle = cv2.circle(blank_img, calc_center(params[0], params[1], params[2], params[3]), calc_radius(params[0], params[1], params[2], params[3]), (255, 0, 0), -1)
circle=circle.astype(np.uint8)
mask_save = IM.fromarray(circle)
mask_save.save('mask.tif')
for i in range(a):
    image = np.array(images[i,:,:])

    masked = cv2.bitwise_and(image, image, mask=circle)
    masked+=np.random.uniform(low=0.0,high=0.0001,size=np.shape(image))
    newFrames.append(masked)

newFrames = np.transpose(newFrames, list(range(1, len(dims) + 1)) + [0])
newFrames = np.reshape(newFrames, (np.prod(dims), a), order='F')
newFrames = np.ascontiguousarray(newFrames, dtype=np.float32) +  np.float32(0)
    # cv2.namedWindow('masked',cv2.WINDOW_NORMAL)
    # cv2.imshow('masked',masked)
    # cv2.waitKey(0)

new_fname = '/home/watson/Documents/caiman_fromCluster/memmap__d1_524_d2_530_d3_1_order_C_frames_10000.mmap'
# big_mov = np.memmap(new_fname,
#                     mode='w+',
#                     dtype=np.float32,
#                     shape=prepare_shape((np.prod(dims), T)),
#                     order='C')
newFrames.tofile(new_fname)
# big_mov[:] = newFrames[:]
# del big_mov
print('done')