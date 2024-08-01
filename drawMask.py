import cv2
from PIL import Image as IM
import glob, os
import numpy as np
import math
import caiman as cm
from caiman.mmapping import prepare_shape
import pickle
import copy
drawing = False
ix,iy = -1,-1
params = [-1,-1,-1,-1]
circle=[]
circle_copy = np.array([])
# mouse callback function
def calc_radius(x1,y1,x2,y2):
    delta_x = abs(x1-x2)
    delta_y = abs(y1-y2)
    return int(delta_x/2)
def calc_center(x1,y1,x2,y2):
    return (int((x1+x2)/2),int((y1+y2)/2))
def draw_circle(event, x, y, flags, param):
    """ Draw rectangle on mouse click and drag """
    global ix,iy,drawing,params,circle_copy,first_image,first_image_copy
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
        if circle_copy.size !=0:
            first_image = cv2.addWeighted(first_image_copy, 1.0, circle_copy, 0.0, 0)
            cv2.imshow('frame', first_image)

        circle_copy = cv2.circle(first_image, calc_center(ix, iy,x,y), calc_radius(ix, iy,x,y), (0, 255, 255), 1)
        print(np.shape(circle_copy))

# fname_new='/home/watson/Documents/caiman_fromCluster/memmap_d1_300_d2_300_d3_1_order_C_frames_52000.mmap'
filePath = '/home/watson/Documents/caiman_fromCluster/51333/day25'

with open(filePath+os.sep+'output_rescaled.pickle','rb') as f:
    pnr=pickle.load(f)

newFrames = []
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('frame', draw_circle)
first_image = pnr[0]
first_image_copy = copy.copy(first_image)
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
mask_save.save(filePath+os.sep+ 'mask.tif')


print('done')