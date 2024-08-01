import cv2
from PIL import Image as IM
import glob, os
import numpy as np
import math

dir = "/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/5133207132023/reward_seeking/days_with_miniscope_recording/day1_poke_lick/session1/14_29_23/My_V4_miniscope"
videoName = 'output_0_51_rescaled'
suffix = '.avi'

pathName = dir+os.sep+videoName+suffix

newFrames = []
# for aviFile in glob.glob("*.avi"):
v = cv2.VideoCapture(pathName)
drawing = False
ix,iy = -1,-1
params = [-1,-1,-1,-1]
circle=[]
# mouse callback function
def draw_rectangle(event, x, y, flags, param):
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
        cv2.rectangle(enhanced_image, (ix, iy), (x, y), (255, 0, 0), -1)
        params = [ix,iy,x,y]

numFrames = v.get(cv2.CAP_PROP_FRAME_COUNT)
# for i in range(1,int(numFrames),1):
hasFrame, frame = v.read()
if hasFrame:
    print(type(frame))
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    enhanced_image = cv2.convertScaleAbs(gray_frame,alpha=1000)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('frame', draw_rectangle)

    cv2.imshow('frame',enhanced_image)

    while True:
        cv2.imshow('frame', enhanced_image)
        k = cv2.waitKey(1)
        if k == ord("c"):  # c to terminate a program

            break
    print(params)
fourcc = cv2.VideoWriter_fourcc(*'FFV1')
out = cv2.VideoWriter(dir+os.sep+videoName+'_cropped'+suffix,fourcc,30,(int(params[2]-params[0]),int(params[3]-params[1])),isColor=False)
# out = cv2.VideoWriter('cropped_output.avi',fourcc,30,(600,600),isColor=False)

while hasFrame:
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cropped_frame = gray_frame[params[1]:params[3],params[0]:params[2]]
    out.write(cropped_frame)
    hasFrame,frame=v.read()

out.release()
v.release()
cv2.destroyAllWindows()