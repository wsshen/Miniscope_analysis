import cv2
from PIL import Image as IM
import glob, os
import numpy as np
# dir = '/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/tail_striatum/2023_03_14/11_43_53/My_V4_Miniscope/'
dir = "/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/5133207132023/reward_seeking/days_with_miniscope_recording/day1_poke_lick/session1/14_29_23/My_V4_miniscope"
# os.chdir(dir)
path = dir+os.sep+'1.avi'
newFrames = []
# for aviFile in glob.glob("*.avi"):
v = cv2.VideoCapture(path)
numFrames = v.get(cv2.CAP_PROP_FRAME_COUNT)
for i in range(1,int(numFrames),1):
    hasFrame, frame = v.read()
    if hasFrame:
        print(type(frame))
        gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        newFrames.append(IM.fromarray(gray_frame))
        ker = cv2.getGaussianKernel(31, 10)
        ker2D = ker.dot(ker.T)
        nz = np.nonzero(ker2D >= ker2D[:, 0].max())
        zz = np.nonzero(ker2D < ker2D[:, 0].max())
        ker2D[nz] -= ker2D[nz].mean()
        ker2D[zz] = 0
        aaa = cv2.filter2D(np.array(gray_frame, dtype=np.float32),
                                -1, ker2D, borderType=cv2.BORDER_REFLECT)
        cv2.startWindowThread()
        cv2.namedWindow('frame')
        cv2.imshow('frame',aaa)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
v.release()
# newFrames[0].save("test.tif",save_all=True,append_images=newFrames[1:])

