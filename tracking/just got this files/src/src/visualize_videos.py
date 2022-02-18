import pandas as pd
import numpy as np
import cv2
import scipy.io
import csv
import numpy as np

# Open the video file in opencv
cap = cv2.VideoCapture("D:/projects/trans-plan/data/2020/May/351cm_height/videos/GXAB0755.MP4")#videoFileNamePath+videoFileName)
data = pd.read_csv("D:/projects/trans-plan/data/2020/May/351cm_height/detections/faster_rcnn_inception_v2_coco_2018_01_28GXAB0755.MP4_detection_test.csv",header=None,names=['framenum', 'v','x','y','w','h','confidence']) 

vid = cv2.VideoWriter('detections_resized.mp4',cv2.VideoWriter_fourcc(*"mp4v"), 60, (1920,1440))
index = 0
frame_num = 0
mynum = 0
#cv2.namedWindow('video',cv2.WINDOW_NORMAL)
#cv2.resizeWindow('video',1536, 864)
while(cap.isOpened()):
    print(frame_num)
    if frame_num > 450:
        break
    ret, frame = cap.read()
    if not ret:
        continue
    frame_num_detection = data['framenum'][index]
    #frame_num_detection = int(frame_num_str.split('.')[0])
    while frame_num == frame_num_detection:
        x = int(data['x'][index])
        y = int(data['y'][index])
        w = int(data['w'][index])
        h = int(data['h'][index])
        mid_pt = [x+w/2, y+h]
        index+=1
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 4)
        cv2.circle(frame, (int(mid_pt[0]),int(mid_pt[1])), 15, (0,0,255), -1)
        frame_num_detection = data['framenum'][index]
        #frame_num_detection = int(frame_num_str.split('.')[0])
    frame_num+=1
   
    res = cv2.resize(frame, (1920, 1440)) 
    vid.write(res)
    #cv2.imshow('video',frame)
    #cv2.waitKey(10)
    #cv2.imwrite('SSDdetections.png',frame)

vid.release()
    