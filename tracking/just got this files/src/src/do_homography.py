import pandas as pd
import numpy as np
import cv2
import os
import scipy.io
import csv
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Circle

curr_folder = os.path.dirname(os.path.abspath(__file__))


# List of points
image_raw_pts_collected = []
image_pts_collected = []
world_pts_collected = []
im_raw_ptnum = 0
im_ptnum = 0
wd_ptnum = 0
image_raw = []
image = []
world = []

def im_raw_clicks(event, x, y, flags, param):
    global image_raw_pts_collected, image_raw, im_raw_ptnum

    if event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        image_raw_pts_collected.append([x, y])
        im_raw_ptnum+=1
        cv2.circle(image_raw,(x,y), 20, (255,255,255), 5)
        cv2.putText(image_raw, str(im_raw_ptnum), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 5, cv2.LINE_AA) 
        cv2.imshow("image_raw", image_raw)

def im_clicks(event, x, y, flags, param):
    global image_pts_collected, image, im_ptnum

    if event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        image_pts_collected.append([x, y])
        im_ptnum+=1
        cv2.circle(image,(x,y), 5, (255,255,255), 3)
        cv2.putText(image, str(im_ptnum), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 5, cv2.LINE_AA) 
        cv2.imshow("image", image)

def wd_clicks(event, x, y, flags, param):
    global world_pts_collected, world, wd_ptnum

    if event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        world_pts_collected.append([x, y])
        wd_ptnum+=1
        cv2.circle(world,(x,y), 5, (255,255,255), 3)
        cv2.putText(world, str(wd_ptnum), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 5, cv2.LINE_AA) 
        cv2.imshow("world", world)

def main():
    global image, image_raw, world, image_raw_pts_collected, image_pts_collected, world_pts_collected
    # Read camera matrices from file
    intrinsics = np.load('intrinsics.npy')
    #intrinsics = intrinsics.transpose()
    distortion = np.load('distortion.npy')

    # Ask user to open video file
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    folder_name = file_path.split('/')[-2]
    file_name = file_path.split('/')[-2] + '_' + file_path.split('/')[-1].split('.')[0]
    image = cv2.imread(file_path)
    
    # Ask user to open map file
    root = tk.Tk()
    root.withdraw()
    map_file_path = filedialog.askopenfilename()
    map_folder_name = map_file_path.split('/')[-2]
    map_file_name = map_file_path.split('/')[-2] + '_' + map_file_path.split('/')[-1].split('.')[0]
    map_image = cv2.imread(map_file_path)
    map_copy = map_image.copy()

    # # Open the video file in opencv
    # cap = cv2.VideoCapture(file_path)#videoFileNamePath+videoFileName)
    # frameNo = 0
    
    # # # Show the 10th frame
    # while(cap.isOpened()):
    #     ret, frame = cap.read()
    #     if not ret:
    #         print('no frames')
    #         break
    #     frameNo+=1
    #     if frameNo < 20:
    #         continue
    #     elif frameNo == 20:
    #         break

    
    # h,  w = frame.shape[:2]
    # # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsics,distortion,(w,h),1,(w,h))
    # # # undistort
    # # undistorted_frame = cv2.undistort(frame, intrinsics, distortion, None, newcameramtx)
    
    # # FISHEYE
    # DIM=(1920, 1440)
    # dim1 = frame.shape[:2][::-1]
    # scaled_K = intrinsics * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    # scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, distortion, dim1, np.eye(3), balance=1)
    # map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, distortion, np.eye(3), new_K, dim1, cv2.CV_16SC2)
    # undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


    # image_raw = frame
    # image = undistorted_frame.copy()
    world = map_image.copy()

    wd_h, wd_w = map_image.shape[:2]

    cv2.namedWindow("image",cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("image", (int(w/3), int(h/3)))
    cv2.setMouseCallback("image", im_clicks)
    cv2.imshow("image", image)

    # cv2.namedWindow("image_raw",cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("image_raw", (int(w/3), int(h/3)))
    # cv2.setMouseCallback("image_raw", im_raw_clicks)
    # cv2.imshow("image_raw", image_raw)

    cv2.namedWindow("world",cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("world", (int(wd_w/3), int(wd_h/3)))
    cv2.setMouseCallback("world", wd_clicks)
    cv2.imshow("world", world)

    cv2.waitKey(0)

    print(image_raw_pts_collected)
    print(world_pts_collected)
    print(image_pts_collected)

    # Find the homography between the points
    cv2hom = cv2.findHomography(np.array(image_pts_collected), np.array(world_pts_collected))
    np.save('homography.npy',cv2hom)

    #2D to 3d projection
    wd_projection_ocv = []
    for pt in image_pts_collected:
        pt.append(1)
        wd_proj_3d = np.matmul(cv2hom[0], pt)
        wd_proj_3d_norm = wd_proj_3d / wd_proj_3d[2]
        wd_projection_ocv.append(wd_proj_3d_norm)    


    for wd_pt, wd_pt_ocv in zip(world_pts_collected, wd_projection_ocv):
        cv2.circle(map_image, (wd_pt[0],wd_pt[1]), 3, (0,255,0), 3)
        cv2.circle(map_image, (int(wd_pt_ocv[0]),int(wd_pt_ocv[1])), 5, (255,0,0), 5)

    warped_image = cv2.warpPerspective(image.copy(), cv2hom[0], (map_copy.shape[1],map_copy.shape[0]))#,flags=cv2.WARP_INVERSE_MAP)
    alpha = 0.7
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(warped_image, alpha, map_copy, beta, 0.0)
    cv2.imshow('warped',dst)
    cv2.imshow("map", map_image)
    cv2.imwrite('warped_'+str(len(world_pts_collected))+'pts.png', dst)
    cv2.imwrite('map_'+str(len(world_pts_collected))+'pts.png', map_image)
    cv2.imwrite('drawnimage.png',image)
    cv2.imwrite('drawnworld.png',world)
    np.save('homography_matrix_'+str(len(world_pts_collected))+'pts.npy', cv2hom)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()


