import pandas as pd
import numpy as np
import cv2
import scipy.io
import csv
import numpy as np
import tkinter as tk
from tkinter import filedialog
#import matlab.engine
import drawhelper


draw_frame = []

def main():
    
    # # Read camera matrices from file
    # intrinsics = np.loadtxt('../extra_data/intrinsic_params_matlab.txt')
    # intrinsics = intrinsics.transpose()
    # distortion = np.loadtxt('../extra_data/distortion_coefficients_matlab.txt')

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

    #frame = cv2.imread(file_path)
    # Open the video file in opencv
    cap = cv2.VideoCapture(file_path)#videoFileNamePath+videoFileName)
    frameNo = 0
    
    # # Show the 10th frame
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            print('no frames')
            break
        frameNo+=1
        if frameNo < 10:
            continue
        elif frameNo == 10:
            break

    h,  w = frame.shape[:2]
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsics,distortion,(w,h),1,(w,h))
    # # undistort
    # undistorted_frame = cv2.undistort(frame, intrinsics, distortion, None, newcameramtx)
    # crop the image
    #x,y,w,h = roi
    #undistorted_frame = undistorted_frame[y:y+h, x:x+w]


    # FISHEYE
    DIM=(1920, 1440)
    dim1 = frame.shape[:2][::-1]
    scaled_K = intrinsics * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, distortion, dim1, np.eye(3), balance=1)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, distortion, np.eye(3), new_K, dim1, cv2.CV_16SC2)
    undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Ask and wait for the user to draw 4 lines
    image_points, draw_frame_im = drawhelper.getpoints(undistorted_frame, 20)

    # Ask user to open maps screenshot
    cv2.imshow('reference',draw_frame_im)
    maps_ss_path = filedialog.askopenfilename()
    #maps_ss_path  = '../data/' + folder_name + '/' + 'map.png'
    maps_ss = cv2.imread(maps_ss_path)
    drawhelper.reset()
    world_points, draw_frame_map = drawhelper.getpoints(maps_ss, 20)
    world_points_3d = []
    image_points_np = []
    for row, rowi in zip(world_points, image_points):
        row_list = list(row)
        row_list.append(0)
        world_points_3d.append(row_list)
        rowi_list = list(rowi)
        image_points_np.append(rowi_list)

    [retval, rvec, tvec] = cv2.solvePnP(np.array(world_points_3d, dtype='float32'), np.array(image_points_np, dtype='float32'), intrinsics, distortion)
    [rvec3, jacobian] = cv2.Rodrigues(rvec)

    np.savetxt('rvec3.csv', rvec3, delimiter=',')
    np.savetxt('tvec.csv', tvec, delimiter=',')
    #rvec3_file_read = np.genfromtxt('../data/' + folder_name + '/'+'rvec3.csv', delimiter=',')
    cam_location = np.matmul(np.linalg.inv(-rvec3),(tvec))
    cv2.circle(draw_frame_map, (cam_location[0],cam_location[1]), 15, (0, 0, 255), 3)
    cv2.putText(draw_frame_map,"Estimated camera position", (cam_location[0]+20,cam_location[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

    rot_trans_matrix = [rvec3[:,0], rvec3[:,1],np.transpose(tvec[:,0])]
    rot_trans_matrix = np.transpose(rot_trans_matrix)
    t = tvec[2]
    rot_trans_matrix = rot_trans_matrix/t[0]
    new_homography_matrix = np.matmul(intrinsics, rot_trans_matrix)

    [imagePoints, jacobian] = cv2.projectPoints(np.array(world_points_3d, dtype='float32'), rvec, tvec, intrinsics, distortion)
    
    warped_image = cv2.warpPerspective(undistorted_frame, new_homography_matrix, (maps_ss.shape[1],maps_ss.shape[0]) ,flags=cv2.WARP_INVERSE_MAP)
    cv2.imwrite(folder_name + '/'+'warped.png', warped_image)
    alpha = 0.6
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(warped_image, alpha, maps_ss, beta, 0.0)
    cv2.imwrite('../data/' + folder_name + '/'+'points_im.png', draw_frame_im)
    cv2.imwrite('../data/' + folder_name + '/'+'points_map.png', draw_frame_map)
    cv2.imwrite('../data/' + folder_name + '/'+'overlay.png', dst)
    cv2.imshow('warped',dst)
    cv2.imshow('im',draw_frame_im)
    cv2.imshow('map',draw_frame_map)
    cv2.waitKey(0)
    index = 1
    # perspective_transform_matrix = cv2.getPerspectiveTransform(np.array([image_points], dtype='float32'), np.array([world_points], dtype='float32'))

    # warped_image = cv2.warpPerspective(frame, perspective_transform_matrix, (2000, 2000))

    # cv2.imshow('warpedimage',warped_image)
    # cv2.imwrite('warpedimageopencv.png',warped_image)
    # transformed_pts = cv2.perspectiveTransform(np.array([image_points], dtype='float32'), perspective_transform_matrix)[0]
    # cv2.waitKey(0)
    # eng = matlab.engine.start_matlab()
    # rvec3_matlab = rvec3.tolist()
    # tvec_matlab = tvec.tolist()
    # eng.run_tracking(matlab.double(rvec3_matlab), matlab.double(tvec_matlab),'D:/projects/trans-plan/updated_algo/Refactored/Detections/','Essa_Burton_00000002.txt', nargout=0)
    # start_points = scipy.io.loadmat('D:/projects/trans-plan/updated_algo/Refactored/Detections/Essa_Burton_00000002.txtstart_points.mat')['start_points']
    # end_points = scipy.io.loadmat('D:/projects/trans-plan/updated_algo/Refactored/Detections/Essa_Burton_00000002.txtend_points.mat')['end_points']

    # index = 1
    
if __name__ == "__main__":
    main()