import os
import glob
import cv2
import numpy as np
import scipy.io
import matlab.engine
import pandas as pd
import time
folder = 'D:/projects/trans-plan/data/2020/May/351cm_height'
tracks_folder = folder+'/'
curr_folder = os.path.dirname(os.path.abspath(__file__))
rvec3 = np.genfromtxt('rvec3.csv', delimiter=',')
tvec = np.genfromtxt('tvec.csv', delimiter=',')
hourly_list = glob.glob(tracks_folder+'/*')
homography_matrix = np.load('homography_matrix_24pts.npy',allow_pickle=True)

print('Starting matlab engine...')
eng = matlab.engine.start_matlab()
print('Matlab engine started.')


eng.cd(r'D:/projects/trans-plan/updated_algo/Refactored/transplan/src')

detection_filename = "D:/projects/trans-plan/data/2020/May/351cm_height/faster_rcnn_inception_v2_coco_2018_01_28GXAB0755.MP4_detection_test.csv"
#curr_folder+'/data/20200224_153147_SSD512_filtered.csv'#os.path.basename(detection_chunk)
rvec3_matlab = rvec3.tolist()
tvec_matlab = tvec.tolist()
h = homography_matrix[0].tolist()
start = time.time()
eng.run_tracking(matlab.double(h), matlab.double(rvec3_matlab), matlab.double(tvec_matlab),'',detection_filename, nargout=0)
end = time.time()
print('Time taken = '+str(end-start))
