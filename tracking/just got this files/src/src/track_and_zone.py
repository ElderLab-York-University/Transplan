import os
import glob
import cv2
import numpy as np
import scipy.io
import matlab.engine
import pandas as pd

curr_src_file_path = os.path.dirname(os.path.realpath(__file__))

folder = 'D:/projects/trans-plan/updated_algo/Refactored/data/EssaRoad_BurtonAvenue/'
tracks_folder = folder+'15min_splits'
rvec3 = np.genfromtxt(folder + '/'+'rvec3.csv', delimiter=',')
tvec = np.genfromtxt(folder + '/'+'tvec.csv', delimiter=',')
hourly_list = glob.glob(tracks_folder+'/*')
zone_nums = {'north':0,'east':1,'south':2,'west':3}
zone_files = glob.glob(folder+'*zone.csv')
zones = {}
zone_counts = {}
for zone_file in zone_files:
    zone_category = zone_file.split('_zone')
    zone_category = zone_category[0].split('\\')
    zone_category = zone_category[-1]
    zones[zone_category] = np.genfromtxt(zone_file, delimiter=',')
    zone_counts[zone_category] = {'L':0,'R':0,'T':0}

print('Starting matlab engine...')
eng = matlab.engine.start_matlab()
print('Matlab engine started.')
eng.cd(curr_src_file_path)

def addToZoneCounts(zone_counts, spoint, epoint, szone, ezone):
    if szone == '':
        print('Point not found!')
        return
    if ezone == '':
        zone_counts[szone]['T']+=1
        return
    turn = zone_nums[ezone] - zone_nums[szone]
    if turn == 2 or turn == -2:
        zone_counts[szone]['T']+=1
    if turn == 3 or turn == -1:
        zone_counts[szone]['R']+=1
    if turn == 1 or turn == -3:
        zone_counts[szone]['L']+=1
    #if turn == 0:
        #zone_counts[szone]['T']+=1
        #return

for hour_folder in hourly_list:
    detection_chunks = glob.glob(hour_folder+'/*.txt')
    
    for detection_chunk in detection_chunks:
        # for zone_file in zone_files:
        #     zone_category = zone_file.split('_zone')
        #     zone_category = zone_category[0].split('\\')
        #     zone_category = zone_category[-1]
        #     zones[zone_category] = np.genfromtxt(zone_file, delimiter=',')
        zone_counts = {'north':{'L':0,'T':0,'R':0},'east':{'L':0,'T':0,'R':0},'south':{'L':0,'T':0,'R':0},'west':{'L':0,'T':0,'R':0}}

        detection_filename = os.path.basename(detection_chunk)
        rvec3_matlab = rvec3.tolist()
        tvec_matlab = tvec.tolist()
        eng.run_tracking(matlab.double(rvec3_matlab), matlab.double(tvec_matlab),hour_folder+'/',detection_filename, nargout=0)
        start_points = scipy.io.loadmat(hour_folder+'/'+detection_filename[0:-4]+'start_points.mat')['start_points']
        end_points = scipy.io.loadmat(hour_folder+'/'+detection_filename[0:-4]+'end_points.mat')['end_points']

        for i in range(len(start_points)):
            spoint = start_points[i]
            epoint = end_points[i]
            sfound = False
            efound = False
            szone = ''
            ezone = ''
            for zone in zones:
                curr_zone_pts = zones[zone]
                spt_test = -2
                ept_test = -2
                if sfound and efound:
                    break
                if not sfound:
                    spt_test = cv2.pointPolygonTest(zones[zone].astype(int), (int(spoint[0]),int(spoint[1])), False)
                if not efound:
                    ept_test = cv2.pointPolygonTest(zones[zone].astype(int), (int(epoint[0]),int(epoint[1])), False)
                if spt_test>=0:#check_pt_in_rect(point, zone):
                    sfound = True
                    szone = zone
                if ept_test>=0:#check_pt_in_rect(point, zone):
                    efound = True
                    ezone = zone
            
            addToZoneCounts(zone_counts, spoint, epoint, szone, ezone)

        reformed_dict = {(outerKey, innerKey): values for outerKey, innerDict in zone_counts.items() for innerKey, values in innerDict.items()}
        df = pd.DataFrame(reformed_dict,index=[0])
        #df = df.transpose()
        df.to_excel(hour_folder+'/'+detection_filename[0:-4]+'_counts.xlsx')

print('Done!')


