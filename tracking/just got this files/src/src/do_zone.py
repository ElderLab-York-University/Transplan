import os
import glob
import cv2
import numpy as np
import scipy.io
import matlab.engine
import pandas as pd


folder = 'D:/projects/trans-plan/data/2020/May/351cm_height'
tracks_folder = folder+'/'
curr_folder = os.path.dirname(os.path.abspath(__file__))


zone_nums = {'north':0,'east':1,'south':2,'west':3}
zone_files = glob.glob('D:/projects/trans-plan/updated_algo/Refactored/transplan/*zone.csv')
zones = {}
zone_counts = {}
for zone_file in zone_files:
    zone_category = zone_file.split('_zone')
    zone_category = zone_category[0].split('\\')
    zone_category = zone_category[-1]
    zones[zone_category] = np.genfromtxt(zone_file, delimiter=',')
    zone_counts[zone_category] = {'L':0,'T':0,'R':0}



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
    

start_points = scipy.io.loadmat(folder+'/detections/merged/60minsstart_points.mat')['start_points']
end_points = scipy.io.loadmat(folder+'/detections/merged/60minsend_points.mat')['end_points']

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

df = pd.DataFrame(zone_counts)
df.transpose().to_excel(folder+'/detections/merged/60mins_zoned.xlsx')

print('Done!')