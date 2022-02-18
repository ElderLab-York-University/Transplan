import cv2
import numpy as np
import scipy.io


start_points = scipy.io.loadmat("D:/projects/trans-plan/updated_algo/Refactored/data/EssaRoad_BurtonAvenue/15min_splits/EssaRoad_BurtonAvenue_00000001_7/15start_points.mat")['start_points']
end_points = scipy.io.loadmat('D:/projects/trans-plan/updated_algo/Refactored/data/EssaRoad_BurtonAvenue/15min_splits/EssaRoad_BurtonAvenue_00000001_7/15end_points.mat')['end_points']

folder = 'D:/projects/trans-plan/updated_algo/Refactored/data/EssaRoad_BurtonAvenue/'

zone_files = glob.glob(folder+'*zone.csv')
zones = {}
zone_counts = {}
for zone_file in zone_files:
    zone_category = zone_file.split('_zone')
    zone_category = zone_category[0].split('\\')
    zone_category = zone_category[-1]
    zones[zone_category] = np.genfromtxt(zone_file, delimiter=',')
    zone_counts[zone_category] = {'L':0,'R':0,'T':0}

zone_colors={'north':[255,0,0],'east':[0,255,0],'west':[0,0,255],'south':[0,128,255]}
for i in range(len(start_points)):
    spoint = start_points[i]
    for zone in zones:
        spt_test = cv2.pointPolygonTest(zones[zone].astype(int), (int(spoint[0]),int(spoint[1])), False)

