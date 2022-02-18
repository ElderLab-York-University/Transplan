import numpy as np
import cv2
import csv
import numpy as np
import tkinter as tk
from tkinter import filedialog
#import matlab.engine
import drawhelper
import os

draw_frame = []

def order_points(pts):
    center_point = pts.sum(axis = 0)/4
    normalized_points = pts - center_point
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    #rect = np.zeros((4, 2), dtype='float32')
    left_pair = [pt for pt in normalized_points if pt[0] < 0]
    right_pair = [pt for pt in normalized_points if pt[0] >= 0]
    if left_pair[0][1] < left_pair[1][1]:
        tl = left_pair[0]
        bl = left_pair[1]
    else:
        tl = left_pair[1]
        bl = left_pair[0]
    if right_pair[0][1] < right_pair[1][1]:
        tr = right_pair[0]
        br = right_pair[1]
    else:
        tr = right_pair[1]
        br = right_pair[0]

    rect = [tl, tr, br, bl]
    rect = rect + center_point
    return rect    

def find_intersection_points_two_lines(refPt):

    x1,y1 = refPt[0][0], refPt[0][1]
    x2,y2 = refPt[1][0], refPt[1][1]

    x3, y3 = refPt[2][0], refPt[2][1]
    x4, y4 = refPt[3][0], refPt[3][1]
    intersection_pt_x = int(  (  (x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)  )/(  (x1 - x2)*(y3-y4) - (y1 - y2)*(x3 - x4)  )  )
    intersection_pt_y = int(  (  (x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)  )/(  (x1 - x2)*(y3-y4) - (y1 - y2)*(x3 - x4)  )  )

    return [intersection_pt_x, intersection_pt_y]

def find_intersection_points(refPt):
    intersection_points = []

    x1,y1 = refPt[0][0], refPt[0][1]
    x2,y2 = refPt[1][0], refPt[1][1]

    x3, y3 = refPt[4][0], refPt[4][1]
    x4, y4 = refPt[5][0], refPt[5][1]
    intersection_pt_x = int(  (  (x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)  )/(  (x1 - x2)*(y3-y4) - (y1 - y2)*(x3 - x4)  )  )
    intersection_pt_y = int(  (  (x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)  )/(  (x1 - x2)*(y3-y4) - (y1 - y2)*(x3 - x4)  )  )
    intersection_points.append((intersection_pt_x, intersection_pt_y))

    x3, y3 = refPt[6][0], refPt[6][1]
    x4, y4 = refPt[7][0], refPt[7][1]
    intersection_pt_x = int(  (  (x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)  )/(  (x1 - x2)*(y3-y4) - (y1 - y2)*(x3 - x4)  )  )
    intersection_pt_y = int(  (  (x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)  )/(  (x1 - x2)*(y3-y4) - (y1 - y2)*(x3 - x4)  )  )
    intersection_points.append((intersection_pt_x, intersection_pt_y))

    x1,y1 = refPt[2][0], refPt[2][1]
    x2,y2 = refPt[3][0], refPt[3][1]

    x3, y3 = refPt[4][0], refPt[4][1]
    x4, y4 = refPt[5][0], refPt[5][1]
    intersection_pt_x = int(  (  (x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)  )/(  (x1 - x2)*(y3-y4) - (y1 - y2)*(x3 - x4)  )  )
    intersection_pt_y = int(  (  (x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)  )/(  (x1 - x2)*(y3-y4) - (y1 - y2)*(x3 - x4)  )  )
    intersection_points.append((intersection_pt_x, intersection_pt_y))

    x3, y3 = refPt[6][0], refPt[6][1]
    x4, y4 = refPt[7][0], refPt[7][1]
    intersection_pt_x = int(  (  (x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)  )/(  (x1 - x2)*(y3-y4) - (y1 - y2)*(x3 - x4)  )  )
    intersection_pt_y = int(  (  (x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)  )/(  (x1 - x2)*(y3-y4) - (y1 - y2)*(x3 - x4)  )  )
    intersection_points.append((intersection_pt_x, intersection_pt_y))

    return intersection_points

def main():

    # Ask user to open video file
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    folder_path = os.path.dirname(os.path.abspath(file_path))
    folder_name = file_path.split('/')[-2]
    file_name = file_path.split('/')[-2] + '_' + file_path.split('/')[-1].split('.')[0]

    maps_ss = cv2.imread(file_path)

    # Ask and wait for the user to draw 4 lines
    image_points, draw_frame_im = drawhelper.getpoints(maps_ss, 12)

    draw_frame_zones = draw_frame_im.copy()
    
    

    intersection_points = find_intersection_points(image_points)
    rect = order_points(np.array(intersection_points, dtype="float32"))
    mid_pt = np.mean(rect,axis=0)
    rect = [[int(pt[0]),int(pt[1])] for pt in rect]
    #mid_pt = [[int(pt[0]),int(pt[1])] for pt in mid_pt]
    mid_pt = mid_pt.tolist()
    mid_pt[0] = int(mid_pt[0])
    mid_pt[1] = int(mid_pt[1])
    
    cv2.circle(draw_frame_im, (int(mid_pt[0]),int(mid_pt[1])), 5, (255,255,255), -1)
    for i, p in enumerate(rect):
        px = int(p[0])
        py = int(p[1])
        cv2.circle(draw_frame_im, (px, py), 3, (255,255,0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(draw_frame,str(px)+','+str(py),(px,py), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(draw_frame_im,str(i),(px,py), font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.imwrite(folder_path+'/intersection_drawing.png', draw_frame_im)

    

    west_zone = [image_points[0],rect[0],mid_pt,rect[3],image_points[2]]
    north_zone = [image_points[4],rect[0],mid_pt,rect[1],image_points[6]]
    east_zone = [image_points[1],rect[1],mid_pt,rect[2],image_points[3]]
    south_zone = [image_points[5],rect[3],mid_pt,rect[2],image_points[7]]

    # pt_0_mid = find_intersection_points_two_lines([rect[0], mid_pt, [0,0], [0, maps_ss.shape[0]]])
    # if pt_0_mid[0] < 0 or pt_0_mid[1] < 0:
    #     pt_0_mid = find_intersection_points_two_lines([rect[0], mid_pt, [0,0], [maps_ss.shape[1], 0]])
    #     west_zone.append([0,0])
    # else:
    #     north_zone.append([0,0])

    # north_zone.append(pt_0_mid)
    # north_zone.append(rect[0])
    # north_zone.append(mid_pt)
    # north_zone.append(rect[1])

    # west_zone.append(pt_0_mid)
    # west_zone.append(rect[0])
    # west_zone.append(mid_pt)
    # west_zone.append(rect[3])

    # pt_1_mid = find_intersection_points_two_lines([rect[1], mid_pt, [0,0], [maps_ss.shape[1], 0]])
    # if pt_1_mid[0] < 0 or pt_1_mid[1] < 0:
    #     pt_1_mid = find_intersection_points_two_lines([rect[1], mid_pt, [maps_ss.shape[1], 0], [maps_ss.shape[1], maps_ss.shape[0]]])
    #     north_zone.append(pt_1_mid)
    #     north_zone.append([maps_ss.shape[1], 0])
    # else:
    #     north_zone.append(pt_1_mid)
    #     east_zone.append([maps_ss.shape[1], 0])

    # east_zone.append(pt_1_mid)
    # east_zone.append(rect[1])
    # east_zone.append(mid_pt)
    # east_zone.append(rect[2])

    # pt_2_mid = find_intersection_points_two_lines([rect[2], mid_pt, [maps_ss.shape[1], maps_ss.shape[0]], [maps_ss.shape[1], 0]])
    # if pt_2_mid[0] < 0 or pt_2_mid[1] < 0:
    #     pt_2_mid = find_intersection_points_two_lines([rect[2], mid_pt, [maps_ss.shape[1], maps_ss.shape[0]], [0, maps_ss.shape[0]]])
    #     east_zone.append(pt_2_mid)
    #     east_zone.append([maps_ss.shape[1], maps_ss.shape[0]])
    # else:
    #     south_zone.append([maps_ss.shape[1], maps_ss.shape[0]])
    #     east_zone.append(pt_2_mid)

    # south_zone.append(pt_2_mid)
    # south_zone.append(rect[2])
    # south_zone.append(mid_pt)
    # south_zone.append(rect[3])

    # pt_3_mid = find_intersection_points_two_lines([rect[3], mid_pt, [0,0], [0, maps_ss.shape[0]]])
    # if pt_3_mid[0] < 0 or pt_3_mid[1] < 0:
    #     pt_3_mid = find_intersection_points_two_lines([rect[3], mid_pt, [0, maps_ss.shape[0]], [maps_ss.shape[1], maps_ss.shape[0]]])
    #     west_zone.append(pt_3_mid)
    #     west_zone.append([0, maps_ss.shape[0]])
    #     south_zone.append(pt_3_mid)
    # else:
    #     south_zone.append(pt_3_mid)
    #     south_zone.append([0, maps_ss.shape[0]])
    #     west_zone.append(pt_3_mid)

    zone_colors = [(128,0,128),(0,0,255),(0,255,0),(255,0,0)]
    cv2.drawContours(draw_frame_zones, np.array([west_zone]), 0, zone_colors[0],thickness=-1)
    cv2.imshow('image',draw_frame_zones)
    cv2.waitKey(0)
    cv2.drawContours(draw_frame_zones, np.array([north_zone]), 0, zone_colors[1],thickness=-1)
    cv2.imshow('image',draw_frame_zones)
    cv2.waitKey(0)
    cv2.drawContours(draw_frame_zones, np.array([east_zone]), 0, zone_colors[2],thickness=-1)
    cv2.imshow('image',draw_frame_zones)
    cv2.waitKey(0)
    cv2.drawContours(draw_frame_zones, np.array([south_zone]), 0, zone_colors[3],thickness=-1)
    cv2.imshow('image',draw_frame_zones)
    cv2.waitKey(0)

    np.savetxt(folder_path+'/west_zone.csv', np.array(west_zone), delimiter=',')
    np.savetxt(folder_path+'/east_zone.csv', np.array(east_zone), delimiter=',')
    np.savetxt(folder_path+'/north_zone.csv', np.array(north_zone), delimiter=',')
    np.savetxt(folder_path+'/south_zone.csv', np.array(south_zone), delimiter=',')

    alpha = 0.5  # Transparency factor.
    draw_frame_zones = cv2.addWeighted(draw_frame_im, alpha, draw_frame_zones, 1 - alpha, 0)
    cv2.imwrite(folder_path+'/intersection_drawing_zones.png', draw_frame_zones)
    #cv2.waitKey(0)
    index = 1

if __name__ == "__main__":
    main()    