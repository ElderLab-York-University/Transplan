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

def intersection_between_two_lines(x1,y1,x2,y2,x3,y3,x4,y4):
    intersection_pt_x = int(  (  (x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)  )/(  (x1 - x2)*(y3-y4) - (y1 - y2)*(x3 - x4)  )  )
    intersection_pt_y = int(  (  (x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)  )/(  (x1 - x2)*(y3-y4) - (y1 - y2)*(x3 - x4)  )  )

    return [intersection_pt_x, intersection_pt_y]

def find_intersection_points(refPt):
    intersection_points = []

    x1_1,y1_1 = refPt[0][0], refPt[0][1]
    x1_2,y1_2 = refPt[1][0], refPt[1][1]

    x2_1,y2_1 = refPt[2][0], refPt[2][1]
    x2_2,y2_2 = refPt[3][0], refPt[3][1]

    x3_1,y3_1 = refPt[4][0], refPt[4][1]
    x3_2,y3_2 = refPt[5][0], refPt[5][1]

    x4_1,y4_1 = refPt[6][0], refPt[6][1]
    x4_2,y4_2 = refPt[7][0], refPt[7][1]

    x5_1,y5_1 = refPt[8][0], refPt[8][1]
    x5_2,y5_2 = refPt[9][0], refPt[9][1]

    x6_1,y6_1 = refPt[10][0], refPt[10][1]
    x6_2,y6_2 = refPt[11][0], refPt[11][1]

    i14 = intersection_between_two_lines(x1_1,y1_1, x1_2,y1_2, x4_1,y4_1, x4_2,y4_2)
    i15 = intersection_between_two_lines(x1_1,y1_1, x1_2,y1_2, x5_1,y5_1, x5_2,y5_2)
    i16 = intersection_between_two_lines(x1_1,y1_1, x1_2,y1_2, x6_1,y6_1, x6_2,y6_2)

    i24 = intersection_between_two_lines(x2_1,y2_1, x2_2,y2_2, x4_1,y4_1, x4_2,y4_2)
    i25 = intersection_between_two_lines(x2_1,y2_1, x2_2,y2_2, x5_1,y5_1, x5_2,y5_2)
    i26 = intersection_between_two_lines(x2_1,y2_1, x2_2,y2_2, x6_1,y6_1, x6_2,y6_2)

    i34 = intersection_between_two_lines(x3_1,y3_1, x3_2,y3_2, x4_1,y4_1, x4_2,y4_2)
    i35 = intersection_between_two_lines(x3_1,y3_1, x3_2,y3_2, x5_1,y5_1, x5_2,y5_2)
    i36 = intersection_between_two_lines(x3_1,y3_1, x3_2,y3_2, x6_1,y6_1, x6_2,y6_2)

    intersection_points = [i14, i15, i16, i24, i25, i26, i34, i35, i36]
    
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
    np.savetxt(folder_path+'/'+file_name+'_9points.txt',intersection_points)
    for i, p in enumerate(intersection_points):
        px = int(p[0])
        py = int(p[1])
        cv2.circle(draw_frame_im, (px, py), 3, (255,255,0), -1)
    cv2.imshow('image', draw_frame_im)
    cv2.waitKey(0)
    cv2.imwrite(folder_path+'/'+file_name+'draw_intersection_9points.png',draw_frame_im)

if __name__ == "__main__":
    main()    