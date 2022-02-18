import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def getpoints(input_frame, NUMBER_OF_POINTS=6):
    global draw_frame, number_of_points_clicked, NUMBER_OF_POINTS_TRANSFORM, draw_window, frame, base_frame, draw_colors
    NUMBER_OF_POINTS_TRANSFORM = NUMBER_OF_POINTS
    viridis = cm.get_cmap('viridis', math.floor(NUMBER_OF_POINTS/2))
    draw_colors =  viridis.colors*255#[(255,0,0),(250,255,75),(0,0,255),(75,250,255)]
    draw_frame = input_frame.copy()
    frame = input_frame.copy()
    base_frame = input_frame.copy()
    while number_of_points_clicked < NUMBER_OF_POINTS_TRANSFORM:
        cv2.imshow(draw_window, draw_frame)
        key = cv2.waitKey(1) & 0xFF
        # if the 'c' key is pressed, break from the loop
        if key == ord("c"):
            number_of_points_clicked = 0
            break
    
    if number_of_points_clicked != NUMBER_OF_POINTS_TRANSFORM:
        print('Not enough points')
        exit()

    cv2.imshow(draw_window, draw_frame)
    cv2.waitKey(1)
    # Reset the callback on the window
    cv2.setMouseCallback(draw_window, empty_func)
    return refPt, draw_frame

def empty_func(event, x, y, flags, param):
    do_nothing = 1

def click_points(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, draw_frame, click_down, number_of_points_clicked, frame, draw_colors_index, draw_colors
    
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        click_down = True
        print('Left click detected')
        print(number_of_points_clicked)
        number_of_points_clicked = number_of_points_clicked + 1
        #draw_frame = frame.copy()
        refPt.append((x, y))
        cv2.circle(draw_frame, (x,y), 5, (0, 255, 0), -1 )
        

    elif event == cv2.EVENT_MOUSEMOVE:
        print(click_down)
        if number_of_points_clicked > 0 and click_down:
            print('Move!')
            draw_frame = frame.copy()
            cv2.line(draw_frame, refPt[number_of_points_clicked-1], (x,y), draw_colors[draw_colors_index], 3)
            cv2.circle(draw_frame,  refPt[number_of_points_clicked-1], 5, (0, 255, 0), -1 )
            cv2.putText(draw_frame,str(int(number_of_points_clicked/2)), (int((refPt[number_of_points_clicked-1][0]+x)/2),int((refPt[number_of_points_clicked-1][1]+y)/2)), cv2.FONT_HERSHEY_SIMPLEX, 3, 255, 3)

    if event == cv2.EVENT_LBUTTONUP:
        click_down = False
        refPt.append((x, y))
        number_of_points_clicked = number_of_points_clicked + 1
        print('Left click detected')
        print(number_of_points_clicked)
        cv2.circle(draw_frame, (x,y), 5, (0, 255, 0), -1 )
        
        draw_colors_index+=1
        frame = draw_frame.copy()

def reset():
    global draw_colors_index, refPt, click_down, number_of_points_clicked, NUMBER_OF_POINTS_TRANSFORM
    global draw_frame, frame, draw_window
    draw_colors_index = 0
    refPt = []
    click_down = 0
    number_of_points_clicked = 0
    NUMBER_OF_POINTS_TRANSFORM = 8
    draw_frame = []
    frame = []
    # Set callback for a window to collect clicks
    cv2.namedWindow(draw_window)
    cv2.setMouseCallback(draw_window, click_points)

draw_colors_index = 0
refPt = []
click_down = 0
number_of_points_clicked = 0
NUMBER_OF_POINTS_TRANSFORM = 8
NUMBER_OF_POINTS = 4
draw_frame = []
frame = []
draw_window = 'Draw ' + str(NUMBER_OF_POINTS) + ' lines on the ground plane'
# Set callback for a window to collect clicks
cv2.namedWindow(draw_window,cv2.WINDOW_NORMAL)
cv2.resizeWindow(draw_window, 1536, 864)
cv2.setMouseCallback(draw_window, click_points)