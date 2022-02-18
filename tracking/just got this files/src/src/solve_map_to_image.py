import cv2
import math
import numpy as np
import os
import utm

from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
curr_folder = os.path.dirname(os.path.abspath(__file__))


latlng_pairs = [(43.84455591119263, -79.38254557831642),
(43.84463909323816, -79.38217369343563),
(43.84463425707591, -79.38213346030041),
(43.844426412015665, -79.38206695329298),
(43.84422492050792, -79.38208371709932),
(43.84417414043204, -79.38228220056642),
(43.844399023297015, -79.382521587721)]

listPt = [(131, 469),
(96, 229),
(116, 220),
(379, 190),
(606, 197),
(731, 220),
(709, 327)]

r = 6371000 # meters
phi_0 = latlng_pairs[0][1]
cos_phi_0 = math.cos(math.radians(phi_0))

def to_xy(point, r, cos_phi_0):
    lam = point[0]
    phi = point[1]
    return (r * math.radians(lam) * cos_phi_0, r * math.radians(phi))

utmx = []
utmy = []

for point in latlng_pairs:
    global z, zl
    x, y, z, zl = utm.from_latlon(point[0], point[1])
    utmx.append(x)
    utmy.append(y)


xy = []
for point in latlng_pairs:
    xy.append(to_xy(point, r, cos_phi_0))

lat = [x[0] for x in latlng_pairs]
lng = [x[1] for x in latlng_pairs]

y = [x[0] for x in xy]
x = [x[1] for x in xy]

lat_o_mean = lat - np.mean(lat)
lng_o_mean = lng - np.mean(lng)

x_o_mean = x - np.mean(x)
y_o_mean = y - np.mean(y)


#lat_o_mean_norm = lat_o_mean / 100000
#lng_o_mean_norm = lng_o_mean / 100000


#plt.xticks(np.arange(min(x_o_mean), max(x_o_mean), 0.05))
#plt.yticks(np.arange(min(y_o_mean), max(y_o_mean), 0.05))
#plt.show()
#plt.figure()
#plt.scatter(lng_o_mean_norm, lat_o_mean_norm)


# image = cv2.imread(file_path)
# h,  w = image.shape[:2]

ptnum = 0
# def clicks(event, x, y, flags, param):
#     global listPt, image, ptnum

#     if event == cv2.EVENT_LBUTTONUP:
#         # record the ending (x, y) coordinates and indicate that
#         # the cropping operation is finished
#         listPt.append([x, y])
#         ptnum+=1
#         cv2.circle(image,(x,y), 20, (255,255,255), 5)
#         cv2.putText(image, str(ptnum), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5, cv2.LINE_AA) 
#         cv2.imshow("image", image)

# cv2.namedWindow("image",cv2.WINDOW_NORMAL)

# cv2.resizeWindow("image", (int(w/5), int(h/5)))
# cv2.setMouseCallback("image", clicks)
# cv2.imshow("image",image)
# cv2.resizeWindow("image", (int(w/3), int(h/3)))
# cv2.waitKey(0)

world_points_3d = [[x[0],x[1],200] for x in zip(utmx,utmy)]
np.save('worldpts.npy',np.array(world_points_3d))
# Read camera matrices from file
# Load intrinsics and distortion
# Read camera matrices from file
intrinsics = np.loadtxt('../extra_data/intrinsic_params_matlab.txt')
intrinsics = intrinsics.transpose()
distortion = np.loadtxt('../extra_data/distortion_coefficients_matlab.txt')

[retval, rvec, tvec] = cv2.solvePnP(np.array(world_points_3d, dtype='float32'), np.array(listPt, dtype='float32'), intrinsics, distortion)
[rvec3, jacobian] = cv2.Rodrigues(rvec)

#rvec3_file_read = np.genfromtxt('../data/' + folder_name + '/'+'rvec3.csv', delimiter=',')
cam_location = np.matmul(np.linalg.inv(-rvec3),(tvec))
roll = math.atan2(-rvec3[2][1], rvec3[2][2])
pitch = math.asin(rvec3[2][0])
yaw = math.atan2(-rvec3[1][0], rvec3[0][0])
cam_height = cam_location[2]
#cv2.circle(image, (cam_location[0],cam_location[1]), 15, (0, 0, 255), 3)



latlongcam = utm.to_latlon(cam_location[0], cam_location[1], z, zl)
la = latlongcam[0].tolist()
lo = latlongcam[1].tolist()
print("cam location = {}, {}, roll = {}, pitch = {}, yaw = {}, cam height = {}".format(la[0],lo[0],roll*180/math.pi, pitch*180/math.pi, yaw*180/math.pi, cam_height))
plt.figure()
plt.scatter(utmx, utmy)
plt.scatter(cam_location[0],cam_location[1],s=40, c='r')
plt.text(cam_location[0],cam_location[1], str(la[0])+','+str(lo[0]), fontsize=12)
plt.show()

