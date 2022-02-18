import cv2
import numpy as np

img_pts = np.array([[171.4856, 221.0246],[530.0677,195.7531],[405.0762, 161.6025],[525.9696,177.9948]], dtype='float32')
img_pts_zeros = np.array([[171.4856, 221.0246, 0],[530.0677,195.7531, 0],[405.0762, 161.6025, 0],[525.9696,177.9948, 0]], dtype='float32')
map_pts = np.array([[399.7839,361.4382],[473.6787, 250.5961],[347.9549, 144.3723],[463.4155, 190.0434]], dtype='float32')
map_pts_zeros = np.array([[399.7839,361.4382,0],[473.6787, 250.5961,0],[347.9549, 144.3723,0],[463.4155, 190.0434,0]], dtype='float32')
homography_matrix = cv2.getPerspectiveTransform(map_pts,img_pts)

img_pts_back = np.matmul(homography_matrix, map_pts_zeros[0])
img_pts_back = [pt/img_pts_back[2] for pt in img_pts_back]
frame = cv2.imread('D:/projects/trans-plan/updated_algo/Refactored/calibresult_matlab.png')
map_img = cv2.imread('D:/projects/trans-plan/updated_algo/Refactored/EssaRoad_BurtonAvenue.png')


# warped_image = cv2.warpPerspective(frame, homography_matrix, (788, 559),flags=cv2.WARP_INVERSE_MAP)

# alpha = 0.8
# beta = (1.0 - alpha)
# dst = cv2.addWeighted(warped_image, alpha, map_img, beta, 0.0)
# cv2.imshow('warped',dst)
# cv2.waitKey(0)


# # Read camera matrices from file
intrinsics = np.loadtxt('intrinsic_params_matlab.txt')
intrinsics = intrinsics.transpose()
distortion = np.loadtxt('distortion_coefficients_matlab.txt')



# pointsOut = cv2.perspectiveTransform(np.array([map_pts]), np.linalg.inv(homography_matrix))

[retval, rvec, tvec] = cv2.solvePnP(map_pts_zeros, img_pts, intrinsics, distortion)
[rvec3, jacobian] = cv2.Rodrigues(rvec)

# [retvalim, rvecim, tvecim] = cv2.solvePnP(img_pts_zeros, map_pts, intrinsics, distortion)
# [rvec3im, jacobianim] = cv2.Rodrigues(rvecim)


[imagePoints, jacobian] = cv2.projectPoints(map_pts_zeros, rvec, tvec, intrinsics, distortion)

cam_location = np.matmul(np.linalg.inv(-rvec3),(tvec))

rot_trans_matrix = [rvec3[:,0], rvec3[:,1],np.transpose(tvec[:,0])]
rot_trans_matrix = np.transpose(rot_trans_matrix)
t = tvec[2]
rot_trans_matrix = rot_trans_matrix/t[0]
new_homography_matrix = np.matmul(intrinsics, rot_trans_matrix)
#################################################################
img_pts_back = np.matmul(new_homography_matrix, map_pts_zeros[0])
img_pts_back = [pt/img_pts_back[2] for pt in img_pts_back]
#################################################################
index = 1
map_img_shape = map_img.shape
two_dshape = map_img_shape[0:2]
warped_image = cv2.warpPerspective(frame, new_homography_matrix, (map_img.shape[1],map_img.shape[0]) ,flags=cv2.WARP_INVERSE_MAP)
alpha = 0.8
beta = (1.0 - alpha)
dst = cv2.addWeighted(warped_image, alpha, map_img, beta, 0.0)
cv2.imshow('warped',dst)
cv2.waitKey(0)
# blank_image = np.zeros((1000,1000,3), np.uint8)

# for y in range(0, frame.shape[0]):
#     for x in range(0, frame.shape[1]):
#         pixel = frame[y,x]
#         [imagePoints, jacobian] = cv2.projectPoints(np.array([[x,y,0]],dtype='float32'), rvecim, tvecim, intrinsics, distortion)
#         imp0 = imagePoints[0][0][0]
#         imp1 = imagePoints[0][0][1]
#         if imp0 > 999 or imp1 > 999 or imp0 < 0 or imp1 < 0:
#             continue
#         temp_blank = blank_image[int(imp0)][int(imp1)]
#         temp_pixel = np.array([pixel])
#         blank_image[int(imp1)][int(imp0)] = np.array([pixel])[0]

# cv2.imshow('transformed?',blank_image)
# cv2.imwrite('transformedim.png',blank_image)
# cv2.waitKey(0)

# index = 1
