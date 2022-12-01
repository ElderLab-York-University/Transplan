# from Libs import *
# from Utils import *

# def resample1(traj):
#     pass

# def resample2(traj):
#     pass

# def group_tracks_by_id(df):
#     # this function was writtern for grouping the tracks with the same id
#     # usinig this one can load the data from a .txt file rather than .mat file
#     all_ids = np.unique(df['id'].to_numpy(dtype=np.int64))
#     data = {"id":[], "trajectory":[], "frames":[]}
#     for idd in tqdm(all_ids):
#         frames = df[df['id']==idd]["fn"].to_numpy(np.float32)
#         id = idd
#         trajectory = df[df['id']==idd][["x", "y"]].to_numpy(np.float32)
        
#         data["id"].append(id)
#         data["frames"].append(frames)
#         data["trajectory"].append(trajectory)
#     df2 = pd.DataFrame(data)
#     return df2

import matplotlib.pyplot as plt
import sympy

# # load the tracks
# df = pd.read_pickle("./../Dataset/DandasStAtNinthLineFull/Results/Tracking/video.tracking.detectron2.sort.reprojected.pkl")
# top_image_path = "./../Dataset/DandasStAtNinthLineFull/video.homography.top.png"
# img = cv.imread(top_image_path)

# unique_track_ids = np.unique(df['id'])
# for  track_id in tqdm(unique_track_ids):
#     df_id = df[df['id']==track_id]


    

# for i, row in df.iterrows():
#     # show the pictures
 
#     command = input()
#     if command=="save":
        # save image

P11 = sympy.Point((1374.75, 230.7))
P12 = sympy.Point((2102.3, 371.7))
L1 = sympy.Line(P11, P12)

P21 = sympy.Point((1809, 950))
P22 = sympy.Point((2106.7, 676.2))
L2 = sympy.Line(P21, P22)

P31 = sympy.Point((120.5, 952.8))
P32 = sympy.Point((149.5, 367.2))
L3 = sympy.Line(P31, P32)

P41 = sympy.Point((310.7, 285.4))
P42 = sympy.Point((1137.4, 208.2))
L4 = sympy.Line(P41, P42)

J12 = L1.intersection(L2)
J23 = L2.intersection(L3)
J34 = L3.intersection(L4)
J41 = L4.intersection(L1)

J1 = (float(J12[0][0]), float(J12[0][1]))
J2 = (float(J23[0][0]), float(J23[0][1]))
J3 = (float(J34[0][0]), float(J34[0][1]))
J4 = (float(J41[0][0]), float(J41[0][1]))

print(J1)
print(J2)
print(J3)
print(J4)


src = "/home/savoji/Desktop/TransPlanProject/Dataset/SOW_src1/src1.homography.street.png"
img = plt.imread(src)
plt.imshow(img)
# plt.scatter(J1[0], J1[1], color='red')
# plt.scatter(J2[0], J2[1], color='red')
# plt.scatter(J3[0], J3[1], color='red')
# plt.scatter(J4[0], J4[1], color='red')
print(img.shape)
plt.show()

