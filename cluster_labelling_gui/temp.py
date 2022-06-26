import numpy as np
import pandas as pd
from pymatreader import read_mat
file_path = "/home/savoji/Desktop/TransPlan Project/Results/GX010069_tracking_sort_reprojected.csv"
data = read_mat(file_path)
df1 = pd.DataFrame(data['recorded_tracks'])
df = df1[['id', 'trajectory']]
df.columns = ['id', 'trajectory']
print([df])
# print(data['recorded_tracks']['trajectory'][0])
# file_name = "/home/savoji/Desktop/TransPlan Project/Results/GX010069_tracking_sort_reprojected.txt"
# data = np.loadtxt(file_name, delimiter=",")
# df = pd.DataFrame(data=data, columns=["fn", "id", "x", "y"])
# tids = np.unique(df['id'].tolist())
# id_index = 