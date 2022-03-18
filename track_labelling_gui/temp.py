import numpy as np
import pandas as pd
file_name = "/home/savoji/Desktop/TransPlan Project/Results/GX010069_tracking_sort_reprojected.txt"
data = np.loadtxt(file_name, delimiter=",")
df = pd.DataFrame(data=data, columns=["fn", "id", "x", "y"])
tids = np.unique(df['id'].tolist())
id_index = 