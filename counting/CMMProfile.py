# this code profiles cmm_truncate code
# assuming equal size for both trajectories
# metrics include:
#                 - time
#                 - memory


trajectory_lengths = [1e1, 1e2, 1e3, 1e4]
import cProfile
import glob
import re
import time
import json
import ctypes
from collections import defaultdict
import cv2
import matplotlib
import numpy as np
from pymatreader import read_mat
import pandas as pd
from matplotlib import cm
from resample_gt_MOI.resample_typical_tracks import track_resample
from tqdm import tqdm

def cmm_distance(traj_a, traj_b, cmmlib):
    cmmlib.cmm_truncate_sides(traj_a[:, 0], traj_a[:, 1], traj_b[:, 0], traj_b[:, 1], traj_a.shape[0],
                                        traj_b.shape[0])


libfile = "./cmm_truncate_linux/build/lib.linux-x86_64-3.7/cmm.cpython-37m-x86_64-linux-gnu.so"
# libfile = 'York/Elderlab/yorku_pipeline_deepsort_features/yorku_pipeline_deepsort_features/cmm_truncate_linux/build/lib.linux-x86_64-3.7/cmm.cpython-37m-x86_64-linux-gnu.so'
cmmlib = ctypes.CDLL(libfile)

# 2. tell Python the argument and result types of function cmm
cmmlib.cmm_truncate_sides.restype = ctypes.c_double
cmmlib.cmm_truncate_sides.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64),
                                            np.ctypeslib.ndpointer(dtype=np.float64),
                                            np.ctypeslib.ndpointer(dtype=np.float64),
                                            np.ctypeslib.ndpointer(dtype=np.float64),
                                            ctypes.c_int, ctypes.c_int]

for tl in trajectory_lengths:
    print(f"profiling for trajectory length {tl}")
    traj_a, traj_b = np.random.randn(int(tl),2).astype(np.float64) , np.random.randn(int(tl),2).astype(np.float64)
    cProfile.run('cmm_distance(traj_a, traj_b, cmmlib)') 