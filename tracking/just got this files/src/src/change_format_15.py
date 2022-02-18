import os
import pandas as pd
import numpy as np
import glob

folder = '/home/yufei/Downloads/December/Eglinton@Creditview/'
FPS = 60
NUM_FRAMES_15MIN = FPS*60*15
files_list = glob.glob(folder+'*.csv')
li = []
first = True
last_frame = 0
for fname in files_list:
    # detection = fname.split('/')[-1].split('.')[0].split("\\")[-1]
    detection = fname.split('/')[-1].split('.')[0].split("_")[1]
    print(detection)
    if not os.path.exists(folder + '15min_splits/' + detection):
        os.makedirs(folder + '15min_splits/' + detection)
    if first:
        df = pd.read_csv(fname,sep=r"[\t,]",names=['frame','class','x1','y1','x2','y2','conf'])
        first = False
        last_frame = df['frame'].values[-1]
    else:
        df = pd.read_csv(fname,sep=r"[\t,]",names=['frame','class','x1','y1','x2','y2','conf'])
        df['frame'] = df['frame'] +last_frame+1
        last_frame = df['frame'].values[-1]
    li.append(df)
frame = pd.concat(li, axis=0, ignore_index=True)
print(li)


    # frames = df['frame']
    # last_frame_num = int(frames.values[-1].split('.')[0])
    # last_frame_num = int(frames.values[-1])
    # vid_length_hr = int((last_frame_num/FPS)/60/60)
    # num_chunks = int(last_frame_num/NUM_FRAMES_15MIN) + 1
    # df_frame_nums = df['frame'].str.split('.')


df_frame_nums = frame['frame']
df_frame_nums_split = df_frame_nums.values.tolist()
# print(df_frame_nums_split)
frame_nums = [int(item) for item in df_frame_nums_split]
mask_1 = [frame_num < NUM_FRAMES_15MIN for frame_num in frame_nums]
mask_2 = [frame_num >= NUM_FRAMES_15MIN and frame_num < 2*NUM_FRAMES_15MIN for frame_num in frame_nums]
mask_3 = [frame_num >= 2*NUM_FRAMES_15MIN and frame_num < 3*NUM_FRAMES_15MIN for frame_num in frame_nums]
mask_4 = [frame_num >= 3*NUM_FRAMES_15MIN for frame_num in frame_nums]
chunk_1 = frame[mask_1]
chunk_2 = frame[mask_2]
if len(chunk_2):
    chunk_2['frame'] = chunk_2['frame']-NUM_FRAMES_15MIN
chunk_3 = frame[mask_3]
if len(chunk_3):
    chunk_3['frame'] = chunk_3['frame']-NUM_FRAMES_15MIN*2
chunk_4 = frame[mask_4]
if len(chunk_4):
    chunk_4['frame'] = chunk_4['frame']-NUM_FRAMES_15MIN*3
chunk_1.to_csv(folder + '15min_splits/'  + detection+ '/' + '00.txt', index = False, header = False)
chunk_2.to_csv(folder + '15min_splits/' + detection+ '/' + '15.txt', index = False, header = False)
chunk_3.to_csv(folder + '15min_splits/' + detection+ '/' + '30.txt', index = False, header = False)
chunk_4.to_csv(folder + '15min_splits/' + detection+ '/' + '45.txt', index = False, header = False)
index = 1

