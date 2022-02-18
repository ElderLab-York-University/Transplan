import os
import pandas as pd
import numpy as np
import glob

folder = 'D:/projects/trans-plan/updated_algo/Refactored/data/FerndaleDrive_WildwoodTrail/'
FPS = 15
NUM_FRAMES_15MIN = FPS*60*15
files_list = glob.glob(folder+'*.txt')

for fname in files_list:
    detection = fname.split('/')[-1].split('.')[0].split("\\")[-1]
    if not os.path.exists(folder + '15min_splits/' + detection):
        os.makedirs(folder + '15min_splits/' + detection)
    df = pd.read_csv(fname,sep=r"[\t,]",names=['frame','x1','y1','x2','y2','conf'])
    frames = df['frame']
    last_frame_num = int(frames.values[-1].split('.')[0])
    vid_length_hr = int((last_frame_num/FPS)/60/60)
    num_chunks = int(last_frame_num/NUM_FRAMES_15MIN) + 1
    df_frame_nums = df['frame'].str.split('.')
    df_frame_nums_split = df_frame_nums.values.tolist()
    frame_nums = [int(item[0]) for item in df_frame_nums_split]
    mask_1 = [frame_num < NUM_FRAMES_15MIN for frame_num in frame_nums]
    mask_2 = [frame_num >= NUM_FRAMES_15MIN and frame_num < 2*NUM_FRAMES_15MIN for frame_num in frame_nums]
    mask_3 = [frame_num >= 2*NUM_FRAMES_15MIN and frame_num < 3*NUM_FRAMES_15MIN for frame_num in frame_nums]
    mask_4 = [frame_num >= 3*NUM_FRAMES_15MIN for frame_num in frame_nums]
    chunk_1 = df[mask_1]
    chunk_2 = df[mask_2]
    chunk_3 = df[mask_3]
    chunk_4 = df[mask_4]
    chunk_1.to_csv(folder + '15min_splits/'  + detection+ '/' + '00.txt', index = False, header = False)
    chunk_2.to_csv(folder + '15min_splits/' + detection+ '/' + '15.txt', index = False, header = False)
    chunk_3.to_csv(folder + '15min_splits/' + detection+ '/' + '30.txt', index = False, header = False)
    chunk_4.to_csv(folder + '15min_splits/' + detection+ '/' + '45.txt', index = False, header = False)
    index = 1
    # chunk = []
    # for row_num,row in df.iterrows():
    #     frame_num = int(row['frame'].split('.')[0])
    #     if frame_num < NUM_FRAMES_15MIN*index:
    #         rowvals = row.values.tolist()
    #         chunk.append(rowvals)
    #     else:

    #         index = index + 1
