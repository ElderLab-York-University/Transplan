import os
import pandas as pd

data_folder = 'D:/projects/trans-plan/updated_algo/Refactored/data/original/'
folder_names = os.listdir(data_folder)

for folder in folder_names:
    sub_folder_names = os.listdir(data_folder+folder)
    for sub_folder in sub_folder_names:
        intersections = os.listdir(data_folder+folder+'/'+sub_folder)
        for intersection in intersections:
            new_folder_name = intersection.replace(" ", "")
            if not os.path.exists(data_folder + new_folder_name):
                os.makedirs(data_folder + new_folder_name)
            videos = os.listdir(data_folder+folder+'/'+sub_folder+'/'+intersection)
            for video in videos:
                rename_str = intersection.replace(" ","") + '_' + video
                if not os.path.exists(data_folder + new_folder_name+'/'+rename_str):
                    os.rename(data_folder+folder+'/'+sub_folder+'/'+intersection+'/'+video,data_folder + new_folder_name+'/'+rename_str)