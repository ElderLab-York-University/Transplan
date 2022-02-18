import pandas as pd
import glob
import numpy as np



path = "D:/projects/trans-plan/updated_algo/Refactored/data/EssaRoad_GowanStreet/15min_splits/"
hourly_list = glob.glob(path+'*')
col_names = ['NL','NT','NR','EL','ET','ER','SL','ST','SR']#,'WL','WT','WR']
full_df = pd.DataFrame(columns=col_names)
for hour_folder in hourly_list:
    detection_chunks = glob.glob(hour_folder+'/*.xlsx')
    for detection_chunk in detection_chunks:
        df = pd.read_excel(detection_chunk)
        df_north_col = df['north']
        b, c = df_north_col.iloc[1], df_north_col.iloc[2]
        temp = df_north_col.iloc[1].copy()
        df_north_col.iloc[1] = c
        df_north_col.iloc[2] = temp

        df_east_col = df['east']
        b, c = df_east_col.iloc[1], df_east_col.iloc[2]
        temp = df_east_col.iloc[1].copy()
        df_east_col.iloc[1] = c
        df_east_col.iloc[2] = temp



        df_south_col = df['south']
        b, c = df_south_col.iloc[1], df_south_col.iloc[2]
        temp = df_south_col.iloc[1].copy()
        df_south_col.iloc[1] = c
        df_south_col.iloc[2] = temp


        # df_west_col = df['west']
        # b, c = df_west_col.iloc[1], df_west_col.iloc[2]
        # temp = df_west_col.iloc[1].copy()
        # df_west_col.iloc[1] = c
        # df_west_col.iloc[2] = temp

        new_df = pd.concat([df_north_col,df_east_col,df_south_col])#,df_west_col])
        new_df = new_df.to_frame()
        #new_df_t = pd.DataFrame([[new_df.values]],columns = col_names)
        new_df_t = new_df.T
        #full_df = pd.concat([full_df,new_df])
        #print(new_df_t.columns)
        new_df_t.columns = ['NL','NT','NR','EL','ET','ER','SL','ST','SR']#,'WL','WT','WR']
        #print(new_df_t.columns)
        full_df = full_df.append(new_df_t,ignore_index=True)
        #print(full_df.head())

full_df.to_excel(path+'ful_df_merged_counts.xlsx')







