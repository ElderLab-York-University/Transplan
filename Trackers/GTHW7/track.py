from Libs import *
from Utils import *

def track(args, detectors):
    input_file=args.GTJson
    camera_name=os.path.abspath(args.Dataset).split("/")[-1][-3:]
    name_to_num={
      "lc1":0,
      "lc2":1,
      "sc1":2,
      "sc2":3,
      "sc3":4,
      "sc4":5
    }    
    camera_num= name_to_num[camera_name]    
    f= open(input_file ,'r')

    data= json.load(f)
    id_counter=0
    f.close()
    skip=6
    start=args.StartFrame if args.StartFrame is not None else 0
    i=0
    detections=[]
    uuid_to_id={}
    for responses in data:
        for response in responses['camera_responses']:
            # print(response['camera_used'])
            if (response['camera_used']==camera_num):
                # print(len(response['annotations']))
                for gt in response['annotations']:
                    if(gt['cuboid_uuid']) not in uuid_to_id:
                        uuid_to_id[gt['cuboid_uuid']]=id_counter
                        id_counter=id_counter+1
                    c = 0 if "Pedestrian" in gt['label'] else 7 if 'Truck' in gt['label'] else 5 if 'Buses' in gt['label'] else 2 if 'Small' in gt['label'] else 1 if 'Unpowered' in gt['label'] else 3
                    id=uuid_to_id[gt['cuboid_uuid']]
                    x1=gt['left']
                    x2=x1+gt['width']
                    y1=gt['top']
                    y2=y1+gt['height']
                    # detections.append([start+int(skip*i), c, 1.0,x1,y1,x2,y2])
                    detections.append([start+int(skip*i), id, x1,y1,x2,y2,c])
                    # mot.append([start+int(skip*i), id, x1,y1, gt['width'], gt['height'], 1, c, 1])
                i=i+1
    detections= np.asarray(detections)
    df=pd.DataFrame(detections,columns=['fn','id','x1','y1','x2','y2','class'])
    df=df.sort_values('fn').reset_index(drop=True)
    # print(df)
    # print(args.TrackingPth)
    df.to_csv(args.TrackingPth, header=None, index=None, sep=',')
    
    # we are assuming that video_name.GTHW7.GTHW7.txt is already in the results/tracking/
    # TODO actually implement it
    match_classes(args)

def df(args):
    data = {}
    tracks_path = args.TrackingPth
    tracks = np.loadtxt(tracks_path, delimiter=',')
    data["fn"]    = tracks[:, 0]
    data["id"]    = tracks[:, 1]
    data["x1"]    = tracks[:, 2]
    data["y1"]    = tracks[:, 3]
    data["x2"]    = tracks[:, 4]
    data["y2"]    = tracks[:, 5]
    data["class"] = tracks[:, 6]
    return pd.DataFrame.from_dict(data)

def df_txt(df, out_path):
    with open(out_path,'w') as out_file:
        for i, row in df.iterrows():
            fn, idd, x1, y1, x2, y2, clss = row['fn'], row['id'], row['x1'], row['y1'], row['x2'], row['y2'], row["class"]
            print('%d,%d,%.4f,%.4f,%.4f,%.4f,%d'%(fn, idd, x1, y1, x2, y2, clss),file=out_file)
            # fn, idd, x1, y1, x2, y2 = row['fn'], row['id'], row['x1'], row['y1'], row['x2'], row['y2']
            # print('%d,%d,%.4f,%.4f,%.4f,%.4f'%(fn, idd, x1, y1, x2, y2),file=out_file)
    # df = pd.read_pickle(args.TrackingPkl)
    # out_path = args.TrackingPth 

def match_classes(args):
    '''
    after running tracking this function will add class labels if necessary
    it directly works on txt file and modifies it
    '''
    # function for iou
    def compute_pairwise_iou(df1, df2):
        boxes_set1 = df1[["x1", "y1", "x2", "y2"]].to_numpy()[np.newaxis, :, :]
        boxes_set2 = df2[["x1", "y1", "x2", "y2"]].to_numpy()[np.newaxis, :, :]
        
        x1 = np.maximum(boxes_set1[:, :, 0][:, :, np.newaxis], boxes_set2[:, :, 0])
        y1 = np.maximum(boxes_set1[:, :, 1][:, :, np.newaxis], boxes_set2[:, :, 1])
        x2 = np.minimum(boxes_set1[:, :, 2][:, :, np.newaxis], boxes_set2[:, :, 2])
        y2 = np.minimum(boxes_set1[:, :, 3][:, :, np.newaxis], boxes_set2[:, :, 3])
        
        intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_bbox1 = (boxes_set1[:, :, 2] - boxes_set1[:, :, 0]) * (boxes_set1[:, :, 3] - boxes_set1[:, :, 1])
        area_bbox2 = (boxes_set2[:, :, 2] - boxes_set2[:, :, 0]) * (boxes_set2[:, :, 3] - boxes_set2[:, :, 1])
        union_area = area_bbox1[:, :, np.newaxis] + area_bbox2 - intersection_area
        
        iou_scores = intersection_area / union_area
        
        return iou_scores[0]

    # make a df from txt file
    data = {}
    tracks_path = args.TrackingPth
    tracks = np.loadtxt(tracks_path, delimiter=',')
    data["fn"] = tracks[:, 0]
    data["id"] = tracks[:, 1]
    data["x1"] = tracks[:, 2]
    data["y1"] = tracks[:, 3]
    data["x2"] = tracks[:, 4]
    data["y2"] = tracks[:, 5]

    track_df = pd.DataFrame.from_dict(data)
    det_df = pd.read_pickle(args.DetectionPkl)
    class_labels = []

    frames = np.unique(track_df["fn"])

    for fn in tqdm(frames):
        scores = compute_pairwise_iou(track_df[track_df["fn"] == fn], det_df[det_df["fn"] == fn])
        costs = 1 - scores
        row_indices, col_indices = scipy.optimize.linear_sum_assignment(costs)
        temp = np.array([-1 for i in range(len(track_df[track_df["fn"] == fn]))])
        temp[row_indices] = det_df.iloc[col_indices]["class"]
        class_labels += [int(c) for c in temp]
    
    # add class as a column to df
    track_df["class"] = class_labels
    # write class to txt file
    df_txt(track_df, args.TrackingPth)