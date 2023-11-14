from .sort.sort import *
from Libs import *
from Utils import *


def track(args, detectors):
    # get arguments form args
    if args.Detector is None:
        return FailLog("To interpret detections you should specify detector")
    # parse detection df using detector module
    # detection_df = detectors[args.Detector].df(args)
    detection_df = pd.read_pickle(args.DetectionPkl)

    output_file = args.TrackingPth
    # tracking hypter parameters
    max_age=19
    min_hits=1
    iou_threshold=0.5
    # finish tracking hyperparams
    mot_tracker = Sort(max_age, min_hits, iou_threshold) #create instance of the SORT tracker
    with open(output_file,'w') as out_file:
        for frame_num in tqdm(range(int(detection_df.fn.min()), int(detection_df.fn.max()+1))): #looping from df.fn.min to df.fn.max
            frame_df = detection_df[detection_df.fn == frame_num]
            # create dets --> this is the part when information is converted/grouped
            dets = frame_df[["x1", "y1", "x2", "y2", "score"]].to_numpy()
            trackers = mot_tracker.update(dets)
            for d in trackers:
                print('%d,%d,%.4f,%.4f,%.4f,%.4f'%(frame_num,d[4],d[0],d[1],d[2],d[3]),file=out_file) # using frame_num so that trakcing df and detection df are synced

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
    det_df = df = pd.read_pickle(args.DetectionPkl)
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