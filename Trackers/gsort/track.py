from .gsort.gsort import *
# from .sort.sort import *
from Libs import *
from Maps import meter_per_pixel


def track(args, detectors):
    # get arguments form args
    if args.Detector is None:
        return FailLog("To interpret detections you should specify detector")
    # parse detection df using detector module
    detection_df = detectors[args.Detector].df(args)
    output_file = args.TrackingPth
    M = np.load(args.HomographyNPY, allow_pickle=True)[0]
    R = meter_per_pixel(args.MetaData['center'])
    mot_tracker = GSort(Homography_M=M, R_meter=R) #create instance of the SORT tracker
    # mot_tracker = Sort()
    with open(output_file,'w') as out_file:
        for frame_num in tqdm(range(int(detection_df.fn.min()), int(detection_df.fn.max())+1)): # this line might not work :))) 
            frame_df = detection_df[detection_df.fn == frame_num]
            # create dets --> this is the part when information is converted/grouped
            dets = frame_df[["x1", "y1", "x2", "y2", "score"]].to_numpy()
            # print("allllllllll dets")
            # print(dets)
            trackers = mot_tracker.update(dets)
            for d in trackers:
                print('%d,%d,%.4f,%.4f,%.4f,%.4f'%(frame_num,d[4],d[0],d[1],d[2],d[3]),file=out_file)

def df(args):
    data = {}
    tracks_path = args.TrackingPth
    tracks = np.loadtxt(tracks_path, delimiter=',')
    data["fn"] = tracks[:, 0]
    data["id"] = tracks[:, 1]
    data["x1"] = tracks[:, 2]
    data["y1"] = tracks[:, 3]
    data["x2"] = tracks[:, 4]
    data["y2"] = tracks[:, 5]
    return pd.DataFrame.from_dict(data)

def df_txt(df, out_path):
    with open(out_path,'w') as out_file:
        for i, row in df.iterrows():
            fn, idd, x1, y1, x2, y2 = row['fn'], row['id'], row['x1'], row['y1'], row['x2'], row['y2']
            print('%d,%d,%.4f,%.4f,%.4f,%.4f'%(fn, idd, x1, y1, x2, y2),file=out_file)
    # df = pd.read_pickle(args.TrackingPkl)
    # out_path = args.TrackingPth 