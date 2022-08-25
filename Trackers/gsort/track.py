from .gsort.gsort import *
from Libs import *


def track(args, detectors):
    # get arguments form args
    if args.Detector is None:
        return FailLog("To interpret detections you should specify detector")
    # parse detection df using detector module
    detection_df = detectors[args.Detector].df(args)
    output_file = args.TrackingPth
    mot_tracker = GSort() #create instance of the SORT tracker
    with open(output_file,'w') as out_file:
        for frame_num in tqdm(range(int(detection_df.fn.max()))): # this line might not work :))) 
            frame_df = detection_df[detection_df.fn == frame_num]
            # create dets --> this is the part when information is converted/grouped
            dets = frame_df[["x1", "y1", "x2", "y2", "score"]].to_numpy()
            trackers = mot_tracker.update(dets)
            for d in trackers:
                print('%d,%d,%.4f,%.4f,%.4f,%.4f'%(frame_num+1,d[4],d[0],d[1],d[2],d[3]),file=out_file)

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