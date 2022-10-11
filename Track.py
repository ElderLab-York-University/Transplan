# Author: Sajjad P. Savaoji May 4 2022
# This py file will handle all the trackings
from Libs import *
from Utils import *
from Detect import *
from counting.resample_gt_MOI.resample_typical_tracks import track_resample

# import all detectros here
# And add their names to the "trackers" dictionary
# -------------------------- 
import Trackers.sort.track
import Trackers.CenterTrack.track
import Trackers.DeepSort.track
import Trackers.ByteTrack.track
import Trackers.gsort.track
import Trackers.OCSort.track
import Trackers.GByteTrack.track
import Trackers.GDeepSort.track
# --------------------------
trackers = {}
trackers["sort"] = Trackers.sort.track
trackers["CenterTrack"] = Trackers.CenterTrack.track
trackers["DeepSort"] = Trackers.DeepSort.track
trackers["ByteTrack"] = Trackers.ByteTrack.track
trackers["gsort"] = Trackers.gsort.track
trackers["OCSort"] = Trackers.OCSort.track
trackers["GByteTrack"] = Trackers.GByteTrack.track
trackers["GDeepSort"] = Trackers.GDeepSort.track
# --------------------------

def track(args):
    if args.Tracker not in os.listdir("./Trackers/"):
        return FailLog("Tracker not recognized in ./Trackers/")

    current_tracker = trackers[args.Tracker]
    current_tracker.track(args, detectors)
    # store pkl version of tracked df
    store_df_pickle(args)
    store_df_pickle_backup(args)
    return SucLog("Tracking files stored")

def store_df_pickle(args):
    # should be called after tracking is done and the results are stored in the .txt file
    df = trackers[args.Tracker].df(args)
    df.to_pickle(args.TrackingPkl)

def store_df_pickle_backup(args):
    # should be called after tracking is done and the results are stored in the .txt file
    df = trackers[args.Tracker].df(args)
    df.to_pickle(args.TrackingPklBackUp)

def vistrack(args):
    current_tracker = trackers[args.Tracker]
    df = current_tracker.df(args)
    video_path = args.Video
    annotated_video_path = args.VisTrackingPth
    # tracks_path = args.TrackingPth

    color = (0, 0, 102)
    cap = cv2.VideoCapture(video_path)
    # Check if camera opened successfully
    if (cap.isOpened()== False): return FailLog("could not open input video")

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    out_cap = cv2.VideoWriter(annotated_video_path,cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width,frame_height))

    # Read until video is completed
    for frame_num in tqdm(range(frames)):
        if (not cap.isOpened()):
            break
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            this_frame_tracks = df[df.fn==(frame_num+1)]
            for i, track in this_frame_tracks.iterrows():
                # plot the bbox + id with colors
                bbid, x1 , y1, x2, y2 = track.id, int(track.x1), int(track.y1), int(track.x2), int(track.y2)
                # print(x1, y1, x2, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'id:', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(frame, f'{int(bbid)}', (x1 + 60, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
                cv2.putText(frame, f'{int(bbid)}', (x1 + 60, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (144, 251, 144), 2)
            out_cap.write(frame)

def vistrackmoi(args):
    current_tracker = trackers[args.Tracker]
    df = current_tracker.df(args)
    video_path = args.Video
    annotated_video_path = args.VisTrackingMoIPth
    df_matching = pd.read_csv(args.CountingIdMatchPth)

    dict_matching = {}
    for i, row in df_matching.iterrows():
        dict_matching[int(row['id'])] = int(row['moi'])
        
    cap = cv2.VideoCapture(video_path)
    # Check if camera opened successfully
    if (cap.isOpened()== False): return FailLog("could not open input video")

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    out_cap = cv2.VideoWriter(annotated_video_path,cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width,frame_height))

    # Read until video is completed
    for frame_num in tqdm(range(frames)):
        if (not cap.isOpened()):
            break
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            this_frame_tracks = df[df.fn==(frame_num+1)]
            for i, track in this_frame_tracks.iterrows():
                # plot the bbox + id with colors
                bbid, x1 , y1, x2, y2 = int(track.id), int(track.x1), int(track.y1), int(track.x2), int(track.y2)
                if bbid in dict_matching:
                    color = moi_color_dict[dict_matching[bbid]]
                else: color = (0, 0, 125)
                # print(x1, y1, x2, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'id:{bbid}', (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                if bbid in dict_matching:
                    cv2.putText(frame, f'moi:{dict_matching[bbid]}', (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            out_cap.write(frame)

    # When everything done, release the video capture object
    cap.release()
    out_cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    return SucLog("Vis Tracking moi file stored")

def trackpostproc(args):
    def update_tracking_changes(df, args):
        trackers[args.Tracker].df_txt(df, args.TrackingPth)
        store_df_pickle(args)

    # apply postprocessing on args.ReprojectedPkLMeter and ReprojectedPkl
    if not args.TrackTh is None:
        df  = remove_short_tracks(args)
        update_tracking_changes(df, args)

    
    return SucLog("track post processing executed with no error")


def remove_short_tracks(args):
    th = args.TrackTh
    df_meter_ungrouped = pd.read_pickle(args.ReprojectedPklMeter)
    df_reg_ungrouped   = pd.read_pickle(args.ReprojectedPkl)
    df_meter = group_tracks_by_id(df_meter_ungrouped)
    df_reg   = group_tracks_by_id(df_reg_ungrouped)

    main_df = pd.read_pickle(args.TrackingPkl)

    to_remove_ids = []
    # resample tracks
    df_meter['trajectory'] = df_meter['trajectory'].apply(lambda x: track_resample(x))
    # df_reg['trajectory'] = df_reg['trajectory'].apply(lambda x: track_resample(x))
    for i, row in df_meter.iterrows():
        if arc_length(row['trajectory']) < th:
            to_remove_ids.append(row['id'])

    mask = []
    for i, row in main_df.iterrows():
        if row['id'] in to_remove_ids:
            mask.append(False)
        else: mask.append(True)

    return main_df[mask]

def group_tracks_by_id(df):
    # this function was writtern for grouping the tracks with the same id
    # usinig this one can load the data from a .txt file rather than .mat file
    all_ids = np.unique(df['id'].to_numpy(dtype=np.int64))
    data = {"id":[], "trajectory":[], "frames":[]}
    for idd in tqdm(all_ids):
        frames = df[df['id']==idd]["fn"].to_numpy(np.float32)
        id = idd
        trajectory = df[df['id']==idd][["x", "y"]].to_numpy(np.float32)
        
        data["id"].append(id)
        data["frames"].append(frames)
        data["trajectory"].append(trajectory)
    df2 = pd.DataFrame(data)
    return df2

def arc_length(track):
        """
        :param track: input track numpy array (M, 2)
        :return: the estimated arc length of the track
        """
        assert track.shape[1] == 2
        accum_dist = 0
        for i in range(1, track.shape[0]):
            dist_ = np.sqrt(np.sum((track[i] - track[i - 1]) ** 2))
            accum_dist += dist_
        return accum_dist