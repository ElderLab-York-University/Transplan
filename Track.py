# Author: Sajjad P. Savaoji May 4 2022
# This py file will handle all the trackings
from Libs import *
from Utils import *
from Detect import *
from counting.resample_gt_MOI.resample_typical_tracks import track_resample
import Homography
import Maps
import TrackLabeling

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
# import Trackers.BOTSort.track
import Trackers.StrongSort.track
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
# trackers["BOTSort"] = Trackers.BOTSort.track
trackers["StrongSort"] = Trackers.StrongSort.track
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
    # current_tracker = trackers[args.Tracker]
    # df = current_tracker.df(args)
    df = pd.read_pickle(args.TrackingPkl)
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

    if not args.ForNFrames is None:
        frames = args.ForNFrames
    # Read until video is completed
    for frame_num in tqdm(range(frames)):
        if (not cap.isOpened()):
            break
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            this_frame_tracks = df[df.fn==(frame_num)]
            for i, track in this_frame_tracks.iterrows():
                # plot the bbox + id with colors
                bbid, x1 , y1, x2, y2 = track.id, int(track.x1), int(track.y1), int(track.x2), int(track.y2)
                # print(x1, y1, x2, y2)
                np.random.seed(int(bbid))
                color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'id:', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(frame, f'{int(bbid)}', (x1 + 60, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
                cv2.putText(frame, f'{int(bbid)}', (x1 + 60, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (144, 251, 144), 2)
            out_cap.write(frame)

def vistracktop(args):
    df = pd.read_pickle(args.ReprojectedPkl)
    annotated_video_path = args.VisTrackingTopPth
    video_path = args.Video
    cap = cv2.VideoCapture(video_path)
    # Check if camera opened successfully
    if (cap.isOpened()== False): return FailLog("could not open input video")

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame = cv.imread(args.HomographyTopView)
    rows2, cols2, dim2 = frame.shape
    frame_width , frame_height  = cols2 , rows2


    color = (0, 0, 102)

    out_cap = cv2.VideoWriter(annotated_video_path,cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width,frame_height))

    if not args.ForNFrames is None:
        frames = min(args.ForNFrames, frames)
    # Read until video is completed
    for frame_num in tqdm(range(frames)):
        frame = cv.imread(args.HomographyTopView)

        this_frame_tracks = df[df.fn==(frame_num)]
        for i, track in this_frame_tracks.iterrows():
            # plot the bbox + id with colors
            bbid, x , y = track.id, int(track.x), int(track.y)
            # print(x1, y1, x2, y2)
            np.random.seed(int(bbid))
            color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            cv.circle(frame, (x,y), radius=2, color=color, thickness=3)
            cv2.putText(frame, f'{int(bbid)}', (x + 10, y-2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame, f'{int(bbid)}', (x + 10, y-2), cv2.FONT_HERSHEY_SIMPLEX, 1, (144, 251, 144), 1)
        out_cap.write(frame)
    return SucLog("executed vistrack top")

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
        print(ProcLog("updating txt, pkl, reprojected, and meter files for tracking"))
        trackers[args.Tracker].df_txt(df, args.TrackingPth)
        store_df_pickle(args)
        Homography.reproject(args, from_back_up=False)
        Maps.pix2meter(args)

    # restore original tracks in txt and pkl
    print(ProcLog("recover tracking from backup"))
    df = pd.read_pickle(args.TrackingPklBackUp)
    update_tracking_changes(df, args)

    # apply postprocessing on args.ReprojectedPkLMeter and ReprojectedPkl
    if not args.TrackTh is None:
        df  = remove_short_tracks(args)
        update_tracking_changes(df, args)

    # apply postprocessing on args.TrackingPkl
    if args.MaskROI:
        print("mask ROI")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = remove_out_of_ROI(args)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)

    if args.RemoveInvalidTracks:
        print("removing invalid tracks")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = remove_invalid_tracks(args)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)
        
    if args.SelectDifEdgeInROI:
        print("remove tracks that begin and end in the same roi region")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = select_based_on_roi(args, different_roi_edge)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)

    if args.SelectEndingInROI:
        print("select ending in ROI")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = select_based_on_roi(args, end_in_roi)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)

    if args.SelectBeginInROI:
        print("select begin in ROI")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = select_based_on_roi(args, begin_in_roi)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)

    if args.HasPointsInROI:
        print("select tracks that have points inside ROI")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = select_based_on_roi(args, has_points_in_roi)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)

    if args.CrossROI:
        print("select tracks that cross roi at least once")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = select_based_on_roi(args, cross_roi, resample_tracks=True)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)

    if args.CrossROIMulti:
        print("select tracks that cross roi multiple edges")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = select_based_on_roi(args, cross_roi_multiple, resample_tracks=True)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)


    return SucLog("track post processing executed with no error")

# def roi_mask_tracks(args):
#     df_meter_ungrouped = pd.read_pickle(args.ReprojectedPklMeter)
#     df_reg_ungrouped   = pd.read_pickle(args.ReprojectedPkl)
#     df_meter = group_tracks_by_id(df_meter_ungrouped)
#     df_reg   = group_tracks_by_id(df_reg_ungrouped)
#     main_df = pd.read_pickle(args.TrackingPkl)

#     mask = []
#     for i , row in main_df.iterrows():

def end_in_roi(pg, traj, th, poly_path):
    d_end, i_end = pg.distance(traj[-1])
    return d_end <= th

def begin_in_roi(pg, traj, th, poly_path):
    d_str, i_str = pg.distance(traj[0])
    return d_str <= th

def has_points_in_roi(pg, traj, th, poly_path):
    for p in traj:
        if poly_path.contains_point(p):
            return True
    return False

def different_roi_edge(pg, traj, th, poly_path):
    d_end, i_end = pg.distance(traj[-1])
    d_str, i_str = pg.distance(traj[0])
    if d_end<=th and d_str<=th:
        return not i_str == i_end
    return True

def cross_roi(pg, traj, *args, **kwargs):
    int_indxes= pg.doIntersect(traj)
    if int_indxes:
        return True
    return False

def cross_roi_multiple(pg, traj, *args, **kwargs):
    int_indxes = pg.doIntersect(traj)
    if len(np.uniwue(int_indxes)) > 1:
        return True
    return False

def select_based_on_roi(args, condition, resample_tracks=False):
    df = pd.read_pickle(args.TrackingPkl)

    tracks_path = args.ReprojectedPkl
    tracks_meter_path = args.ReprojectedPklMeter
    meta_data = args.MetaData # dict is already loaded
    HomographyNPY = args.HomographyNPY
    M = np.load(HomographyNPY, allow_pickle=True)[0]

    # load data
    tracks = group_tracks_by_id(pd.read_pickle(tracks_path))
    tracks_meter = group_tracks_by_id(pd.read_pickle(tracks_meter_path))
    tracks['index_mask'] = tracks_meter['trajectory'].apply(lambda x: track_resample(x, return_mask=True, threshold=args.ResampleTH))

    # create roi polygon
    roi_rep = []
    for p in args.MetaData["roi"]:
        point = np.array([p[0], p[1], 1])
        new_point = M.dot(point)
        new_point /= new_point[2]
        roi_rep.append([new_point[0], new_point[1]])

    pg = TrackLabeling.MyPoly(roi_rep, args.MetaData["roi_group"])
    th = args.MetaData["roi_percent"] * np.sqrt(pg.area)
    poly_path = mplPath.Path(np.array(roi_rep))

    ids_to_keep = []
    for i, row in tqdm(tracks.iterrows(), total=len(tracks)):
        if resample_tracks:
            traj = row["trajectory"][row["index_mask"]]
        else:
            traj = row["trajectory"]

        if condition(pg, traj, th, poly_path):
            ids_to_keep.append(row["id"])

    mask = []
    for i, row in df.iterrows():
        if row["id"] in ids_to_keep:
            mask.append(True)
        else:
            mask.append(False)
    return df[mask]


def remove_invalid_tracks(args):
    df = pd.read_pickle(args.TrackingPkl)
    mask = []
    to_remove_ids = []
    ids = np.unique(df["id"])

    for id in ids:
        df_id = df[df["id"] == id]
        if len(df_id) <= 2:
            to_remove_ids.append(id)

    for i, row in df.iterrows():
        if row["id"] in to_remove_ids:
            mask.append(False)
        else:
            mask.append(True)

    return df[mask]

def remove_out_of_ROI(args):
    df_reproj = pd.read_pickle(args.ReprojectedPkl)
    df_main   = pd.read_pickle(args.TrackingPkl)
    mask = []

    HomographyNPY = args.HomographyNPY
    M = np.load(HomographyNPY, allow_pickle=True)[0]

    roi_rep = []
    for p in args.MetaData["roi"]:
        point = np.array([p[0], p[1], 1])
        new_point = M.dot(point)
        new_point /= new_point[2]
        roi_rep.append([new_point[0], new_point[1]])
    pg = TrackLabeling.MyPoly(roi_rep, args.MetaData["roi_group"])
    poly_path = mplPath.Path(np.array(roi_rep))

    for i, row in tqdm(df_reproj.iterrows(), total=len(df_reproj)):
        x, y = row.x, row.y
        p = [x, y]
        if poly_path.contains_point(p):
        # if pg.encloses_point(p):
            mask.append(True)
        else:
            mask.append(False)
    return df_main[mask]

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