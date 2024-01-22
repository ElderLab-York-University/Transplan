# Author: Sajjad P. Savaoji May 4 2022
# This py file will handle all the trackings
from Libs import *
from Utils import *
from Detect import *
from counting.resample_gt_MOI.resample_typical_tracks import track_resample
import Homography
import Maps
import TrackLabeling
import copy
from counting.counting import group_tracks_by_id

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
import Trackers.GTHW7.track
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
trackers["GTHW7"] = Trackers.GTHW7.track
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
    df.to_pickle(args.TrackingPkl, protocol=4)

def store_df_pickle_backup(args):
    # should be called after tracking is done and the results are stored in the .txt file
    df = trackers[args.Tracker].df(args)
    df.to_pickle(args.TrackingPklBackUp, protocol=4)

def vistrack(args):
    # current_tracker = trackers[args.Tracker]
    # df = current_tracker.df(args)
    df = pd.read_pickle(args.TrackingPkl)
    video_path = args.Video
    annotated_video_path = args.VisTrackingPth
    # tracks_path = args.TrackingPth
    if(args.CalcDistance):
        df, p1s=calculate_distance(args)
    # if(args.CalcSpeed):
    #     calculate_speeds(args)
    color = (0, 0, 102)
    cap = cv2.VideoCapture(video_path)
    # Check if camera opened successfully
    if (cap.isOpened()== False): return FailLog("could not open input video")

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    out_cap = cv2.VideoWriter(annotated_video_path,cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width,frame_height))
    out_cap2 = cv2.VideoWriter("./video.mp4",cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width,frame_height))
    
    print(args.VisTrackingPth)
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
                if(args.CalcDistance):
                    distance=track.distance
                    p1=p1s[i]
                    cv2.putText(frame, f'distance:', (x1, y1-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)   
                    cv2.putText(frame, f'{int(distance)}', (x1 + 140, y1-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (144, 251, 144), 2)
                    roi = args.MetaData["roi"]
                    for i in range(0, len(roi)):
                        p1 = tuple(roi[i-1])
                        p2 = tuple(roi[i])
                        cv2.line(frame, p1, p2, (128, 128, 0), 25)
                    
                    # cv.circle(frame, (int(p1.x),int(p1.y)), radius=2, color=color, thickness=3)
        # alpha = 0.6
        # M = np.load(args.HomographyNPY, allow_pickle=True)[0]
        # roi_rep = []
        # roi = args.MetaData["roi"]
        # roi_group = args.MetaData["roi_group"]
        # for p in roi:
        #     point = np.array([p[0], p[1], 1])
        #     new_point = M.dot(point)
        #     new_point /= new_point[2]
        #     roi_rep.append([int(new_point[0]), int(new_point[1])])
        # img1=frame.copy()
        # rows1, cols1, dim1 = frame.shape
        # poly_path = mplPath.Path(np.array(roi_rep))
        # for i in range(rows1):
        #     for j in range(cols1):
        #         if not poly_path.contains_point([j, i]):
        #             img1[i][j] = [0, 0, 0]    
        # frame = cv.addWeighted(img1, alpha, frame, 1 - alpha, 0)
        
        out_cap.write(frame)
        out_cap2.write(frame)
    print(df)
    cap.release()
    out_cap.release()
    out_cap2.release()
    return SucLog("track vis successful")
def meter_per_pixel(center, zoom=19):
    m_per_p = 156543.03392 * np.cos(center[0] * np.pi / 180) / np.power(2, zoom)
    return m_per_p

def calculate_speeds(args):
    
    print("Calculating Speeds!")
    df = pd.read_pickle(args.ReprojectedPklMeter)  
    track_df=pd.read_pickle(args.TrackingPkl)
    ids= np.unique(df['id'])
    speeds=np.zeros(len(df))
    for track_id in ids:
        bbox_idx=np.where(df['id']==track_id)[0]
        bboxes=df.iloc[bbox_idx]
        frames=sorted(bboxes['fn'])
        for i in range(len(frames)):
            if(i<len(frames)-1):
                current_frame=bboxes[bboxes['fn']==frames[i]]
                next_frame=bboxes[bboxes['fn']==frames[i+1]]
                current_position=np.array([current_frame.x, current_frame.y])
                next_position=np.array([next_frame.x, next_frame.y])
                displacement= np.abs(next_position-current_position)
                input()
            else:
                speeds[bbox_idx[i]]=-1
    print(ids)
    input()
    

def calculate_distance(args):
    df = pd.read_pickle(args.ReprojectedPklMeter)  
    track_df=pd.read_pickle(args.TrackingPkl)
    video_path = args.Video
    cap = cv2.VideoCapture(video_path)
    # Check if camera opened successfully
    if (cap.isOpened()== False): return FailLog("could not open input video")

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    M = np.load(args.HomographyNPY, allow_pickle=True)[0]
    roi_rep = []
    roi = args.MetaData["roi"]
    r = meter_per_pixel(args.MetaData['center'])        
    for p in roi:
        point = np.array([p[0], p[1], 1])
        new_point = M.dot(point)
        new_point /= new_point[2]
        roi_rep.append((int(r*new_point[0]), int(r*new_point[1])))
    poly_path = mplPath.Path(np.array(roi_rep))
    poly=Polygon(roi_rep)
    distances=[]
    p1s=[]
    for frame_num in tqdm(range(frames)):
        this_frame_tracks = df[df.fn==(frame_num)]
        for i, track in this_frame_tracks.iterrows():
            # plot the bbox + id with colors
            # bbid, x , y = track.id, int(track.x), int(track.y)
            pcp=[track.x, track.y]
            if not poly_path.contains_point(pcp):
                pcp=SPoint(track.x, track.y)            
                p1, p2 = nearest_points(poly, pcp)
                distances.append(pcp.distance(p1))
                p1s.append(p1)
            else:
                distances.append(0)
                p1s.append(SPoint(0,0))
                
    track_df['distance']=distances
    return track_df, p1s

def vistracktop(args):
    df = pd.read_pickle(args.ReprojectedPkl)
    annotated_video_path = args.VisTrackingTopPth
    video_path = args.Video
    cap = cv2.VideoCapture(video_path)
    # Check if camera opened successfully
    if (cap.isOpened()== False): return FailLog("could not open input video")

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    img1 = cv.imread(args.HomographyStreetView)
    frame = cv.imread(args.HomographyTopView)
    rows2, cols2, dim2 = frame.shape

    alpha=0.6
    M = np.load(args.HomographyNPY, allow_pickle=True)[0]
    frame_width , frame_height  = cols2 , rows2
    img12 = cv.warpPerspective(img1, M, (cols2, rows2))
    img2 = cv.addWeighted(frame, alpha, img12, 1 - alpha, 0)


    color = (0, 0, 102)

    out_cap = cv2.VideoWriter(annotated_video_path,cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width,frame_height))

    if not args.ForNFrames is None:
        frames = min(args.ForNFrames, frames)
    # Read until video is completed
    for frame_num in tqdm(range(frames)):
        frame = copy.deepcopy(img2)

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
    if args.ForNFrames is not None:
        frames = min(frames, args.ForNFrames)
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
        Homography.reproject(args, method=args.BackprojectionMethod,
                            source = args.BackprojectSource, from_back_up=False)
        
        Maps.pix2meter(args)

    # restore original tracks in txt and pkl
    print(ProcLog("recover tracking from backup"))
    df = pd.read_pickle(args.TrackingPklBackUp)
    update_tracking_changes(df, args)

    # apply postprocessing on args.ReprojectedPkLMeter and ReprojectedPkl
    if not args.TrackTh is None:
        df  = remove_short_tracks(args)
        update_tracking_changes(df, args)

    print(f"starting with {len(np.unique(df['class']))} tracks")
    print(np.unique(df['class']))

    # unify classes in each track
    if args.UnifyTrackClass:
        print("unify classes")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = unify_classes_in_tracks(df, args)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)  

    # filter tracks based on classes
    if args.classes_to_keep:
        print("filter classes")
        print(f"starting with {len(np.unique(df['class']))} classes")
        df = filter_det_class(df, args)
        print(f"ending with {len(np.unique(df['class']))} classes")
        update_tracking_changes(df, args)   

    # apply postprocessing on args.TrackingPkl
    if args.MaskROI:
        print("mask ROI")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = remove_out_of_ROI(args)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)

    if args.MaskGPFrame:
        print("mask GP Frame")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = remove_out_of_GP_frame(args)
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

    # if args.SelectEndingInROI:
    #     print("select ending in ROI")
    #     print(f"starting with {len(np.unique(df['id']))} tracks")
    #     df = select_based_on_roi(args, end_in_roi)
    #     print(f"ending with {len(np.unique(df['id']))} tracks")
    #     update_tracking_changes(df, args)

    # if args.SelectBeginInROI:
    #     print("select begin in ROI")
    #     print(f"starting with {len(np.unique(df['id']))} tracks")
    #     df = select_based_on_roi(args, begin_in_roi)
    #     print(f"ending with {len(np.unique(df['id']))} tracks")
    #     update_tracking_changes(df, args)

    if args.HasPointsInROI:
        print("select tracks that have points inside ROI")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = select_based_on_roi(args, has_points_in_roi)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)

    if args.MovesInROI:
        print("select tracks that move at least args.resampleTH in ROI")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = select_based_on_roi(args, moves_in_roi)
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

    if args.JustEnterROI:
        print("select tracks that just enter roi")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = select_based_on_roi(args, just_enter_roi, resample_tracks=True)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)

    if args.JustExitROI:
        print("select tracks that just exit roi")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = select_based_on_roi(args, just_exit_roi, resample_tracks=True)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)

    if args.WithinROI:
        print("select tracks that are completely within roi")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = select_based_on_roi(args, within_roi, resample_tracks=True)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)

    if args.ExitOrCrossROI:
        print("select tracks that either exit roi or cross roi multi")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = select_based_on_roi(args, just_exit_or_cross_multi, resample_tracks=True)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)
    
    if args.SelectToBeCounted:
        print("select tracks to be counted")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = select_based_on_roi(args, to_be_counted, resample_tracks=True)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)

    if args.Interpolate:
        print("Interpolate Tracks")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = interpolate_tracks(args)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)

    return SucLog("track post processing executed with no error")


def unify_classes_in_tracks(df, args):
    uids = np.unique(df.id)
    for ui in uids:
        ui_classes = df.loc[df.id == ui, "class"].to_numpy()
        ui_classes = ui_classes[ui_classes >= 0]
        class_major = scipy.stats.mode(ui_classes, keepdims=False)
        class_major = int(class_major[0])
        df.loc[df.id == ui, "class"] = class_major
    return df

# TODO seems like these functions were used for ROI with thresholding
# not the new crossing method

def end_in_roi(pg, traj, th, poly_path, *args, **kwargs):
    p = traj[-1]
    if poly_path.contains_point(p):
            return True
    return False

def begin_in_roi(pg, traj, th, poly_path, *args, **kwargs):
    p = traj[0]
    if poly_path.contains_point(p):
            return True
    return False

def moves_in_roi(pg, traj, th, poly_path, *args, **kwargs):
    in_roi_traj = TrackLabeling.get_in_roi_points(traj, poly_path, return_mask=False)
    if len(in_roi_traj) > 1:
        return True
    return False

def has_points_in_roi(pg, traj, th, poly_path, *args, **kwargs):
    for p in traj:
        if poly_path.contains_point(p):
            return True
    return False

def different_roi_edge(pg, traj, th, poly_path, *args, **kwargs):
    d_end, i_end = pg.distance(traj[-1])
    d_str, i_str = pg.distance(traj[0])
    if d_end<=th and d_str<=th:
        return not i_str == i_end
    return True

def cross_roi(pg, traj, *args, **kwargs):
    int_indxes = pg.doIntersect(traj, ret_points=False)
    if int_indxes:
        return True
    return False

def just_enter_roi(pg, traj, th, poly_path, *args, **kwargs):
    int_indxes = pg.doIntersect(traj, ret_points=False)
    cross_once = len(int_indxes)==1
    start_in_roi = poly_path.contains_point(traj[0])
    end_in_roi = poly_path.contains_point(traj[-1])
    if cross_once and (not start_in_roi) and end_in_roi:
        return True
    else:
        return False
    
def just_exit_roi(pg, traj, th, poly_path, *args, **kwargs):
    int_indxes = pg.doIntersect(traj, ret_points=False)
    cross_once = len(int_indxes)==1
    end_in_roi = poly_path.contains_point(traj[-1])
    start_in_roi = poly_path.contains_point(traj[0])
    if cross_once and (not end_in_roi) and start_in_roi:
        return True
    else:
        return False
    
def within_roi(pg, traj, th, poly_path, *args, **kwargs):
    int_indxes = pg.doIntersect(traj, ret_points=False)
    dont_cross_roi = len(int_indxes)==0
    start_in_roi = poly_path.contains_point(traj[0])
    end_in_roi = poly_path.contains_point(traj[-1])
    if dont_cross_roi and start_in_roi and end_in_roi:
        return True
    else:
        return False
    
def cross_roi_multiple(pg, traj, *args, **kwargs):
    int_indxes = pg.doIntersect(traj, ret_points=False)
    if len(np.unique(int_indxes)) > 1:
        return True
    return False

def just_exit_or_cross_multi(pg, traj, th, poly_path, *args, **kwargs):
    if cross_roi_multiple(pg, traj, th, poly_path) or just_exit_roi(pg, traj, th, poly_path):
        return True
    else:
        return False
    
def unfinished_track(pg, traj, th, poly_path, *args, **kwargs):
    last_recorded_frame = kwargs["frames"][-1]
    last_video_frame    = kwargs["last_frame"]
    if last_recorded_frame >= last_video_frame - kwargs["UnfinishedTrackFrameTh"]:
        return True
    else:
        return False
    
def unstarted_track(pg, traj, th, poly_path, *args, **kwargs):
    first_recorded_frame  = kwargs["frames"][0]
    first_video_frame    = 0
    if first_recorded_frame <= first_video_frame + kwargs["UnfinishedTrackFrameTh"]:
        return True
    else:
        return False
    
def to_be_counted(pg, traj, th, poly_path, *args, **kwargs):
    if cross_roi_multiple(pg, traj, th, poly_path, *args, **kwargs) or\
       just_enter_roi(pg, traj, th, poly_path, *args, **kwargs):
        return True
    else:
        return False

def interpolate_tracks(args):
    df = pd.read_pickle(args.TrackingPkl)
    columns_to_inter_polate = ["x1", "y1", "x2", "y2", "fn"]
    regular_columns = []
    for col in df.columns:
        if not col in columns_to_inter_polate:
            regular_columns.append(col)

    print(regular_columns)
    print(columns_to_inter_polate)

    intpol_df_data = {}
    for col in df.columns:
        intpol_df_data[col] = []

    unique_ids = np.unique(df["id"].to_numpy())
    for uid in tqdm(unique_ids, desc="IntPolTracks"):
        df_id = df[df["id"] == uid].sort_values(by=["fn"])
        for i in range(1, len(df_id)):
            cur_row = df_id.iloc[i]
            pre_row = df_id.iloc[i-1]
            if cur_row["fn"] == pre_row["fn"]:
                print("two detections with same id in tracking results")
                print(cur_row["fn"])
            assert cur_row["fn"] > pre_row["fn"]
            frame_diff = int(cur_row["fn"] - pre_row["fn"])
            if frame_diff > 1 and frame_diff <= args.InterpolateTh:
                weights = np.linspace(0, 1, frame_diff, endpoint=False)
                for w in weights:
                    for col in regular_columns:
                        intpol_df_data[col].append(pre_row[col])
                    for col in columns_to_inter_polate:
                        pre_value = pre_row[col]
                        cur_value = cur_row[col]
                        intpol_df_data[col].append((1-w)*float(pre_value) + (w)*float(cur_value))
            else:
                for col in df_id.columns:
                    intpol_df_data[col].append(pre_row[col])

        # add the last row as it is   
        for col in df_id.columns:
            intpol_df_data[col].append(df_id.iloc[-1][col])

        interpol_df = pd.DataFrame.from_dict(intpol_df_data).sort_values(by=["fn"])

    print(f"original df len: {len(df)}")
    print(f"interpol df len: {len(interpol_df)}")

    return interpol_df
    
def select_based_on_roi(args, condition, resample_tracks=False):
    # get last frame number in the video
    cap = cv2.VideoCapture(args.Video)
    # Check if camera opened successfully
    if (cap.isOpened()== False): return FailLog("could not open input video")
    last_frame  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    first_frame = int(0)

    df = pd.read_pickle(args.TrackingPkl)

    tracks_path = args.ReprojectedPkl
    tracks_meter_path = args.ReprojectedPklMeter
    meta_data = args.MetaData # dict is already loaded
    HomographyNPY = args.HomographyNPY
    M = np.load(HomographyNPY, allow_pickle=True)[0]

    # load data
    tracks = group_tracks_by_id(pd.read_pickle(tracks_path), gp=True)
    tracks_meter = group_tracks_by_id(pd.read_pickle(tracks_meter_path), gp=True)
    tracks['index_mask'] = tracks_meter['trajectory'].apply(lambda x: track_resample(x, return_mask=True, threshold=args.ResampleTH))

    # create roi polygon on ground plane
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
            frames = row.frames[row.index_mask]
        else:
            traj = row["trajectory"]
            frames = row.frames

        if condition(pg, traj, th, poly_path,
                    frames=frames, first_frame=first_frame,
                    last_frame=last_frame, UnfinishedTrackFrameTh=args.UnfinishedTrackFrameTh):
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

def remove_out_of_GP_frame (args):
    df_reproj = pd.read_pickle(args.ReprojectedPkl)
    df_main   = pd.read_pickle(args.TrackingPkl)
    mask = []

    HomographyNPY = args.HomographyNPY
    M = np.load(HomographyNPY, allow_pickle=True)[0]

    # read top
    frame = cv.imread(args.HomographyTopView)
    rows2, cols2, dim2 = frame.shape
    frame_width , frame_height  = cols2 , rows2

    roi_rep = [[0, 0], [0, frame_height], [frame_width, frame_height], [frame_width, 0]]
    
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
    df_meter = group_tracks_by_id(df_meter_ungrouped, gp=True)
    df_reg   = group_tracks_by_id(df_reg_ungrouped, gp=True)

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