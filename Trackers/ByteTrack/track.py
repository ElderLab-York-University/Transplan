from Libs import *
from Utils import *

def track(args, *oargs):
    setup(args)
    env_name = args.Tracker
    exec_path = "./Trackers/ByteTrack/run.py"
    conda_pyrun(env_name, exec_path, args)
    if getattr(args, "BTCorrectClasses", False):
        match_classes_frame_local(args)
    else:
        match_classes(args)  # preserve legacy output for reproducibility

def df(args):
    data = {}
    tracks_path = args.TrackingPth
    tracks = np.loadtxt(tracks_path, delimiter=',')
    data["fn"]    = tracks[:, 0]
    data["id"]    = tracks[:, 1]
    data["score"] = tracks[:, 2]
    data["x1"]    = tracks[:, 3]
    data["y1"]    = tracks[:, 4]
    data["x2"]    = tracks[:, 5]
    data["y2"]    = tracks[:, 6]
    data["class"] = tracks[:, 7]
    return pd.DataFrame.from_dict(data)

def df_txt(df, out_path):
    with open(out_path,'w') as out_file:
        for i, row in df.iterrows():
            fn, idd,score, x1, y1, x2, y2, clss = row['fn'], row['id'],row['score'], row['x1'], row['y1'], row['x2'], row['y2'], row["class"]
            print('%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%d'%(fn, idd, score, x1, y1, x2, y2, clss),file=out_file)
    # df = pd.read_pickle(args.TrackingPkl)
    # out_path = args.TrackingPth 

def match_classes_frame_local(args):
    """Assign detection classes to tracks within each actual frame.

    The legacy matcher accidentally indexes the *full* detection table with
    indices from a *single-frame* assignment; this opt-in replacement fixes
    that for class-sensitive evaluation and downstream MoI analysis.
    """
    from scipy.optimize import linear_sum_assignment

    cols = ["fn", "id", "score", "x1", "y1", "x2", "y2"]
    try:
        tracks = pd.read_csv(args.TrackingPth, header=None, names=cols)
    except pd.errors.EmptyDataError:
        tracks = pd.DataFrame(columns=cols)
    if tracks.empty:
        tracks["class"] = pd.Series(dtype=int)
        df_txt(tracks, args.TrackingPth)
        return

    dets = pd.read_pickle(args.DetectionPkl)
    labels = np.full(len(tracks), -1, dtype=int)
    for fn, tframe in tracks.groupby("fn", sort=False):
        dframe = dets.loc[dets["fn"] == fn]
        if dframe.empty:
            continue
        a = tframe[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        b = dframe[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        x_left = np.maximum(a[:, None, 0], b[None, :, 0])
        y_top = np.maximum(a[:, None, 1], b[None, :, 1])
        x_right = np.minimum(a[:, None, 2], b[None, :, 2])
        y_bottom = np.minimum(a[:, None, 3], b[None, :, 3])
        intersection = np.maximum(x_right - x_left, 0) * np.maximum(y_bottom - y_top, 0)
        a_area = np.maximum(a[:, 2] - a[:, 0], 0) * np.maximum(a[:, 3] - a[:, 1], 0)
        b_area = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)
        union = a_area[:, None] + b_area[None, :] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        rows, columns = linear_sum_assignment(1 - iou)
        valid = iou[rows, columns] > 0
        labels[tframe.index.to_numpy()[rows[valid]]] = (
            dframe["class"].to_numpy()[columns[valid]].astype(int)
        )

    tracks["class"] = labels
    df_txt(tracks, args.TrackingPth)


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
    data["fn"]    = tracks[:, 0]
    data["id"]    = tracks[:, 1]
    data["score"] = tracks[:, 2]
    data["x1"]    = tracks[:, 3]
    data["y1"]    = tracks[:, 4]
    data["x2"]    = tracks[:, 5]
    data["y2"]    = tracks[:, 6]

    track_df = pd.DataFrame.from_dict(data)
    det_df = df = pd.read_pickle(args.DetectionPkl)
    class_labels = []

    frames = np.unique(track_df["fn"])

    for fn in tqdm(frames):
        scores = compute_pairwise_iou(track_df[track_df["fn"] == fn], det_df[det_df["fn"] == fn])
        costs = 1 - scores
        row_indices, col_indices = scipy.optimize.linear_sum_assignment(costs)
        class_labels += list(det_df.iloc[col_indices]["class"])
    
    # add class as a column to df
    track_df["class"] = class_labels

    # write class to txt file
    df_txt(track_df, args.TrackingPth)

def setup(args):
    env_name = args.Tracker
    src_url = "https://github.com/ifzhang/ByteTrack.git"
    rep_path = "./Trackers/ByteTrack/ByteTrack"
    if not "ByteTrack" in os.listdir("./Trackers/ByteTrack/"):
        # clone source repo
        os.system(f"git clone --recurse-submodules {src_url} {rep_path}")
     
    if not env_name in get_conda_envs():
        make_conda_env(env_name, libs="python=3.8 pip")
        os.system(f"conda install -n {args.Tracker} pytorch torchvision cudatoolkit -c pytorch -y")
        cwd = os.getcwd()
        os.chdir(rep_path)
        os.system(f"conda run --live-stream -n {args.Tracker} pip install -r requirements.txt")
        os.system(f"conda run --live-stream -n {args.Tracker} pip install numpy==1.23.5")
        os.system(f"conda run --live-stream -n {args.Tracker} python3 setup.py develop")
        os.system(f"conda run --live-stream -n {args.Tracker} pip install cython")
        os.system(f"conda run --live-stream -n {args.Tracker} pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'")
        os.system(f"conda run --live-stream -n {args.Tracker} pip install cython_bbox")
        os.system(f"conda run --live-stream -n {args.Tracker} pip install protobuf==3.20.3")

        os.chdir(cwd)