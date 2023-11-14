from Libs import *
from Utils import *

def track(args, *oargs):
    setup(args)
    env_name = args.Tracker
    exec_path = "./Trackers/DeepSort/run.py"
    conda_pyrun(env_name, exec_path, args)
    match_classes(args)


def df(args):
    # fn,id,class,score,bbox(4 numbers)
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
    data["fn"]    = tracks[:, 0]
    data["id"]    = tracks[:, 1]
    data["x1"]    = tracks[:, 2]
    data["y1"]    = tracks[:, 3]
    data["x2"]    = tracks[:, 4]
    data["y2"]    = tracks[:, 5]

    track_df = pd.DataFrame.from_dict(data)
    det_df = pd.read_pickle(args.DetectionPkl)
    class_labels = []

    uni_frames = np.unique(track_df["fn"])

    for fn in tqdm(uni_frames):
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


def setup(args):
    env_name = args.Tracker
    src_url = "https://github.com/nwojke/deep_sort.git"
    rep_path = "./Trackers/DeepSort/DeepSort"
    if not "DeepSort" in os.listdir("./Trackers/DeepSort/"):
        # clone source repo
        os.system(f"git clone --recurse-submodules {src_url} {rep_path}")

        url = 'https://drive.google.com/uc?id=1bB66hP9voDXuoBoaCcKYY7a8IYzMMs4P&export=download'
        os.system("mkdir ./Trackers/DeepSort/DeepSort/models/")
        d_path = './Trackers/DeepSort/DeepSort/models/mars-small128.pb'
        download_url_to(url, d_path)
    
    
    if not env_name in get_conda_envs():
        make_conda_env(env_name, libs="python=3.6")
        os.system(f"conda run --live-stream -n {args.Tracker} conda install pip")
        os.system(f"conda run --live-stream -n {args.Tracker} pip install --upgrade pip setuptools wheel")
        os.system(f"conda run --live-stream -n {args.Tracker} conda install -c anaconda cudatoolkit=10.0 cudnn=7.6.0 -y")
        os.system(f"conda run --live-stream -n {args.Tracker} pip install tensorflow_gpu==1.15.0 opencv-python==4.6.0.66 numpy scikit-learn==0.22.2 tqdm")
        os.system(f"conda run --live-stream -n {args.Tracker} pip install pickle5 pandas")

