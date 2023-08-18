from Libs import *
from Utils import *

def track(args, *oargs):
    setup(args)
    env_name = args.Tracker
    exec_path = "./Trackers/StrongSort/run.py"
    conda_pyrun(env_name, exec_path, args)
    match_classes(args)

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
            print('%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f'%(fn, idd, score, x1, y1, x2, y2, clss),file=out_file)
    # df = pd.read_pickle(args.TrackingPkl)
    # out_path = args.TrackingPth 

def match_classes(args):
    # already hass classes in txt
    pass

def setup(args):
    env_name = args.Tracker
    src_url = "https://github.com/mikel-brostrom/yolov8_tracking.git "
    rep_path = "./Trackers/StrongSort/strongsort"
    if not "strongsort" in os.listdir("./Trackers/StrongSort/"):
        # clone source repo
        os.system(f"git clone --recurse-submodules {src_url} {rep_path}")
     
    if not env_name in get_conda_envs():
        make_conda_env(env_name, libs="python=3.8 pip")
        cwd = os.getcwd()
        os.chdir(rep_path)
        os.system(f"conda run -n {args.Tracker} --live-stream pip install -r requirements.txt")
        # download re-id weights
        # list of available re-ID models here 
        # https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO
        try:
            os.system(f"mkdir weights")
        except: pass
        # os.chdir("weights")
        # url = 'https://drive.google.com/uc?id=1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA&export=download'
        # d_path = './osnet_x1_0.pt'
        # download_url_to(url, d_path)
        os.chdir(cwd)


        # os.system(f"conda install -n {args.Tracker} pytorch torchvision cudatoolkit -c pytorch -y")
        # os.system(f"conda run -n {args.Tracker} --live-stream python3 setup.py develop")
        # os.system(f"conda run -n {args.Tracker} --live-stream pip install cython")
        # os.system(f"conda run -n {args.Tracker} --live-stream pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'")
        # os.system(f"conda run -n {args.Tracker} --live-stream pip install cython_bbox")
