from Libs import *
from Utils import *

def track(args, *oargs):
    setup(args)
    env_name = args.Tracker
    exec_path = "./Trackers/ByteTrack/run.py"
    conda_pyrun(env_name, exec_path, args)

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
    return pd.DataFrame.from_dict(data)


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
        os.system(f"conda run -n {args.Tracker} pip install -r requirements.txt")
        os.system(f"conda run -n {args.Tracker} python3 setup.py develop")
        os.system(f"conda run -n {args.Tracker} pip install cython")
        os.system(f"conda run -n {args.Tracker} pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'")
        os.system(f"conda run -n {args.Tracker} pip install cython_bbox")

        os.chdir(cwd)