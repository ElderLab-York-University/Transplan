from Libs import *
from Utils import *
def track(args, *oargs):
    setup(args)
    env_name = args.Tracker
    exec_path = "./Trackers/DeepSort/run.py"
    conda_pyrun(env_name, exec_path, args)


def df(args):
    # fn,id,class,score,bbox(4 numbers)
    data = {}
    tracks_path = args.TrackingPth
    tracks = np.loadtxt(tracks_path, delimiter=',')
    data["fn"]    = tracks[:, 0]
    data["id"]    = tracks[:, 1]
    data["class"] = tracks[:, 2]
    data["score"] = tracks[:, 3]
    data["x1"]    = tracks[:, 4]
    data["y1"]    = tracks[:, 5]
    data["x2"]    = tracks[:, 6]
    data["y2"]    = tracks[:, 7]
    return pd.DataFrame.from_dict(data)


def setup(args):
    env_name = args.Tracker
    src_url = "https://github.com/nwojke/deep_sort.git"
    rep_path = "./Trackers/DeepSort/DeepSort"
    if not "DeepSort" in os.listdir("./Trackers/DeepSort/"):
        # clone source repo
        os.system(f"git clone --recurse-submodules {src_url} {rep_path}")
    if not env_name in get_conda_envs():
        make_conda_env(env_name, libs="python=3.6")
