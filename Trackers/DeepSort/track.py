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
        os.system(f"conda run -n {args.Tracker} pip3 install tensorflow==1.15.5 opencv-python numpy scikit-learn==0.22.2 tqdm")
        os.system(f"conda run -n {args.Tracker} pip3 install wheel")
        os.system(f"conda run -n {args.Tracker} pip3 install pickle5 pandas")
        print("______++++++++++++________")
