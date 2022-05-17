from Libs import *
from Utils import *

def track(args, *oargs):
    setup(args)
    env_name = args.Tracker
    exec_path = "./Trackers/CenterTrack/run.py"
    conda_pyrun(env_name, exec_path, args)

def df(args):
    raise NotImplemented

def setup(args):
    env_name = args.Tracker
    src_url = "https://github.com/xingyizhou/CenterTrack.git"
    rep_path = "./Trackers/CenterTrack/CenterTrack"
    if not "CenterTrack" in os.listdir("./Trackers/CenterTrack/"):
        # clone source repo
        os.system(f"git clone --recurse-submodules {src_url} {rep_path}")

        # clone submodules cause -r is not working
        os.system(f"git clone  https://github.com/nutonomy/nuscenes-devkit {rep_path}/src/tools/nuscenes-devkit")
        os.system(f"git clone  https://github.com/nutonomy/nuscenes-devkit {rep_path}/src/tools/nuscenes-devkit-alpha02")
        os.system(f"git clone https://github.com/CharlesShang/DCNv2/  {rep_path}/src/lib/model/networks/DCNv2")
    
        # download COCO weights
        os.system("mkdir ./Trackers/CenterTrack/CenterTrack/models/")
        url = 'https://drive.google.com/uc?id=1tJCEJmdtYIh8VuN8CClGNws3YO7QGd40&export=download'
        d_path = './Trackers/CenterTrack/CenterTrack/models/coco_tracking.pth'
        download_url_to(url, d_path)

    if not env_name in get_conda_envs():
        make_conda_env(env_name, libs="python=3.6")
        # install library on conda env
        print("here I am 1")
        os.system(f"conda install -n {args.Tracker} pytorch=1.4 torchvision cudatoolkit=10.0 -c pytorch -y")
        print("here I am 2")
        os.system(f"conda run -n {args.Tracker} pip3 install cython")
        print("here I am 3")
        os.system(f"conda run -n {args.Tracker} pip3 install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'")
        print("here I am 4")
        os.system(f"conda run -n {args.Tracker} pip3 install -r ./Trackers/CenterTrack/CenterTrack/requirements.txt")
        print("I am here 5")
        # setup_path = "./Trackers/CenterTrack/CenterTrack/src/lib/model/networks/DCNv2/setup.py"
        # os.system(f"conda run -n {args.Tracker} python3 {setup_path} build develop")
        cwd = os.getcwd()
        os.chdir("./Trackers/CenterTrack/CenterTrack/src/lib/model/networks/DCNv2")
        os.system(f"conda run -n {args.Tracker} ./make.sh")
        os.chdir(cwd)
        print("after installing DCNv2")