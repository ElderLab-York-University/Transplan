from re import U
from Libs import *
from Utils import *

def track(args, *oargs):
    setup(args)
    env_name = args.Tracker
    exec_path = "./Trackers/FairMot/run.py"
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


def setup(args):
    env_name = args.Tracker
    src_url = "https://github.com/ifzhang/FairMOT.git"
    rep_path = "./Trackers/FairMot/FairMot"
    if not "FairMot" in os.listdir("./Trackers/FairMot/"):
        # clone source repo
        os.system(f"git clone --recurse-submodules {src_url} {rep_path}")

        url = 'https://drive.google.com/uc?export=download&id=1iqRQjsG9BawIl8SlFomMg5iwkb6nqSpi&confirm=t&uuid=95d59101-1ad7-40ae-af82-17dff13805b0'
        os.system("mkdir ./Trackers/FairMot/FairMot/models/")
        d_path = './Trackers/FairMot/FairMot/models/fairmot_dla34.pth'
        download_url_to(url, d_path)

        url="https://drive.google.com/u/0/uc?id=1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT&export=download"
        d_path = './Trackers/FairMot/FairMot/models/ctdet_coco_dla_2x.pth'
        download_url_to(url, d_path)

        url="https://drive.google.com/uc?id=1i8WBqrOiX6e9qRb_6NATzUsjGtiiy3vq&export=download"
        d_path = './Trackers/FairMot/FairMot/models/hrnetv2_w18_imagenet_pretrained.pth'
        download_url_to(url, d_path)

        url="https://drive.google.com/uc?id=1ohYqJSGEJII8EZNMRrn7A31tidu05Czx&export=download&confirm=t&uuid=54ec27e2-3f5a-4196-8fa0-0621b2176dd3"
        d_path = './Trackers/FairMot/FairMot/models/hrnetv2_w32_imagenet_pretrained.pth'
        download_url_to(url, d_path)
        
        url="https://docs.google.com/uc?export=download&id=1udpOPum8fJdoEQm6n0jsIgMMViOMFinu&confirm=t&uuid=5631bc0e-c307-468f-9f24-18a9cad55022"
        d_path="./Trackers/FairMot/FairMot/models/all_dla34.pth"
        download_url_to(url, d_path)


        

    
    
    if not env_name in get_conda_envs():
        make_conda_env(env_name, libs="python=3.8")
        os.system(f"conda install -n {args.Tracker} pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch")
        os.system(f"conda run -n {args.Tracker} pip3 install cython")
        os.system(f"conda run -n {args.Tracker} pip3 install pickle5 pandas")
        os.system(f"conda run -n {args.Tracker} pip3 install -r ./Trackers/FairMot/FairMot/requirements.txt")
        os.system(f"git clone -b pytorch_1.7 https://github.com/ifzhang/DCNv2.git ./Trackers/FairMot/FairMot/DCNv2")
        os.system(f"cd ./Trackers/FairMot/FairMot/DCNv2 \n conda run -n {args.Tracker} ./make.sh")
        # os.system(f"python3 ./Trackers/FairMot/FairMot/DCNv2/setup.py build develop")

        print("______++++++++++++________")
