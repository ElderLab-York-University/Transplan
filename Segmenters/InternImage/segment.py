from Libs import *
from Utils import *

def segment(args):
    setup(args)
    env_name = args.Segmenter
    exec_path = "./Segmenters/InternImage/run.py"
    conda_pyrun(env_name, exec_path, args)


def setup(args):
    env_name = args.Segmenter
    src_url = "https://github.com/OpenGVLab/InternImage.git"
    rep_path = "./Segmenters/InternImage/InternImage"
    
    print(env_name)
    if not "InternImage" in os.listdir("./Segmenters/InternImage/"):
      os.system(f"git clone {src_url} {rep_path}")
      if not "checkpoint_dir" in os.listdir("./Segmenters/InternImage/InternImage"):
        os.system(f"mkdir ./Segmenters/InternImage/InternImage/checkpoint_dir")
        os.system(f"wget -c https://github.com/OpenGVLab/InternImage/releases/download/det_model/cascade_internimage_xl_fpn_1x_coco.pth\
        -O ./Segmenters/InternImage/InternImage/checkpoint_dir/cascade_internimage_xl_fpn_1x_coco.pth")

        os.system(f"wget -c  https://huggingface.co/OpenGVLab/InternImage/resolve/main/cascade_internimage_xl_fpn_3x_coco.pth\
        -O ./Segmenters/InternImage/InternImage/checkpoint_dir/cascade_internimage_xl_fpn_3x_coco.pth")

       
    if not env_name in get_conda_envs():
        make_conda_env(env_name, libs="python=3.7")
        print("here I am 1")
        os.system(f"conda run --live-stream -n {env_name} conda install pip")
        os.system(f"conda run --live-stream -n {env_name} pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113  -f https://download.pytorch.org/whl/torch_stable.html")
        print("here I am 1.5")
        os.system(f"conda run --live-stream -n {env_name} pip install easydict llvmlite numba pyyaml tqdm")
        print("here I am 2")
        os.system(f"conda run --live-stream -n {env_name} pip install openmim")
        print("here I am 2.3")
        os.system(f"conda run --live-stream -n {env_name} mim install mmcv-full==1.5.0")
        print("here I am 2.6")
        os.system(f"conda run --live-stream -n {env_name} pip install timm==0.6.11 mmdet==2.28.1")
        print("here I am 3")
        os.system(f"conda run --live-stream -n {env_name} pip install opencv-python termcolor yacs pyyaml scipy tqdm")
        print("here I am 4")
        os.system(f"conda run --live-stream -n {env_name} python3 ./Segmenters/InternImage/InternImage/detection/ops_dcnv3/setup.py build install")        
        os.system(f"conda run --live-stream -n {env_name} python3 ./Segmenters/InternImage/InternImage/detection/ops_dcnv3/test.py")
        os.system(f"conda run --live-stream -n {env_name} pip install pandas==1.1.5")