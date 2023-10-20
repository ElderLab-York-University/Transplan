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
        os.system(f"conda run --live-stream -n {env_name} conda install pip -y")
        os.system(f"conda run --live-stream -n {env_name} conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y")
        os.system(f"conda run --live-stream -n {env_name} conda install -c nvidia cuda-nvcc=11.3 -y")  
        os.system(f"conda run --live-stream -n {env_name} pip install easydict llvmlite numba pyyaml tqdm")
        os.system(f"conda run --live-stream -n {env_name} pip install openmim")
        os.system(f"conda run --live-stream -n {env_name} mim install mmcv-full==1.5.0")
        os.system(f"conda run --live-stream -n {env_name} pip install timm==0.6.11 mmdet==2.28.1")
        os.system(f"conda run --live-stream -n {env_name} pip install opencv-python termcolor yacs pyyaml scipy tqdm")
        os.system(f"conda run --live-stream -n {env_name} wget -c https://github.com/OpenGVLab/InternImage/releases/download/whl_files/DCNv3-1.0+cu113torch1.11.0-cp37-cp37m-linux_x86_64.whl\
                -O  ./Segmenters/InternImage/InternImage/DCNv3-1.0+cu113torch1.11.0-cp37-cp37m-linux_x86_64.whl")
        os.system(f"conda run --live-stream -n {env_name} pip install ./Segmenters/InternImage/InternImage/DCNv3-1.0+cu113torch1.11.0-cp37-cp37m-linux_x86_64.whl")
        os.system(f"conda run --live-stream -n {env_name} python3 ./Segmenters/InternImage/InternImage/detection/ops_dcnv3/test.py")
        os.system(f"conda run --live-stream -n {env_name} pip install pandas==1.1.5")