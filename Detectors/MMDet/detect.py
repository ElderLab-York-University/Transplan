from Libs import *
from Utils import *

def detect(args):
    raise NotImplemented
    # this detector is an abstract detector
    # it plays as a wraper for all detectors under mmdet

def setup(args):
    env_name = "MMDet"
    src_url = "https://github.com/open-mmlab/mmdetection.git"
    rep_path = "./Detectors/MMDet/mmdetection"

    if not "mmdetection" in os.listdir("./Detectors/MMDet/"):
        os.system(f"git clone {src_url} {rep_path}")

    if not env_name in get_conda_envs():
        initial_directory = os.getcwd()
        make_conda_env(env_name, libs="python=3.8")
        # install pytorch
        os.system(f"conda run -n {env_name} --live-stream conda install pip")
        os.system(f"conda run -n {env_name} --live-stream pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113  -f https://download.pytorch.org/whl/torch_stable.html")
        # os.system(f"conda run -n {env_name} --live-stream conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
        os.system(f"conda run -n {env_name} --live-stream pip install -U openmim")
        os.system(f"conda run -n {env_name} --live-stream mim install mmengine")
        os.system(f"conda run -n {env_name} --live-stream mim install mmcv>=2.0.0")
        os.chdir(rep_path)
        os.system(f"conda run -n {env_name} --live-stream pip install -v -e .")

        # verify correct installation
        os.system(f"conda run -n {env_name} --live-stream mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .")
        os.system(f"conda run -n {env_name} --live-stream python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu")

        os.chdir(initial_directory)

def df(args):
  file_path = args.DetectionDetectorPath
  data = {}
  data["fn"], data["class"], data["score"], data["x1"], data["y1"], data["x2"], data["y2"] = [], [], [], [], [], [], []
  with open(file_path, "r+") as f:
    lines = f.readlines()
    for line in lines:
      splits = line.split()
      fn , clss, score, x1, y1, x2, y2 = float(splits[0]), float(splits[1]), float(splits[2]), float(splits[3]), float(splits[4]), float(splits[5]), float(splits[6])
      data["fn"].append(fn)
      data["class"].append(clss)
      data["score"].append(score)
      data["x1"].append(x1)
      data["y1"].append(y1)
      data["x2"].append(x2)
      data["y2"].append(y2)
  return pd.DataFrame.from_dict(data)

def df_txt(df,text_result_path):
  # store a modified version of detection df to the same txt file
  # used in the post processig part of the detection
  # df is in the same format specified in the df function
  with open(text_result_path, "w") as text_file:
    pass

  with open(text_result_path, "w") as text_file:
    for i, row in tqdm(df.iterrows()):
      frame_num, clss, score, x1, y1, x2, y2 = row["fn"], row['class'], row["score"], row["x1"], row["y1"], row["x2"], row["y2"]
      text_file.write(f"{frame_num} {clss} {score} {x1} {y1} {x2} {y2}\n")