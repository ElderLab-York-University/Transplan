from Libs import *
from Utils import *

def detect(args):
    raise NotImplementedError
    # this detector is an abstract detector
    # it plays as a wraper for all detectors under mmdet

def setup(args):
    env_name = "MMDet3D"
    src_url = "https://github.com/open-mmlab/mmdetection3d"
    rep_path = "./Detectors/MMDet3D/mmdetection3d"

    if not "mmdetection3d" in os.listdir("./Detectors/MMDet3D/"):
        os.system(f"git clone {src_url} {rep_path}")

    if not env_name in get_conda_envs():
        initial_directory = os.getcwd()
        make_conda_env(env_name, libs="python=3.8")
        # make sure pip is installed
        os.system(f"conda run -n {env_name} --live-stream conda install pip")
        # install pytorch 1.8+
        os.system(f"conda run -n {env_name} --live-stream conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y")
        os.system(f"conda run -n {env_name} --live-stream conda install -c nvidia cuda-nvcc=11.3 -y")  
        # install mm dependancy
        os.system(f"conda run -n {env_name} --live-stream pip install -U openmim")
        os.system(f"conda run -n {env_name} --live-stream mim install mmengine")
        os.system(f"conda run -n {env_name} --live-stream mim install 'mmcv>=2.0.0rc4'")
        os.system(f"conda run -n {env_name} --live-stream mim install 'mmdet>=3.0.0'")
        # install mmdet3D from source code
        os.chdir(rep_path)
        os.system(f"conda run -n {env_name} --live-stream pip install -v -e .")
        # verify correct installation
        os.system(f"conda run -n {env_name} --live-stream mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car --dest .")
        os.system(f"conda run -n {env_name} --live-stream python demo/pcd_demo.py demo/data/kitti/000008.bin pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth --show")

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

def df_3D(args):
  file_path = args.Detection3DPath
  data = {}
  data["fn"], data["class"], data["score"], data["x1"], data["y1"], data["x2"], data["y2"], \
  data["x3"], data["y3"], data["x4"], data["y4"], \
  data["x5"], data["y5"], data["x6"], data["y6"], \
  data["x7"], data["y7"], data["x8"], data["y8"], = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
  with open(file_path, "r+") as f:
    lines = f.readlines()
    for line in lines:
      splits = line.split()
      fn , clss, score, x1, y1, x2, y2, x3,y3, x4,y4, x5,y5,x6,y6,x7,y7,x8,y8 = float(splits[0]), float(splits[1]), float(splits[2]), float(splits[3]), float(splits[4]), float(splits[5]), float(splits[6]), float(splits[7]) , float(splits[8]) , float(splits[9]) ,float(splits[10]), float(splits[11]) ,float(splits[12]), float(splits[13]), float(splits[14]), float(splits[15]), float(splits[16]), float(splits[17]), float(splits[18])
      data["fn"].append(fn)
      data["class"].append(clss)
      data["score"].append(score)
      data["x1"].append(x1)
      data["y1"].append(y1)
      data["x2"].append(x2)
      data["y2"].append(y2)
      data["x3"].append(x3)
      data["y3"].append(y3)
      data["x4"].append(x4)
      data["y4"].append(y4)
      data["x5"].append(x5)
      data["y5"].append(y5)
      data["x6"].append(x6)
      data["y6"].append(y6)
      data["x7"].append(x7)
      data["y7"].append(y7)
      data["x8"].append(x8)
      data["y8"].append(y8)


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

# def fine_tune(train_config_path, work_dir, resume):
#   raise NotImplementedError
#   env_name = "MMDet "
#   ngpus = get_numgpus_torch(env_name)
#   port  = get_available_port()
#   if resume:
#     os.system(f"conda run -n {env_name} --live-stream PORT={port} ./Detectors/MMDet/mmdetection/tools/dist_train.sh {train_config_path} {ngpus} --work-dir={work_dir} --resume")
#   else:
#     os.system(f"conda run -n {env_name} --live-stream PORT={port} ./Detectors/MMDet/mmdetection/tools/dist_train.sh {train_config_path} {ngpus} --work-dir={work_dir}")

# def modify_train_config(train_config_path, args, args_mp, args_gt, args_mp_gt):
#   raise NotImplementedError
#   file_name = train_config_path
#   root_dataset = os.path.abspath(args_gt.Dataset)
#   for arg_p in args_mp_gt:
#     if arg_p.SubID == args_gt.TrainPart:
#       arg_p_train = arg_p
#     if arg_p.SubID == args_gt.ValidPart:
#       arg_p_valid = arg_p

#   train_ann_file = os.path.relpath(arg_p_train.DetectionCOCO, root_dataset)
#   valid_ann_file = os.path.relpath(arg_p_valid.DetectionCOCO, root_dataset)

#   train_data_prefix = arg_p_train.SubID
#   valid_data_prefix = arg_p_valid.SubID

#   with open(arg_p_train.DetectionCOCO, "r") as f:
#     ann = json.load(f)
#     num_classes = len(ann["categories"])
#     classes = tuple([d["name"] for d in ann["categories"]])  

#   # create the new content
#   NewLine = "\n"
#   new_content = ""
#   new_content += f"data_root = '{root_dataset}/'" + NewLine
#   new_content += f"train_ann_file = '{train_ann_file}'" + NewLine
#   new_content += f"train_data_prefix = '{train_data_prefix}/'" + NewLine
#   new_content += f"valid_ann_file = '{valid_ann_file}'" + NewLine
#   new_content += f"valid_data_prefix = '{valid_data_prefix}/'" + NewLine
#   new_content += f"BatchSize = {args_gt.BatchSize}" + NewLine
#   new_content += f"NumWorkers = {args_gt.NumWorkers}" + NewLine
#   new_content += f"Epochs = {args_gt.Epochs}" + NewLine
#   new_content += f"ValInterval = {args_gt.ValInterval}" + NewLine
#   new_content += f"num_classes = {num_classes}" + NewLine
#   new_content += f"classes = {classes}" + NewLine

#   # Read the content of the file
#   with open(file_name, 'r') as file:
#       lines = file.readlines()

#   # Find the indices of the comment lines that mark the section to change
#   start_index = None
#   end_index = None
#   for i, line in enumerate(lines):
#       if line.strip() == "#CHANGE#BELOW#":
#           start_index = i + 1
#       elif line.strip() == "#CHANGE#ABOVE#":
#           end_index = i
#           break

#   # Modify the content between the comment lines
#   new_lines = lines[:start_index]
#   new_lines += new_content
#   new_lines += lines[end_index:]

#   # Write the modified content back to the file
#   with open(file_name, 'w') as file:
#       file.writelines(new_lines)