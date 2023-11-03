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

        # os.system(f"conda run --live-stream -n {env_name} conda install pytorch torchvision -c pytorch -y")

        os.system(f"conda run --live-stream -n {env_name} conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y")
        os.system(f"conda run --live-stream -n {env_name} conda install -c nvidia cuda-nvcc=11.3 -y")  
        # os.system(f"conda run -n {env_name} --live-stream pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113  -f https://download.pytorch.org/whl/torch_stable.html")
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

def fine_tune(train_config_path, work_dir, resume):
  env_name = "MMDet "
  ngpus = get_numgpus_torch(env_name)
  if resume:
    os.system(f"conda run -n {env_name} --live-stream ./Detectors/MMDet/mmdetection/tools/dist_train.sh {train_config_path} {ngpus} --work-dir={work_dir} --resume")
  else:
    os.system(f"conda run -n {env_name} --live-stream ./Detectors/MMDet/mmdetection/tools/dist_train.sh {train_config_path} {ngpus} --work-dir={work_dir}")

def modify_train_config(train_config_path, args, args_mp, args_gt, args_mp_gt):
  file_name = train_config_path
  root_dataset = os.path.abspath(args_gt.Dataset)
  for arg_p in args_mp_gt:
    if arg_p.SubID == args_gt.TrainPart:
      arg_p_train = arg_p
    if arg_p.SubID == args_gt.ValidPart:
      arg_p_valid = arg_p

  train_ann_file = os.path.relpath(arg_p_train.DetectionCOCO, root_dataset)
  valid_ann_file = os.path.relpath(arg_p_valid.DetectionCOCO, root_dataset)

  train_data_prefix = arg_p_train.SubID
  valid_data_prefix = arg_p_valid.SubID

  with open(arg_p_train.DetectionCOCO, "r") as f:
    ann = json.load(f)
    num_classes = len(ann["categories"])
    classes = tuple([d["name"] for d in ann["categories"]])  

  # create the new content
  NewLine = "\n"
  new_content = ""
  new_content += f"data_root = '{root_dataset}/'" + NewLine
  new_content += f"train_ann_file = '{train_ann_file}'" + NewLine
  new_content += f"train_data_prefix = '{train_data_prefix}/'" + NewLine
  new_content += f"valid_ann_file = '{valid_ann_file}'" + NewLine
  new_content += f"valid_data_prefix = '{valid_data_prefix}/'" + NewLine
  new_content += f"BatchSize = {args_gt.BatchSize}" + NewLine
  new_content += f"NumWorkers = {args_gt.NumWorkers}" + NewLine
  new_content += f"Epochs = {args_gt.Epochs}" + NewLine
  new_content += f"ValInterval = {args_gt.ValInterval}" + NewLine
  new_content += f"num_classes = {num_classes}" + NewLine
  new_content += f"classes = {classes}" + NewLine

  # Read the content of the file
  with open(file_name, 'r') as file:
      lines = file.readlines()

  # Find the indices of the comment lines that mark the section to change
  start_index = None
  end_index = None
  for i, line in enumerate(lines):
      if line.strip() == "#CHANGE#BELOW#":
          start_index = i + 1
      elif line.strip() == "#CHANGE#ABOVE#":
          end_index = i
          break

  # Modify the content between the comment lines
  new_lines = lines[:start_index]
  new_lines += new_content
  new_lines += lines[end_index:]

  # Write the modified content back to the file
  with open(file_name, 'w') as file:
      file.writelines(new_lines)