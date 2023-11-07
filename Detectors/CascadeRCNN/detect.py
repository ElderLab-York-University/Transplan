from Libs import *
from Utils import *
from Detectors.MMDet.detect import setup as mm_setup
from Detectors.MMDet.detect import df as mm_df
from Detectors.MMDet.detect import df_txt as mm_df_txt
from Detectors.MMDet.detect import fine_tune as mm_fine_tune
from Detectors.MMDet.detect import modify_train_config as mm_modify_train_config

def detect(args, *oargs):
    env_name        = "MMDet"
    exec_path       = "./Detectors/MMDet/run.py"
    config_file     = "./Detectors/MMDet/mmdetection/configs/cascade_rcnn/cascade-rcnn_x101-64x4d_fpn_1x_coco.py"
    checkpoint_file = "./Detectors/MMDet/mmdetection/checkpoints/cascade_rcnn_x101_64x4d_fpn_1x_coco_20200515_075702-43ce6a30.pth"

    setup(args)
    args.MMDetConfig = config_file 
    args.MMDetCheckPoint = checkpoint_file
    conda_pyrun(env_name, exec_path, args)

def df(args):
    return mm_df(args)

def df_txt(df,text_result_path):
    return mm_df_txt(df,text_result_path)

def setup(args):
    checkpoint_file = "./Detectors/MMDet/mmdetection/checkpoints/cascade_rcnn_x101_64x4d_fpn_1x_coco_20200515_075702-43ce6a30.pth"
    checkpoint_url  = "https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco/cascade_rcnn_x101_64x4d_fpn_1x_coco_20200515_075702-43ce6a30.pth"
    checkpoints_dir = "./Detectors/MMDet/mmdetection/checkpoints"

    mm_setup(args)

    if not os.path.isdir(checkpoints_dir):
        os.system(f"mkdir {checkpoints_dir}")

    if not os.path.isfile(checkpoint_file):
        print(f"downloading checkpoint: {checkpoint_url}\n storing to: {checkpoint_file}")
        download_url_to(checkpoint_url, checkpoint_file)

def fine_tune(args, args_mp, args_gt, args_mp_gt):
    # modify config file with args
    train_config_path = "./Detectors/CascadeRCNN/train_config.py"
    mm_modify_train_config(train_config_path, args, args_mp, args_gt, args_mp_gt)

    work_dir = args.DetectorCheckPointDir
    if not os.path.isdir(work_dir):
        os.system(f"mkdir {work_dir}")

    mm_fine_tune(train_config_path, work_dir, args.Resume)