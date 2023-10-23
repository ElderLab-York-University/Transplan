from Libs import *
from Utils import *
from Detectors.MMDet.detect import setup as mm_setup
from Detectors.MMDet.detect import df as mm_df
from Detectors.MMDet.detect import df_txt as mm_df_txt

def detect(args, *oargs):
    env_name        = "MMDet"
    exec_path       = "./Detectors/MMDet/run.py"
    config_file     = "./Detectors/MMDet/mmdetection/configs/centernet/centernet-update_r50-caffe_fpn_ms-1x_coco.py"
    checkpoint_file = "./Detectors/MMDet/mmdetection/checkpoints/centernet-update_r50-caffe_fpn_ms-1x_coco_20230512_203845-8306baf2.pth"

    setup(args)
    args.MMDetConfig = config_file 
    args.MMDetCheckPoint = checkpoint_file
    conda_pyrun(env_name, exec_path, args)

def df(args):
    return mm_df(args)

def df_txt(df,text_result_path):
    return mm_df_txt(df,text_result_path)

def setup(args):
    checkpoint_file = "./Detectors/MMDet/mmdetection/checkpoints/centernet-update_r50-caffe_fpn_ms-1x_coco_20230512_203845-8306baf2.pth"
    checkpoint_url  = "https://download.openmmlab.com/mmdetection/v3.0/centernet/centernet-update_r50-caffe_fpn_ms-1x_coco/centernet-update_r50-caffe_fpn_ms-1x_coco_20230512_203845-8306baf2.pth"
    checkpoints_dir = "./Detectors/MMDet/mmdetection/checkpoints"

    mm_setup(args)

    if not os.path.isdir(checkpoints_dir):
        os.system(f"mkdir {checkpoints_dir}")

    if not os.path.isfile(checkpoint_file):
        print(f"downloading checkpoint: {checkpoint_url}\n storing to: {checkpoint_file}")
        download_url_to(checkpoint_url, checkpoint_file)